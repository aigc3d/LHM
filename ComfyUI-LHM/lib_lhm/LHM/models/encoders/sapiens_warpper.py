import functools
import gc
import multiprocessing as mp
import os
import pdb
import time
import traceback as tb
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool, Process, cpu_count
from multiprocessing.pool import Pool
from typing import Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from accelerate.logging import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

timings = {}
BATCH_SIZE = 64


class AsyncWorkerExceptionsWrapper:
    def __init__(self, callable):
        self.__callable = callable
        self._logger = mp.log_to_stderr()

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            self._logger.error(tb.format_exc())
            raise

        # It was fine, give a normal answer
        return result


class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, shape=None, mean=None, std=None):
        self.image_list = image_list
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None

    def __len__(self):
        return len(self.image_list)

    def _preprocess(self, img):
        if self.shape:
            img = cv2.resize(
                img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()  # bgr2rgb
        if self.mean is not None and self.std is not None:
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img

    def __getitem__(self, idx):
        orig_img_dir = self.image_list[idx]
        orig_img = cv2.imread(orig_img_dir)
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img = self._preprocess(orig_img)
        return orig_img_dir, orig_img, img


def warmup_model(model, batch_size):
    # Warm up the model with a dummy input.
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=torch.bfloat16).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s


def inference_model(model, imgs, dtype=torch.bfloat16):
    # forward the model
    with torch.no_grad():
        (results,) = model(imgs.to(dtype).cuda())
        imgs.cpu()

    return results


def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)


def feat_save(feature, output_path):
    pred_save_path = os.path.join(
        output_path.replace(".jpg", ".npy")
        .replace(".jpeg", ".npy")
        .replace(".png", ".npy")
    )
    np.save(pred_save_path, feature)


def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()


class SapiensWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        encoder_feat_dim: int = 384,
        resolution=1024,
        antialias: bool = True,
    ):
        super().__init__()
        self.model = self._build_sapiens(model_name)
        self.resolution = resolution

        self.antialias = antialias
        self.register_buffer(
            "mean", torch.Tensor([0.4844, 0.4570, 0.4062]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.2295, 0.2236, 0.2256]), persistent=False
        )

        if freeze:
            self._freeze()
        else:
            raise NotImplementedError(
                "Fine-tuning is not supported yet."
            )  # sapiens is too larger to finetune the model end-to-end.

    def _preprocess_image(
        self, image: torch.tensor, resolution: int = 1024
    ) -> torch.Tensor:

        _, __, H, W = image.shape
        max_size = max(H, W)
        H_pad = max_size - H
        W_pad = max_size - W
        pad_size = (
            W_pad // 2,
            max_size - (W + W_pad // 2),
            H_pad // 2,
            max_size - (H + H_pad // 2),
            0,
            0,
            0,
            0,
        )

        image = F.pad(image, pad_size, value=1)

        image = kornia.geometry.resize(
            image,
            (resolution, resolution),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        image = kornia.enhance.normalize(image, self.mean, self.std)

        return image

    @staticmethod
    def _build_sapiens(model_name: str, pretrained: bool = True):

        logger.debug(f"Using Sapiens model: {model_name}")
        USE_TORCHSCRIPT = "_torchscript" in model_name

        # build the model from a checkpoint file
        model = load_model(model_name, use_torchscript=USE_TORCHSCRIPT)
        if not USE_TORCHSCRIPT:
            raise NotImplementedError
        else:
            dtype = torch.float32  # TorchScript models use float32
            model = model.cuda()
        return model

    def _freeze(self):
        logger.warning(f"======== Freezing Sapiens Model ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @torch.compile
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, C, H, W]
        # mod: [N, D] or None
        # RGB image with [0,1] scale and properly sized

        image = self._preprocess_image(image, self.resolution)

        # NOTE that, only supports
        patch_h, patch_w = (
            image.shape[-2] // 16,
            image.shape[-1] // 16,
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (out_local,) = self.model(image)

        out_global = None
        if out_global is not None:
            raise NotImplementedError("Global feature is not supported yet.")
        else:
            ret = out_local.permute(0, 2, 3, 1).flatten(1, 2)

        return ret


def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 1024],
        help="input image size (height, width)",
    )

    args = parser.parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    USE_TORCHSCRIPT = "_torchscript" in args.checkpoint

    # build the model from a checkpoint file
    model = load_model(args.checkpoint, use_torchscript=USE_TORCHSCRIPT)
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        model.to(dtype)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        model = model.cuda()

    imgs = torch.randn(2, 3, 1024, 1024).float().cuda()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        (results,) = model(imgs.cuda())

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        model.to(dtype)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        model = model.to(args.device)

    image_names = []

    pdb.set_trace()

    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        valid_images_len = len(batch_imgs)
        batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        results = inference_model(model, batch_imgs, dtype=dtype)
        args_list = [
            (
                feat.cpu().float().numpy(),
                os.path.join(args.output_root, os.path.basename(img_name)),
            )
            for feat, img_name in zip(results[:valid_images_len], batch_image_name)
        ]
        feat_save_pool.run_async(args_list)

    feat_save_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()
