import os
import sys
current_dir_path = os.path.dirname(__file__)
sys.path.append(current_dir_path + "/../ComfyUI-LHM")

import torch
torch._dynamo.config.disable = True
import numpy as np
from PIL import Image
import cv2
import comfy.model_management as model_management
from omegaconf import OmegaConf
import time
import argparse
import uuid
from rembg import remove

from lib_lhm.engine.pose_estimation.pose_estimator import PoseEstimator
from lib_lhm.engine.SegmentAPI.base import Bbox
from lib_lhm.LHM.utils.face_detector import VGGHeadDetector

from lib_lhm.LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    resize_image_keepaspect_np,
    predict_motion_seqs_from_images,
)

from lib_lhm.LHM.utils.hf_hub import wrap_model_hub
from lib_lhm.LHM.utils.ffmpeg_utils import images_to_video
from lib_lhm.engine.SegmentAPI.base import Bbox

# several util funcs we use
def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    # obtain bbox
    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)

    # scale box to 1.05
    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box

def infer_preprocess_image(
    rgb_np,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)
    """

    # rgb = np.array(Image.open(rgb_path))
    rgb = rgb_np[:,:,::-1]
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]

    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard

    target_w = int(min(w * scale_ratio, h))
    offset_w = (target_w - w) // 2
    # resize to target ratio.
    if offset_w > 0:
        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        offset_w = -offset_w 
        rgb = rgb[:,offset_w:-offset_w,:]
        mask = mask[:,offset_w:-offset_w]

    # resize to target ratio.

    rgb = np.pad(
        rgb,
        ((0, 0), (offset_w, offset_w), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    mask = np.pad(
        mask,
        ((0, 0), (offset_w, offset_w)),
        mode="constant",
        constant_values=0,
    )

    rgb = rgb / 255.0  # normalize to [0, 1]
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # resize to specific size require by preprocessor of smplx-estimator.
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    # resize to render_tgt_size for training

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:

        # ******************** Merge *********************** #
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert (
            abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
        ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert (
            abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
        ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"

        # ******************** Merge *********************** #
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 1, H, W]
    return rgb, mask, intr

def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")

    args.config = args.infer if args.config is None else args.config

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train

def _build_model(cfg):
    from .lib_lhm.LHM.models import model_dict

    hf_model_cls = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"])
    model = hf_model_cls.from_pretrained(cfg.model_name)

    return model

class LHMReconstructionNode:
    """
    ComfyUI node for LHM (Large Animatable Human Model) reconstruction.
    
    This node takes an input image and generates:
    1. A processed image with background removal and recentering
    2. An animation sequence based on provided motion data
    3. A 3D mesh of the reconstructed human (optional)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "motion": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    # RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image", "animation_images")
    # RETURN_NAMES = ("processed_image",)
    FUNCTION = "execute"
    CATEGORY = "LHM"
    
    def __init__(self):
        """Initialize the node with empty model and components."""

        os.environ.update({
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": "./models/checkpoints/LHM/exps/releases/video_human_benchmark/human-lrm-500M/step_060000/",
            "APP_INFER": "./custom_nodes/ComfyUI-LHM/lib_lhm/configs/inference/human-lrm-500M.yaml",
            "APP_TYPE": "infer.human_lrm",
            "NUMBA_THREADING_LAYER": 'omp',
        })
        self.LHM_Model_Dict = {}

    def execute(self, input_image, motion):
        """
        Main method to process an input image and generate human reconstruction outputs.
        
        Args:
            input_image: Input image tensor from ComfyUI
            motion_path: Path to the motion sequence data
            
        Returns:
            Tuple of (processed_image, animation_sequence)
        """   
        if 'pose_estimator' not in self.LHM_Model_Dict:
            pose_estimator = PoseEstimator(
                "./models/checkpoints/LHM/pretrained_models/human_model_files/", device='cpu'
            )
            pose_estimator.to('cuda')
            pose_estimator.device = 'cuda'
            self.LHM_Model_Dict['pose_estimator'] = pose_estimator
        
        if 'face_detector' not in self.LHM_Model_Dict:
            facedetector = VGGHeadDetector(
                "./models/checkpoints/LHM/pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
                device='cpu',
            )
            # facedetector
            self.LHM_Model_Dict['face_detector'] = facedetector
        
        if 'cfg' not in self.LHM_Model_Dict:
            cfg, cfg_train = parse_configs()
            self.LHM_Model_Dict['cfg'] = cfg

        
        # print("Load weights Done.")

        # print(motion.shape, motion.max(), motion.dtype)
        print("MOTIONPATH:", motion)

        print("######start to process input#######")

        task_uid = str(uuid.uuid1())

        os.makedirs(os.path.join("./lhm_temp_files", task_uid), exist_ok=True)

        # for idx, motion_img in enumerate(motion):
        #     motion_img = (motion_img.cpu().numpy()*255.0).astype(np.uint8)
        #     Image.fromarray(motion_img).save(os.path.join("./lhm_temp_files", task_uid, 'imgs_png/{:05d}.png'.format(idx)))

        motion_path = None

        # images_path = os.path.join("./lhm_temp_files", task_uid, "imgs_png")

        save_root = os.path.join("./lhm_temp_files", task_uid)

        image_raw = os.path.join("./lhm_temp_files", task_uid, 'raw.png')

        Image.fromarray((input_image.squeeze(0).cpu().numpy()*255.0).astype(np.uint8)).save(image_raw)

        shape_pose = self.LHM_Model_Dict['pose_estimator'](image_raw)

        assert shape_pose.is_full_body, f"The input image is illegal, {shape_pose.msg}"


        # torch.Size([1, 1920, 1440, 3]) <class 'torch.Tensor'> torch.float32 torch.float32 tensor(1.) tensor(0.)
        # print(input_image.shape, type(input_image), input_image.dtype, input_image.max(), input_image.min())

        # remove the background of input image
        input_np = (input_image.squeeze(0).detach().cpu().numpy()[:,:,::-1]*255.0).astype(np.uint8)
        output_np = remove(input_np)
        parsing_mask = output_np[:,:,3]
        aspect_standard = 5.0 / 3

        source_size = self.LHM_Model_Dict['cfg'].source_size
        render_size = self.LHM_Model_Dict['cfg'].render_size
        motion_img_need_mask = self.LHM_Model_Dict['cfg'].get("motion_img_need_mask", False)  # False
        vis_motion = self.LHM_Model_Dict['cfg'].get("vis_motion", False)  # False

        process_image, _, _ = infer_preprocess_image(
            input_np,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )

        # print(process_image.dtype, process_image.max(), process_image.shape)
        # Image.fromarray((process_image[0].permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)).save("./test.png")

        # detect the head
        try:
            rgb = torch.from_numpy(input_np).permute(2,0,1).to('cuda')
            bbox = self.LHM_Model_Dict['face_detector'].detect_face(rgb)
            head_rgb = rgb[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            head_rgb = head_rgb.permute(1, 2, 0)
            src_head_rgb = head_rgb.cpu().numpy()
            print("w head input!")
        except:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)

        # resize to dino size
        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.LHM_Model_Dict['cfg'].src_head_size, self.LHM_Model_Dict['cfg'].src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.LHM_Model_Dict['cfg'].src_head_size, self.LHM_Model_Dict['cfg'].src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]s


        motion_seq = prepare_motion_seqs(
            None,
            motion,
            save_root=save_root, # this is used when we need to extract the motion
            fps=30,
            bg_color=1,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
            vis_motion=vis_motion,
        )
        
        if 'lhm' not in self.LHM_Model_Dict:
            lhm = _build_model(self.LHM_Model_Dict['cfg'])
            lhm.to('cuda')
            self.LHM_Model_Dict['lhm'] = lhm


        camera_size = len(motion_seq["motion_seqs"])
        shape_param = shape_pose.beta

        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        self.LHM_Model_Dict['lhm'].to(dtype)

        smplx_params = motion_seq['smplx_params']
        smplx_params['betas'] = shape_param.to(device)

        gs_model_list, query_points, transform_mat_neutral_pose = self.LHM_Model_Dict['lhm'].infer_single_view(
            process_image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )

        # rendering !!!!
        start_time = time.time()
        batch_dict = dict()
        batch_size = 80  # avoid memeory out!

        for batch_i in range(0, camera_size, batch_size):
            with torch.no_grad():
                # TODO check device and dtype
                # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])
                keys = [
                    "root_pose",
                    "body_pose",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                    "lhand_pose",
                    "rhand_pose",
                    "trans",
                    "focal",
                    "princpt",
                    "img_size_wh",
                    "expr",
                ]
                batch_smplx_params = dict()
                batch_smplx_params["betas"] = shape_param.to(device)
                batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
                for key in keys:
                    batch_smplx_params[key] = motion_seq["smplx_params"][key][
                        :, batch_i : batch_i + batch_size
                    ].to(device)

                res = self.LHM_Model_Dict['lhm'].animation_infer(gs_model_list, query_points, batch_smplx_params,
                    render_c2ws=motion_seq["render_c2ws"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_intrs=motion_seq["render_intrs"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                )

            for accumulate_key in ["comp_rgb", "comp_mask"]:
                if accumulate_key not in batch_dict:
                    batch_dict[accumulate_key] = []
                batch_dict[accumulate_key].append(res[accumulate_key].detach().cpu())
            del res
            torch.cuda.empty_cache()

        for accumulate_key in ["comp_rgb", "comp_mask"]:
            batch_dict[accumulate_key] = torch.cat(batch_dict[accumulate_key], dim=0)

        print(f"time elapsed: {time.time() - start_time}")
        rgb = batch_dict["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask = batch_dict["comp_mask"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask[mask < 0.5] = 0.0

        rgb = rgb * mask + (1 - mask) * 1
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)

            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]))[
                    None, :, :, :
                ],
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate(
                [rgb, motion_seq["vis_motion_render"], vis_ref_img], axis=2
            )

        print(rgb.shape, rgb.max())
        del self.LHM_Model_Dict['lhm'] # avoid memory explosion
        return process_image.permute(0,2,3,1), torch.from_numpy(rgb)/255.0

class LHMMotionNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "execute"
    CATEGORY = "LHM"
    
    def __init__(self):
        """Initialize the node with empty model and components."""
        self.Motion_Model_Dict = {}

    def execute(self, motion):
        
        # Get some params
        print("MOTIONPATH:", motion)

        print("######start to process input#######")

        task_uid = str(uuid.uuid1())
        
        save_root =  os.path.join("./lhm_motion_temp_files", task_uid)

        os.makedirs(save_root, exist_ok=True)

        motion_seqs_dir, image_folder = predict_motion_seqs_from_images(
            motion, save_root
        )

        print(motion_seqs_dir)

        return (motion_seqs_dir,)

class LHMOfflineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "motion": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("processed_image", "animation_images")
    FUNCTION = "execute"
    CATEGORY = "LHM"
    
    def __init__(self):
        """Initialize the node with empty model and components."""

        os.environ.update({
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": "./models/checkpoints/LHM/exps/releases/video_human_benchmark/human-lrm-500M/step_060000/",
            "APP_INFER": "./custom_nodes/ComfyUI-LHM/lib_lhm/configs/inference/human-lrm-500M.yaml",
            "APP_TYPE": "infer.human_lrm",
            "NUMBA_THREADING_LAYER": 'omp',
        })
        self.LHM_Model_Dict = {}

    def execute(self, input_image, motion):
        """
        Main method to process an input image and generate human reconstruction outputs.
        
        Args:
            input_image: Input image tensor from ComfyUI
            motion_path: Path to the motion sequence data
            
        Returns:
            Tuple of (processed_image, animation_sequence)
        """   
        if 'pose_estimator' not in self.LHM_Model_Dict:
            pose_estimator = PoseEstimator(
                "./models/checkpoints/LHM/pretrained_models/human_model_files/", device='cpu'
            )
            pose_estimator.to('cuda')
            pose_estimator.device = 'cuda'
            self.LHM_Model_Dict['pose_estimator'] = pose_estimator
        
        if 'face_detector' not in self.LHM_Model_Dict:
            facedetector = VGGHeadDetector(
                "./models/checkpoints/LHM/pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
                device='cpu',
            )
            # facedetector
            self.LHM_Model_Dict['face_detector'] = facedetector
        
        if 'cfg' not in self.LHM_Model_Dict:
            cfg, cfg_train = parse_configs()
            self.LHM_Model_Dict['cfg'] = cfg

        print("MOTIONPATH:", motion)

        print("######start to process input#######")

        task_uid = str(uuid.uuid1())

        os.makedirs(os.path.join("./lhm_temp_files", task_uid), exist_ok=True)

        motion_path = motion

        save_root = os.path.join("./lhm_temp_files", task_uid)

        image_raw = os.path.join("./lhm_temp_files", task_uid, 'raw.png')

        Image.fromarray((input_image.squeeze(0).cpu().numpy()*255.0).astype(np.uint8)).save(image_raw)

        shape_pose = self.LHM_Model_Dict['pose_estimator'](image_raw)

        assert shape_pose.is_full_body, f"The input image is illegal, {shape_pose.msg}"

        # remove the background of input image
        input_np = (input_image.squeeze(0).detach().cpu().numpy()[:,:,::-1]*255.0).astype(np.uint8)
        output_np = remove(input_np)
        parsing_mask = output_np[:,:,3]
        aspect_standard = 5.0 / 3

        source_size = self.LHM_Model_Dict['cfg'].source_size
        render_size = self.LHM_Model_Dict['cfg'].render_size
        motion_img_need_mask = self.LHM_Model_Dict['cfg'].get("motion_img_need_mask", False)  # False
        vis_motion = self.LHM_Model_Dict['cfg'].get("vis_motion", False)  # False

        process_image, _, _ = infer_preprocess_image(
            input_np,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )

        # detect the head
        try:
            rgb = torch.from_numpy(input_np).permute(2,0,1).to('cuda')
            bbox = self.LHM_Model_Dict['face_detector'].detect_face(rgb)
            head_rgb = rgb[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            head_rgb = head_rgb.permute(1, 2, 0)
            src_head_rgb = head_rgb.cpu().numpy()
            print("w head input!")
        except:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)

        # resize to dino size
        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.LHM_Model_Dict['cfg'].src_head_size, self.LHM_Model_Dict['cfg'].src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.LHM_Model_Dict['cfg'].src_head_size, self.LHM_Model_Dict['cfg'].src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]s


        motion_seq = prepare_motion_seqs(
            motion_path,
            None,
            save_root=save_root, # this is used when we need to extract the motion
            fps=30,
            bg_color=1,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
            vis_motion=vis_motion,
        )
        
        if 'lhm' not in self.LHM_Model_Dict:
            lhm = _build_model(self.LHM_Model_Dict['cfg'])
            lhm.to('cuda')
            self.LHM_Model_Dict['lhm'] = lhm


        camera_size = len(motion_seq["motion_seqs"])
        shape_param = shape_pose.beta

        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        self.LHM_Model_Dict['lhm'].to(dtype)

        smplx_params = motion_seq['smplx_params']
        smplx_params['betas'] = shape_param.to(device)

        gs_model_list, query_points, transform_mat_neutral_pose = self.LHM_Model_Dict['lhm'].infer_single_view(
            process_image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )

        # rendering !!!!
        start_time = time.time()
        batch_dict = dict()
        batch_size = 80  # avoid memeory out!

        for batch_i in range(0, camera_size, batch_size):
            with torch.no_grad():
                # TODO check device and dtype
                # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])
                keys = [
                    "root_pose",
                    "body_pose",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                    "lhand_pose",
                    "rhand_pose",
                    "trans",
                    "focal",
                    "princpt",
                    "img_size_wh",
                    "expr",
                ]
                batch_smplx_params = dict()
                batch_smplx_params["betas"] = shape_param.to(device)
                batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
                for key in keys:
                    batch_smplx_params[key] = motion_seq["smplx_params"][key][
                        :, batch_i : batch_i + batch_size
                    ].to(device)

                res = self.LHM_Model_Dict['lhm'].animation_infer(gs_model_list, query_points, batch_smplx_params,
                    render_c2ws=motion_seq["render_c2ws"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_intrs=motion_seq["render_intrs"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                )

            for accumulate_key in ["comp_rgb", "comp_mask"]:
                if accumulate_key not in batch_dict:
                    batch_dict[accumulate_key] = []
                batch_dict[accumulate_key].append(res[accumulate_key].detach().cpu())
            del res
            torch.cuda.empty_cache()

        for accumulate_key in ["comp_rgb", "comp_mask"]:
            batch_dict[accumulate_key] = torch.cat(batch_dict[accumulate_key], dim=0)

        print(f"time elapsed: {time.time() - start_time}")
        rgb = batch_dict["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask = batch_dict["comp_mask"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask[mask < 0.5] = 0.0

        rgb = rgb * mask + (1 - mask) * 1
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)

            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]))[
                    None, :, :, :
                ],
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate(
                [rgb, motion_seq["vis_motion_render"], vis_ref_img], axis=2
            )
        return process_image.permute(0,2,3,1), torch.from_numpy(rgb)/255.0

NODE_CLASS_MAPPINGS = {
    "LHM": LHMReconstructionNode,
    "LHM_Motion_Extract": LHMMotionNode,
    "LHMOffline": LHMOfflineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LHM": "Large Animatable Human Model",
    "LHM_Motion_Extract": "Motion Extraction(LHM)",
    "LHMOffline": "Large Animatable Human Model(Offline)",
}