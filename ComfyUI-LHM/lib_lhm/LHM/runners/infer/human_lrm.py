# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu  & Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:30:37
# @Function      : Inference code for human_lrm model

import argparse
import os
import pdb
import time

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox

try:
    from engine.SegmentAPI.SAM import SAM2Seg
except:
    print("\033[31mNo SAM2 found! Try using rembg to remove the background. This may slightly degrade the quality of the results!\033[0m")
    from rembg import remove

from LHM.datasets.cam_utils import (
    build_camera_principle,
    build_camera_standard,
    create_intrinsics,
    surrounding_views_linspace,
)
from LHM.models.modeling_human_lrm import ModelHumanLRM
from LHM.runners import REGISTRY_RUNNERS
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    resize_image_keepaspect_np,
)
from LHM.utils.download_utils import download_extract_tar_from_url
from LHM.utils.face_detector import FaceDetector

# from LHM.utils.video import images_to_video
from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.logging import configure_logger
from LHM.utils.model_card import MODEL_CARD, MODEL_PATH

from .base_inferrer import Inferrer

logger = get_logger(__name__)


def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


def resize_with_padding(img, target_size, padding_color=(255, 255, 255)):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    dw = target_w - new_w
    dh = target_h - new_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    return padded


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

def query_model_name(model_name):
    if model_name in MODEL_PATH:
        model_path = MODEL_PATH[model_name]
        if not os.path.exists(model_path):
            model_url = MODEL_CARD[model_name]
            download_extract_tar_from_url(model_url, './')
    return model_path

def infer_preprocess_image(
    rgb_path,
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

    rgb = np.array(Image.open(rgb_path))
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
        model_name = query_model_name(os.environ.get("APP_MODEL_NAME"))
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")
    else:
        model_name = cli_cfg.model_name
        cli_cfg.model_name = query_model_name(model_name)

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
        cfg.mesh_dump = os.path.join("exps", "meshs", _relative_path)  # output path

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


@REGISTRY_RUNNERS.register("infer.human_lrm")
class HumanLRMInferrer(Inferrer):

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"
    # EXP_TYPE: str = "human_lrm_sd3"

    def __init__(self):
        super().__init__()

        self.cfg, cfg_train = parse_configs()

        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )  # logger function

        self.facedetect = FaceDetector(
            "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
            device=avaliable_device(),
        )
        self.pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device=avaliable_device()
        )
        try:
            self.parsingnet = SAM2Seg()
        except:
            self.parsingnet = None 

        self.model: ModelHumanLRM = self._build_model(self.cfg).to(self.device)

        self.motion_dict = dict()

    def _build_model(self, cfg):
        from LHM.models import model_dict

        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])

        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def _default_source_camera(
        self,
        dist_to_center: float = 2.0,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, -dist_to_center],
                    [0, 1, 0, 0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(
            canonical_camera_extrinsics, canonical_camera_intrinsics
        )
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(
        self,
        n_views: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(
            n_views=n_views, device=device
        )
        render_camera_intrinsics = (
            create_intrinsics(
                f=0.75,
                c=0.5,
                device=device,
            )
            .unsqueeze(0)
            .repeat(render_camera_extrinsics.shape[0], 1, 1)
        )
        render_cameras = build_camera_standard(
            render_camera_extrinsics, render_camera_intrinsics
        )
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_video(
        self,
        planes: torch.Tensor,
        frame_size: int,
        render_size: int,
        render_views: int,
        render_fps: int,
        dump_video_path: str,
    ):
        N = planes.shape[0]
        render_cameras = self._default_render_cameras(
            n_views=render_views, batch_size=N, device=self.device
        )
        render_anchors = torch.zeros(N, render_cameras.shape[1], 2, device=self.device)
        render_resolutions = (
            torch.ones(N, render_cameras.shape[1], 1, device=self.device) * render_size
        )
        render_bg_colors = (
            torch.ones(
                N, render_cameras.shape[1], 1, device=self.device, dtype=torch.float32
            )
            * 1.0
        )

        frames = []
        for i in range(0, render_cameras.shape[1], frame_size):
            frames.append(
                self.model.synthesizer(
                    planes=planes,
                    cameras=render_cameras[:, i : i + frame_size],
                    anchors=render_anchors[:, i : i + frame_size],
                    resolutions=render_resolutions[:, i : i + frame_size],
                    bg_colors=render_bg_colors[:, i : i + frame_size],
                    region_size=render_size,
                )
            )
        # merge frames
        frames = {k: torch.cat([r[k] for r in frames], dim=1) for k in frames[0].keys()}
        # dump
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        for k, v in frames.items():
            if k == "images_rgb":
                images_to_video(
                    images=v[0],
                    output_path=dump_video_path,
                    fps=render_fps,
                    gradio_codec=self.cfg.app_enabled,
                )

    def crop_face_image(self, image_path):
        rgb = np.array(Image.open(image_path))
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        bbox = self.facedetect(rgb)
        head_rgb = rgb[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        head_rgb = head_rgb.permute(1, 2, 0)
        head_rgb = head_rgb.cpu().numpy()
        return head_rgb

    @torch.no_grad()
    def parsing(self, img_path):

        parsing_out = self.parsingnet(img_path=img_path, bbox=None)

        alpha = (parsing_out.masks * 255).astype(np.uint8)

        return alpha

    def infer_mesh(
        self,
        image_path: str,
        dump_tmp_dir: str,  
        dump_mesh_dir: str,
        shape_param=None,
    ):

        source_size = self.cfg.source_size
        aspect_standard = 5.0 / 3

        parsing_mask = self.parsing(image_path)

        # prepare reference image
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )
        try:
            src_head_rgb = self.crop_face_image(image_path)
        except:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)


        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )
        

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # save masked image for vis
        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        smplx_params =  dict()
        # cano pose setting
        smplx_params['betas'] = shape_param.to(device)

        smplx_params['root_pose'] = torch.zeros(1,1,3).to(device)
        smplx_params['body_pose'] = torch.zeros(1,1,21, 3).to(device)
        smplx_params['jaw_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['leye_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['reye_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['lhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
        smplx_params['rhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
        smplx_params['expr'] = torch.zeros(1, 1, 100).to(device)
        smplx_params['trans'] = torch.zeros(1, 1, 3).to(device)

        self.model.to(dtype)

        gs_app_model_list, query_points, transform_mat_neutral_pose = self.model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            None,
            None,
            None,
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )
        smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose

        output_gs = self.model.animation_infer_gs(gs_app_model_list, query_points, smplx_params)

        output_gs_path = '_'.join(os.path.basename(image_path).split('.')[:-1])+'.ply'

        print(f"save mesh to {output_gs_path}")
        output_gs.save_ply(os.path.join(dump_mesh_dir, output_gs_path))


    def infer_single(
        self,
        image_path: str,
        motion_seqs_dir,
        motion_img_dir,
        motion_video_read_fps,
        export_video: bool,
        export_mesh: bool,
        dump_tmp_dir: str,  # require by extracting motion seq from video, to save some results
        dump_image_dir: str,
        dump_video_path: str,
        shape_param=None,
    ):

        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        # render_views = self.cfg.render_views
        render_fps = self.cfg.render_fps
        # mesh_size = self.cfg.mesh_size
        # mesh_thres = self.cfg.mesh_thres
        # frame_size = self.cfg.frame_size
        # source_cam_dist = self.cfg.source_cam_dist if source_cam_dist is None else source_cam_dist
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False


        if self.parsingnet is not None:
            parsing_mask = self.parsing(image_path)
        else:
            img_np = cv2.imread(image_path)
            remove_np = remove(img_np)
            parsing_mask = remove_np[...,3]

        # prepare reference image
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )
        try:
            src_head_rgb = self.crop_face_image(image_path)
        except:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)


        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # save masked image for vis
        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # read motion seq

        motion_name = os.path.dirname(
            motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
        )
        motion_name = os.path.basename(motion_name)

        if motion_name in self.motion_dict:
            motion_seq = self.motion_dict[motion_name]
        else:
            motion_seq = prepare_motion_seqs(
                motion_seqs_dir,
                motion_img_dir,
                save_root=dump_tmp_dir,
                fps=motion_video_read_fps,
                bg_color=1.0,
                aspect_standard=aspect_standard,
                enlarge_ratio=[1.0, 1, 0],
                render_image_res=render_size,
                multiply=16,
                need_mask=motion_img_need_mask,
                vis_motion=vis_motion,
            )
            self.motion_dict[motion_name] = motion_seq

        camera_size = len(motion_seq["motion_seqs"])

        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        self.model.to(dtype)
        smplx_params = motion_seq['smplx_params']
        smplx_params['betas'] = shape_param.to(device)
        gs_model_list, query_points, transform_mat_neutral_pose = self.model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
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

        batch_list = [] 
        batch_size = 40  # avoid memeory out!

        for batch_i in range(0, camera_size, batch_size):
            with torch.no_grad():
                # TODO check device and dtype
                # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])

                print(f"batch: {batch_i}, total: {camera_size //batch_size +1} ")

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

                # def animation_infer(self, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors, render_h, render_w):
                res = self.model.animation_infer(gs_model_list, query_points, batch_smplx_params,
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

            comp_rgb = res["comp_rgb"] # [Nv, H, W, 3], 0-1
            comp_mask = res["comp_mask"] # [Nv, H, W, 3], 0-1
            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
            batch_list.append(batch_rgb)

            del res
            torch.cuda.empty_cache()
        
        rgb = np.concatenate(batch_list, axis=0)

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        print(f"save video to {dump_video_path}")


        images_to_video(
            rgb,
            output_path=dump_video_path,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )

    def infer(self):

        image_paths = []
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths.append(self.cfg.image_input)
        else:
            omit_prefix = self.cfg.image_input
            suffixes = (".jpg", ".jpeg", ".png", ".webp", ".JPG")
            for root, dirs, files in os.walk(self.cfg.image_input):
                for file in files:
                    if file.endswith(suffixes):
                        image_paths.append(os.path.join(root, file))
            image_paths.sort()

        # alloc to each DDP worker
        image_paths = image_paths[
            self.accelerator.process_index :: self.accelerator.num_processes
        ]


        for image_path in tqdm(image_paths,
            disable=not self.accelerator.is_local_main_process,
        ):

            # prepare dump paths
            image_name = os.path.basename(image_path)
            uid = image_name.split(".")[0]
            subdir_path = os.path.dirname(image_path).replace(omit_prefix, "")
            subdir_path = (
                subdir_path[1:] if subdir_path.startswith("/") else subdir_path
            )
            print("subdir_path and uid:", subdir_path, uid)

            # setting config
            motion_seqs_dir = self.cfg.motion_seqs_dir
            motion_name = os.path.dirname(
                motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
            )
            motion_name = os.path.basename(motion_name)
            dump_video_path = os.path.join(
                self.cfg.video_dump,
                subdir_path,
                motion_name,
                f"{uid}.mp4",
            )
            dump_image_dir = os.path.join(
                self.cfg.image_dump,
                subdir_path,
            )
            dump_mesh_dir = os.path.join(
                self.cfg.mesh_dump,
                subdir_path,
            )
            dump_tmp_dir = os.path.join(self.cfg.image_dump, subdir_path, "tmp_res")
            os.makedirs(dump_image_dir, exist_ok=True)
            os.makedirs(dump_tmp_dir, exist_ok=True)
            os.makedirs(dump_mesh_dir, exist_ok=True)

            shape_pose = self.pose_estimator(image_path)
            assert shape_pose.is_full_body, f"The input image is illegal, {shape_pose.msg}"

            if self.cfg.export_mesh is not None:
                self.infer_mesh(
                    image_path,
                    dump_tmp_dir=dump_tmp_dir,
                    dump_mesh_dir=dump_mesh_dir,
                    shape_param=shape_pose.beta,
                )
            else:
                self.infer_single(
                    image_path,
                    motion_seqs_dir=self.cfg.motion_seqs_dir,
                    motion_img_dir=self.cfg.motion_img_dir,
                    motion_video_read_fps=self.cfg.motion_video_read_fps,
                    export_video=self.cfg.export_video,
                    export_mesh=self.cfg.export_mesh,
                    dump_tmp_dir=dump_tmp_dir,
                    dump_image_dir=dump_image_dir,
                    dump_video_path=dump_video_path,
                    shape_param=shape_pose.beta,
                )


@REGISTRY_RUNNERS.register("infer.human_lrm_video")
class HumanLRMVideoInferrer(HumanLRMInferrer):
    """video reconstruction for in the wild data"""

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"

    def infer_single(
        self,
        image_path: str,
        motion_seqs_dir,
        motion_img_dir,
        motion_video_read_fps,
        export_video: bool,
        export_mesh: bool,
        dump_tmp_dir: str,  # require by extracting motion seq from video, to save some results
        dump_image_dir: str,
        dump_video_path: str,
    ):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        # render_views = self.cfg.render_views
        render_fps = self.cfg.render_fps
        # mesh_size = self.cfg.mesh_size
        # mesh_thres = self.cfg.mesh_thres
        # frame_size = self.cfg.frame_size
        # source_cam_dist = self.cfg.source_cam_dist if source_cam_dist is None else source_cam_dist
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False

        parsing_mask = self.parsing(image_path)

        save_dir = os.path.join(dump_image_dir, "rgb")
        if os.path.exists(save_dir):
            return

        # prepare reference image
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )
        src_head_rgb = self.crop_face_image(image_path)


        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # save masked image for vis
        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )

        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # read motion seq

        if not os.path.exists(motion_seqs_dir):
            return

        motion_seq = prepare_motion_seqs(
            motion_seqs_dir,
            os.path.basename(image_path),
            save_root=dump_tmp_dir,
            fps=motion_video_read_fps,
            bg_color=1.0,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
            vis_motion=vis_motion,
        )
        motion_seqs = motion_seq["motion_seqs"]

        device = "cuda"
        dtype = torch.float32
        self.model.to(dtype)


        with torch.no_grad():
            # TODO check device and dtype
            # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])
            render_intrs = motion_seq["render_intrs"].to(device)
            render_intrs[..., 0, 0] *= 2
            render_intrs[..., 1, 1] *= 2
            render_intrs[..., 0, 2] *= 2
            render_intrs[..., 1, 2] *= 2
            # smplx_params["focal"] *= 2
            # smplx_params["princpt"] *= 2
            # smplx_params["img_size_wh"] *= 2

            res = self.model.infer_single_view(
                image.unsqueeze(0).to(device, dtype),
                src_head_rgb.unsqueeze(0).to(device, dtype),
                None,
                None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=render_intrs,
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                smplx_params={
                    k: v.to(device) for k, v in motion_seq["smplx_params"].items()
                },
            )

        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        mask = res["comp_mask"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        # mask[mask > 0.5] = 1.0
        # mask[mask < 0.4] = 0.0
        rgb = rgb * mask + (1 - mask) * 1

        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        rgba_numpy = np.concatenate([rgb, mask], axis=-1)

        for rgb_i, (rgba, motion_seq) in enumerate(zip(rgba_numpy, motion_seqs)):

            rgb_i = int(os.path.basename(motion_seq).replace(".json", ""))
            save_file = os.path.join(dump_image_dir, "rgb", f"{rgb_i:05d}.png")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            Image.fromarray(rgba).save(save_file)

    def infer(self):

        image_paths = []

        omit_prefix = self.cfg.image_input
        suffixes = (".jpg", ".jpeg", ".png", ".webp")

        front_view_dict = dict()
        with open(os.path.join(self.cfg.image_input, "front_view.txt"), "r") as f:
            for line in f.readlines():
                name, idx = line.strip().split(" ")
                idx = int(idx)
                front_view_dict[name] = idx

        for root, dirs, files in os.walk(self.cfg.image_input):
            for dir in dirs:
                if dir in front_view_dict:
                    idx = front_view_dict[dir]
                else:
                    raise ValueError("no front view")
                img_path = os.path.join(root, dir, f"{idx:06d}.png")
                if dir in front_view_dict:
                    print(img_path)
                image_paths.append(img_path)

        image_paths.sort()

        # alloc to each DDP worke
        image_paths = image_paths[
            self.accelerator.process_index :: self.accelerator.num_processes
        ]

        for image_path in tqdm(
            image_paths, disable=not self.accelerator.is_local_main_process
        ):

            # prepare dump paths
            image_name = os.path.basename(image_path)
            uid = image_name.split(".")[0]
            subdir_path = os.path.dirname(image_path).replace(omit_prefix, "")
            subdir_path = (
                subdir_path[1:] if subdir_path.startswith("/") else subdir_path
            )
            print("subdir_path and uid:", subdir_path, uid)

            # setting config
            motion_seqs_dir = self.cfg.motion_seqs_dir
            motion_name = os.path.dirname(
                motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
            )
            motion_name = os.path.basename(motion_name)
            dump_video_path = os.path.join(
                self.cfg.video_dump.replace("videos", "videos_benchmark"),
                subdir_path,
                motion_name,
                f"{uid}.mp4",
            )
            dump_image_dir = os.path.join(
                self.cfg.image_dump.replace("images", "images_benchmark"),
                subdir_path,
            )

            dump_tmp_dir = os.path.join(self.cfg.image_dump, subdir_path, "tmp_res")
            os.makedirs(dump_image_dir, exist_ok=True)
            os.makedirs(dump_tmp_dir, exist_ok=True)

            item_name = os.path.basename(os.path.dirname(image_path))

            self.infer_single(
                image_path,
                motion_seqs_dir=os.path.join(self.cfg.motion_seqs_dir, item_name),
                motion_img_dir=self.cfg.motion_img_dir,
                motion_video_read_fps=self.cfg.motion_video_read_fps,
                export_video=self.cfg.export_video,
                export_mesh=self.cfg.export_mesh,
                dump_tmp_dir=dump_tmp_dir,
                dump_image_dir=dump_image_dir,
                dump_video_path=dump_video_path,
            )
