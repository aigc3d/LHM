from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict
import time
from mmpose.apis.inference import batch_inference_pose_model

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal

from ultralytics import YOLO
from mmpose.apis import (
    init_pose_model,
    get_track_id,
    vis_pose_result,
)

ROOT_DIR = osp.abspath(f"{__file__}/../../")
VIT_DIR = osp.join(ROOT_DIR, "third-party/ViTPose")

VIS_THRESH = 0.5
BBOX_CONF = 0.5
TRACKING_THR = 0.1
MINIMUM_FRMAES = 15
MINIMUM_JOINTS = 6

class DetectionModel(object):
    def __init__(self, pose_model_ckpt, device, with_tracker=True):
        
        # ViTPose
        pose_model_cfg = osp.join(VIT_DIR, 'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py')
        #'vitpose-h-multi-coco.pth')
        self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=device)
        
        # YOLO
        bbox_model_ckpt = os.path.join(os.path.dirname(pose_model_ckpt), 'yolov8x.pt')
        if with_tracker:
            self.bbox_model = YOLO(bbox_model_ckpt)
        else:
            self.bbox_model = None
    
        self.device = device
        self.initialize_tracking()
        
    def initialize_tracking(self, ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
        }
        
    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])
        
    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results['keypoints'].copy()
        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(),
                  kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2]+bb[0])/2, (bb[3]+bb[1])/2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb
        
        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox
    
    def compute_bbox(self, img):
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
        )[0].boxes.xyxy.detach().cpu().numpy()

        bboxes = [{'bbox': bbox} for bbox in bboxes]
        imgs = [img for _ in range(len(bboxes))]
        return bboxes, imgs
    
    def batch_detection(self, bboxes, imgs, batch_size=32):
        all_poses = []
        all_bboxes = []
        for i in range(0, len(bboxes), batch_size):
            poses, bbox_xyxy = batch_inference_pose_model(
                self.pose_model,
                imgs[i:i+batch_size],
                bboxes[i:i+batch_size],
                return_heatmap=False)
            all_poses.append(poses)
            all_bboxes.append(bbox_xyxy)
        all_poses = np.concatenate(all_poses)
        all_bboxes = np.concatenate(all_bboxes)
        return all_poses, all_bboxes
        
    def track(self, img, fps, length):
        # bbox detection
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
        )[0].boxes.xyxy.detach().cpu().numpy()

        pose_results = [{'bbox': bbox} for bbox in bboxes]
        
       
        pose_results, self.next_id = get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=TRACKING_THR,
            use_one_euro=True,
            fps=fps)
        
        for pose_result in pose_results:
            
            _id = pose_result['track_id']
            xyxy = pose_result['bbox']
            bbox = xyxy# self.xyxy_to_cxcys(xyxy)
            
            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
        
        self.frame_id += 1
        self.pose_results_last = pose_results
    
    def process(self, fps):

        for key in ['id', 'frame_id', 'bbox']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        #self.compute_bboxes_from_keypoints()
            
        output = defaultdict(lambda: defaultdict(list))
        ids = np.unique(self.tracking_results['id'])

        for _id in ids:
            idxs = np.where(self.tracking_results['id'] == _id)[0]

            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]

        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox
        
        return output
    
    def visualize(self, img, pose_results):
        vis_img = vis_pose_result(
            self.pose_model,
            img,
            pose_results,
            dataset=self.pose_model.cfg.data['test']['type'],
            dataset_info = None, #self.pose_model.cfg.data['test'].get('dataset_info', None),
            kpt_score_thr=0.3,
            radius=4,
            thickness=1,
            show=False
        )
        return vis_img
