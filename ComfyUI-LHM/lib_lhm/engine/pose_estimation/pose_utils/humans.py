# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import numpy as np
import torch.nn.functional as F
import torch
import roma
from smplx.joint_names import JOINT_NAMES

def rot6d_to_rotmat(x):
    """
    6D rotation representation to 3x3 rotation matrix.
    Args:
        x: (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    y = roma.special_gramschmidt(x)
    return y

def get_smplx_joint_names(*args, **kwargs):
    return JOINT_NAMES[:127]

COCO17_JOINTS_NAME = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 
    3: 'left_ear', 4: 'right_ear', 5:'left_shoulder', 
    6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 
    12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 
    15: 'left_ankle', 16: 'right_ankle'
}

OPENPOSE25_JOINTS_NAME = {
    0: 'nose', 1: 'neck', 2: 'right_shoulder', 3: 'right_elbow', 4: 'right_wrist',
    5: 'left_shoulder', 6: 'left_elbow', 7: 'left_wrist', 8: 'MidHip', 9: 'right_hip', 10: 'right_knee', 11: 'right_ankle', 12: 'left_hip',
    13: 'left_knee', 14: 'left_ankle', 15: 'right_eye', 16: 'left_eye', 17: 'right_ear', 18: 'left_ear', 19: 'LBigToe',
    20: 'LSmallToe', 21: 'left_heel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'right_heel',
}


def joints_smplx_to_coco():
    smplx_joints_name = get_smplx_joint_names()
    joints_idx = []
    for k, v in COCO17_JOINTS_NAME.items():
        joints_idx.append(smplx_joints_name.index(v))

    return joints_idx

def joints_openpose25_to_coco17():
    idx_list = [0] * 17
    is_found = False
    for coco_key, coco_value in COCO17_JOINTS_NAME.items():
        is_found = False
        for openpose_key, openpose_value in OPENPOSE25_JOINTS_NAME.items():
            if coco_value == openpose_value:
                idx_list[coco_key] = openpose_key
                is_found = True
                break
        assert is_found, f'{coco_key} is not found in openpose keypoints'
    return idx_list



COCO_WHOLEBODY_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_bigtoe",
    "left_smalltoe",
    "left_heel",
    "right_bigtoe",
    "right_smalltoe",
    "right_heel",
    "right_contour_1",  # original name: face_contour_1
    "right_contour_2",  # original name: face_contour_2
    "right_contour_3",  # original name: face_contour_3
    "right_contour_4",  # original name: face_contour_4
    "right_contour_5",  # original name: face_contour_5
    "right_contour_6",  # original name: face_contour_6
    "right_contour_7",  # original name: face_contour_7
    "right_contour_8",  # original name: face_contour_8
    "contour_middle",  # original name: face_contour_9
    "left_contour_8",  # original name: face_contour_10
    "left_contour_7",  # original name: face_contour_11
    "left_contour_6",  # original name: face_contour_12
    "left_contour_5",  # original name: face_contour_13
    "left_contour_4",  # original name: face_contour_14
    "left_contour_3",  # original name: face_contour_15
    "left_contour_2",  # original name: face_contour_16
    "left_contour_1",  # original name: face_contour_17
    "right_eyebrow_1",
    "right_eyebrow_2",
    "right_eyebrow_3",
    "right_eyebrow_4",
    "right_eyebrow_5",
    "left_eyebrow_5",
    "left_eyebrow_4",
    "left_eyebrow_3",
    "left_eyebrow_2",
    "left_eyebrow_1",
    "nosebridge_1",
    "nosebridge_2",
    "nosebridge_3",
    "nosebridge_4",
    "right_nose_2",  # original name: nose_1
    "right_nose_1",  # original name: nose_2
    "nose_middle",  # original name: nose_3
    "left_nose_1",  # original name: nose_4
    "left_nose_2",  # original name: nose_5
    "right_eye_1",
    "right_eye_2",
    "right_eye_3",
    "right_eye_4",
    "right_eye_5",
    "right_eye_6",
    "left_eye_4",
    "left_eye_3",
    "left_eye_2",
    "left_eye_1",
    "left_eye_6",
    "left_eye_5",
    "right_mouth_1",  # original name: mouth_1
    "right_mouth_2",  # original name: mouth_2
    "right_mouth_3",  # original name: mouth_3
    "mouth_top",  # original name: mouth_4
    "left_mouth_3",  # original name: mouth_5
    "left_mouth_2",  # original name: mouth_6
    "left_mouth_1",  # original name: mouth_7
    "left_mouth_5",  # original name: mouth_8
    "left_mouth_4",  # original name: mouth_9
    "mouth_bottom",  # original name: mouth_10
    "right_mouth_4",  # original name: mouth_11
    "right_mouth_5",  # original name: mouth_12
    "right_lip_1",  # original name: lip_1
    "right_lip_2",  # original name: lip_2
    "lip_top",  # original name: lip_3
    "left_lip_2",  # original name: lip_4
    "left_lip_1",  # original name: lip_5
    "left_lip_3",  # original name: lip_6
    "lip_bottom",  # original name: lip_7
    "right_lip_3",  # original name: lip_8
    "left_hand_root",
    "left_thumb_1",
    "left_thumb_2",
    "left_thumb_3",
    "left_thumb",
    "left_index_1",
    "left_index_2",
    "left_index_3",
    "left_index",
    "left_middle_1",
    "left_middle_2",
    "left_middle_3",
    "left_middle",
    "left_ring_1",
    "left_ring_2",
    "left_ring_3",
    "left_ring",
    "left_pinky_1",
    "left_pinky_2",
    "left_pinky_3",
    "left_pinky",
    "right_hand_root",
    "right_thumb_1",
    "right_thumb_2",
    "right_thumb_3",
    "right_thumb",
    "right_index_1",
    "right_index_2",
    "right_index_3",
    "right_index",
    "right_middle_1",
    "right_middle_2",
    "right_middle_3",
    "right_middle",
    "right_ring_1",
    "right_ring_2",
    "right_ring_3",
    "right_ring",
    "right_pinky_1",
    "right_pinky_2",
    "right_pinky_3",
    "right_pinky",
]

SMPLX_KEYPOINTS = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eyeball",
    "right_eyeball",
    "left_index_1",
    "left_index_2",
    "left_index_3",
    "left_middle_1",
    "left_middle_2",
    "left_middle_3",
    "left_pinky_1",
    "left_pinky_2",
    "left_pinky_3",
    "left_ring_1",
    "left_ring_2",
    "left_ring_3",
    "left_thumb_1",
    "left_thumb_2",
    "left_thumb_3",
    "right_index_1",
    "right_index_2",
    "right_index_3",
    "right_middle_1",
    "right_middle_2",
    "right_middle_3",
    "right_pinky_1",
    "right_pinky_2",
    "right_pinky_3",
    "right_ring_1",
    "right_ring_2",
    "right_ring_3",
    "right_thumb_1",
    "right_thumb_2",
    "right_thumb_3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_bigtoe",
    "left_smalltoe",
    "left_heel",
    "right_bigtoe",
    "right_smalltoe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eyebrow_1",
    "right_eyebrow_2",
    "right_eyebrow_3",
    "right_eyebrow_4",
    "right_eyebrow_5",
    "left_eyebrow_5",
    "left_eyebrow_4",
    "left_eyebrow_3",
    "left_eyebrow_2",
    "left_eyebrow_1",
    "nosebridge_1",
    "nosebridge_2",
    "nosebridge_3",
    "nosebridge_4",
    "right_nose_2",  # original name: nose_1
    "right_nose_1",  # original name: nose_2
    "nose_middle",  # original name: nose_3
    "left_nose_1",  # original name: nose_4
    "left_nose_2",  # original name: nose_5
    "right_eye_1",
    "right_eye_2",
    "right_eye_3",
    "right_eye_4",
    "right_eye_5",
    "right_eye_6",
    "left_eye_4",
    "left_eye_3",
    "left_eye_2",
    "left_eye_1",
    "left_eye_6",
    "left_eye_5",
    "right_mouth_1",  # original name: mouth_1
    "right_mouth_2",  # original name: mouth_2
    "right_mouth_3",  # original name: mouth_3
    "mouth_top",  # original name: mouth_4
    "left_mouth_3",  # original name: mouth_5
    "left_mouth_2",  # original name: mouth_6
    "left_mouth_1",  # original name: mouth_7
    "left_mouth_5",  # original name: mouth_8
    "left_mouth_4",  # original name: mouth_9
    "mouth_bottom",  # original name: mouth_10
    "right_mouth_4",  # original name: mouth_11
    "right_mouth_5",  # original name: mouth_12
    "right_lip_1",  # original name: lip_1
    "right_lip_2",  # original name: lip_2
    "lip_top",  # original name: lip_3
    "left_lip_2",  # original name: lip_4
    "left_lip_1",  # original name: lip_5
    "left_lip_3",  # original name: lip_6
    "lip_bottom",  # original name: lip_7
    "right_lip_3",  # original name: lip_8
    "right_contour_1",  # original name: face_contour_1
    "right_contour_2",  # original name: face_contour_2
    "right_contour_3",  # original name: face_contour_3
    "right_contour_4",  # original name: face_contour_4
    "right_contour_5",  # original name: face_contour_5
    "right_contour_6",  # original name: face_contour_6
    "right_contour_7",  # original name: face_contour_7
    "right_contour_8",  # original name: face_contour_8
    "contour_middle",  # original name: face_contour_9
    "left_contour_8",  # original name: face_contour_10
    "left_contour_7",  # original name: face_contour_11
    "left_contour_6",  # original name: face_contour_12
    "left_contour_5",  # original name: face_contour_13
    "left_contour_4",  # original name: face_contour_14
    "left_contour_3",  # original name: face_contour_15
    "left_contour_2",  # original name: face_contour_16
    "left_contour_1",  # original name: face_contour_17
]

LEFT_HAND_KEYPOINTS = [
    "left_wrist",
    "left_index_1",
    "left_index_2",
    "left_index_3",
    "left_middle_1",
    "left_middle_2",
    "left_middle_3",
    "left_pinky_1",
    "left_pinky_2",
    "left_pinky_3",
    "left_ring_1",
    "left_ring_2",
    "left_ring_3",
    "left_thumb_1",
    "left_thumb_2",
    "left_thumb_3",
]

RIGHT_HAND_KEYPOINTS = [
    "right_wrist",
    "right_index_1",
    "right_index_2",
    "right_index_3",
    "right_middle_1",
    "right_middle_2",
    "right_middle_3",
    "right_pinky_1",
    "right_pinky_2",
    "right_pinky_3",
    "right_ring_1",
    "right_ring_2",
    "right_ring_3",
    "right_thumb_1",
    "right_thumb_2",
    "right_thumb_3",
    
]

COCO_PLUS_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    "left_bigtoe",
    "left_smalltoe",
    "left_heel",
    "right_bigtoe",
    "right_smalltoe",
    "right_heel",
]

KEYPOINTS_FACTORY = {
    "smplx": SMPLX_KEYPOINTS,
    "coco_wholebody": COCO_WHOLEBODY_KEYPOINTS,
    "left_hand": LEFT_HAND_KEYPOINTS,
    "right_hand": RIGHT_HAND_KEYPOINTS,
    "coco_plus": COCO_PLUS_KEYPOINTS,
}

MAPPING_CACHE = {}

def get_mapping(
    src: str,
    dst: str,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
):
    """Get mapping list from src to dst.

    Args:
        src (str): source data type from keypoints_factory.
        dst (str): destination data type from keypoints_factory.
        approximate (bool): control whether approximate mapping is allowed.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.

    Returns:
        list:
            [src_to_intersection_idx, dst_to_intersection_index,
             intersection_names]
    """
    if src.lower() in MAPPING_CACHE.keys() and dst.lower() in MAPPING_CACHE[src.lower()].keys():
        return MAPPING_CACHE[src.lower()][dst.lower()]
    
    src_names = keypoints_factory[src.lower()]
    dst_names = keypoints_factory[dst.lower()]

    dst_idxs, src_idxs, intersection = [], [], []
    full_mapping_idx = []
    unmapped_names, approximate_names = [], []
    for dst_idx, dst_name in enumerate(dst_names):
        try:
            src_idx = src_names.index(dst_name)
        except ValueError:
            src_idx = -1
        if src_idx >= 0:
            dst_idxs.append(dst_idx)
            src_idxs.append(src_idx)
            intersection.append(dst_name)
        full_mapping_idx.append(src_idx)
            # approximate mapping

    mapping_list = (dst_idxs, src_idxs, intersection, full_mapping_idx)
    if not src.lower() in MAPPING_CACHE.keys():
        MAPPING_CACHE[src.lower()] = {}
    MAPPING_CACHE[src.lower()][dst.lower()] = mapping_list
    return mapping_list

    
if __name__ == '__main__':
    print(joints_smplx_to_coco())