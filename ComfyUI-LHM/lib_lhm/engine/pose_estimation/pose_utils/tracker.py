import numpy as np
import warnings
import torch


def bbox_xyxy_to_cxcywh(bboxes: np.ndarray, scale=1.0, device=None):
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    cx = (bboxes[..., 0] + bboxes[..., 2]) / 2.0
    cy = (bboxes[..., 1] + bboxes[..., 3]) / 2.0
    new_bboxes = torch.stack([cx, cy, w * scale, h * scale], dim=-1)
    if device is not None:
        new_bboxes = torch.tensor(new_bboxes, device=device)
    return new_bboxes


def compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-8
        warnings.warn("union_area=0 is unexpected")

    iou = inter_area / union_area

    return iou


def track_by_iou(res, results_last, thr):
    """Get track id using IoU tracking greedily.

    Args:
        res (dict): The bbox & pose results of the person instance.
        results_last (list[dict]): The bbox & pose & track_id info of the
            last frame (bbox_result, pose_result, track_id).
        thr (float): The threshold for iou tracking.

    Returns:
        int: The track id for the new person instance.
        list[dict]: The bbox & pose & track_id info of the persons
            that have not been matched on the last frame.
        dict: The matched person instance on the last frame.
    """

    bbox = list(res["bbox"])

    max_iou_score = -1
    max_index = -1
    match_result = {}
    for index, res_last in enumerate(results_last):
        bbox_last = list(res_last["bbox"])

        iou_score = _compute_iou(bbox, bbox_last)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = index

    if max_iou_score > thr:
        track_id = results_last[max_index]["track_id"]
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last, match_result


def track_by_area(humans, target_img_size, threshold=0.3):
    if len(humans) < 1:
        return None

    IMAGE_AREA = target_img_size**2
    target_human = None
    max_area = -1
    for human in humans:
        j2d_coco = human["j2d"].to(torch.float)  # [joints_smplx_to_coco()].to(torch.float)

        # compute bbox
        j2d_area = (j2d_coco[..., 0].max() - j2d_coco[..., 0].min()) * (
            j2d_coco[..., 1].max() - j2d_coco[..., 1].min()
        )
        if max_area < j2d_area:
            max_area = j2d_area
            target_human = human
    # if max_area / IMAGE_AREA < threshold:
    #     return None
    return target_human
