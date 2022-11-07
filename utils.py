import numpy as np


def calculate_iou_scores(pred_bb, gt_bb):
    gt_bb = np.expand_dims(gt_bb, 0)
    pred_bb = np.expand_dims(pred_bb, 1)
    int_x1 = np.maximum(pred_bb[..., 0], gt_bb[..., 0]) 
    int_y1 = np.maximum(pred_bb[..., 1], gt_bb[..., 1]) 
    int_x2 = np.minimum(pred_bb[..., 2], gt_bb[..., 2]) 
    int_y2 = np.minimum(pred_bb[..., 3], gt_bb[..., 3]) 
    int_w = np.maximum(0, int_x2 - int_x1)
    int_h = np.maximum(0, int_y2 - int_y1)
    int_area = np.multiply(int_w, int_h)
    gtbb_area = (gt_bb[..., 2] - gt_bb[..., 0]) * (gt_bb[..., 3] - (gt_bb[..., 1]))
    predbb_area = (pred_bb[..., 2] - pred_bb[..., 0]) * (pred_bb[..., 3] - pred_bb[..., 1])
    union_area = gtbb_area + predbb_area - int_area
    iou_matrix = int_area / union_area
    return iou_matrix
