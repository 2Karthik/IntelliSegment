import os
import glob
import numpy as np
from PIL import Image

def calculate_mask_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def calculate_mask_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / total

def calculate_f_score(pred, gt, beta2=0.3):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    if precision + recall == 0:
        return 0.0
    
    f_beta = ((1 + beta2) * precision * recall) / (beta2 * precision + recall + 1e-8)
    return f_beta

def evaluate_pascal(pred_dir, gt_dir):
    """
    Evaluates predictions against ground truth for PASCAL dataset.
    Returns: mIoU, mDice, F-Score
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")) + glob.glob(os.path.join(pred_dir, "*.jpg")))
    
    iou_list, dice_list, f_list = [], [], []

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, filename)
        
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, os.path.splitext(filename)[0] + ".png")
            if not os.path.exists(gt_path):
                continue

        pred = np.array(Image.open(pred_path).convert('L')) > 128
        gt = np.array(Image.open(gt_path).convert('L')) > 128

        iou_list.append(calculate_mask_iou(pred, gt))
        dice_list.append(calculate_mask_dice(pred, gt))
        f_list.append(calculate_f_score(pred, gt))

    m_iou = np.mean(iou_list) if iou_list else 0.0
    m_dice = np.mean(dice_list) if dice_list else 0.0
    f_score = np.mean(f_list) if f_list else 0.0

    return m_iou, m_dice, f_score