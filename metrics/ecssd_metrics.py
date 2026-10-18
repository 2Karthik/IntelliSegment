import os
import glob
import numpy as np
from PIL import Image

def compute_mae(pred, gt):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - gt))

def compute_s_object(pred, gt):
    """Object-aware structural similarity component of S-measure."""
    x = np.mean(pred[gt == 1]) if np.sum(gt == 1) > 0 else 0
    sigma_x = np.std(pred[gt == 1]) if np.sum(gt == 1) > 0 else 0
    score_fg = (2 * x) / (x**2 + 1 + sigma_x + 1e-8)

    y = np.mean(1 - pred[gt == 0]) if np.sum(gt == 0) > 0 else 0
    sigma_y = np.std(1 - pred[gt == 0]) if np.sum(gt == 0) > 0 else 0
    score_bg = (2 * y) / (y**2 + 1 + sigma_y + 1e-8)

    w = np.mean(gt)
    return w * score_fg + (1 - w) * score_bg

def compute_s_region(pred, gt):
    """Region-aware structural similarity component of S-measure."""
    h, w = pred.shape
    x_c, y_c = int(np.round(np.mean(np.where(gt)[0]))) if np.sum(gt) > 0 else h // 2, \
               int(np.round(np.mean(np.where(gt)[1]))) if np.sum(gt) > 0 else w // 2

    gt_1, pred_1 = gt[:x_c, :y_c], pred[:x_c, :y_c]
    gt_2, pred_2 = gt[:x_c, y_c:], pred[:x_c, y_c:]
    gt_3, pred_3 = gt[x_c:, :y_c], pred[x_c:, :y_c]
    gt_4, pred_4 = gt[x_c:, y_c:], pred[x_c:, y_c:]

    w1 = (gt_1.size) / (gt.size + 1e-8)
    w2 = (gt_2.size) / (gt.size + 1e-8)
    w3 = (gt_3.size) / (gt.size + 1e-8)
    w4 = (gt_4.size) / (gt.size + 1e-8)

    def ssim(p, g):
        if g.size == 0:
            return 0.0
        p_m, g_m = np.mean(p), np.mean(g)
        p_s, g_s = np.var(p), np.var(g)
        cov = np.mean((p - p_m) * (g - g_m))
        return (2 * p_m * g_m + 1e-8) * (2 * cov + 1e-8) / ((p_m**2 + g_m**2 + 1e-8) * (p_s + g_s + 1e-8))

    return w1 * ssim(pred_1, gt_1) + w2 * ssim(pred_2, gt_2) + w3 * ssim(pred_3, gt_3) + w4 * ssim(pred_4, gt_4)

def compute_s_alpha(pred, gt, alpha=0.5):
    """Structure Measure S_alpha combining object and region metrics."""
    s_o = compute_s_object(pred, gt)
    s_r = compute_s_region(pred, gt)
    return alpha * s_o + (1 - alpha) * s_r

def compute_e_phi(pred, gt):
    """Enhanced-Alignment Measure E_phi."""
    align_matrix = 2 * (pred * gt) / (pred**2 + gt**2 + 1e-8)
    enhanced_matrix = ((align_matrix + 1)**2) / 4.0
    return np.mean(enhanced_matrix)

def evaluate_ecssd(pred_dir, gt_dir):
    """
    Evaluates predictions against ground truth for ECSSD dataset.
    Returns: S_alpha, E_phi, MAE
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")) + glob.glob(os.path.join(pred_dir, "*.jpg")))
    
    s_alpha_list, e_phi_list, mae_list = [], [], []

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, filename)
        
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, os.path.splitext(filename)[0] + ".png")
            if not os.path.exists(gt_path):
                continue

        pred = np.array(Image.open(pred_path).convert('L')).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32) / 255.0
        gt = (gt > 0.5).astype(np.float32)

        mae_list.append(compute_mae(pred, gt))
        s_alpha_list.append(compute_s_alpha(pred, gt))
        e_phi_list.append(compute_e_phi(pred, gt))

    s_alpha = np.mean(s_alpha_list) if s_alpha_list else 0.0
    e_phi = np.mean(e_phi_list) if e_phi_list else 0.0
    mae = np.mean(mae_list) if mae_list else 0.0

    return s_alpha, e_phi, mae