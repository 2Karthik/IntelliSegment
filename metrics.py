import numpy as np

class EvaluationMetrics:
    """Utility class for computing segmentation and perceptual metrics."""

    @staticmethod
    def compute_mae(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        p = pred_mask.astype(np.float32) / 255.0 if pred_mask.max() > 1.0 else pred_mask.astype(np.float32)
        g = gt_mask.astype(np.float32) / 255.0 if gt_mask.max() > 1.0 else gt_mask.astype(np.float32)
        return float(np.mean(np.abs(p - g)))

    @staticmethod
    def compute_iou_dice_fscore(pred_mask: np.ndarray, gt_mask: np.ndarray, beta2: float = 0.3) -> dict:
        p = (pred_mask > 0.5).astype(np.uint8)
        g = (gt_mask > 0.5).astype(np.uint8)
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        p_sum, g_sum = p.sum(), g.sum()

        iou = intersection / (union + 1e-8)
        dice = (2.0 * intersection) / (p_sum + g_sum + 1e-8)
        precision = intersection / (p_sum + 1e-8)
        recall = intersection / (g_sum + 1e-8)
        f_score = ((1 + beta2) * precision * recall) / (beta2 * precision + recall + 1e-8)
        return {"iou": float(iou), "dice": float(dice), "f_score": float(f_score)}

    @staticmethod
    def compute_e_phi(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        p = (pred_mask > 0.5).astype(np.float32)
        g = (gt_mask > 0.5).astype(np.float32)
        p_dev, g_dev = p - np.mean(p), g - np.mean(g)
        align_matrix = 2 * (g_dev * p_dev) / (g_dev**2 + p_dev**2 + 1e-8)
        return float(np.mean(((align_matrix + 1)**2) / 4.0))

    @staticmethod
    def compute_s_alpha(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        p = (pred_mask > 0.5).astype(np.float32)
        g = (gt_mask > 0.5).astype(np.uint8)
        x_m, y_m = np.mean(p), np.mean(g)
        x_v, y_v = np.var(p), np.var(g)
        cov = np.mean((p - x_m) * (g - y_m))
        ssim = ((2 * x_m * y_m + 0.0001) * (2 * cov + 0.0009)) / ((x_m**2 + y_m**2 + 0.0001) * (x_v + y_v + 0.0009) + 1e-8)
        return float(ssim)
