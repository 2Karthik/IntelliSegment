import numpy as np

def _binary_pair(pred, gt, threshold=0.5):
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: prediction {p.shape} vs ground truth {g.shape}")
    if p.max(initial=0) > 1:
        p = p / 255.0
    if g.max(initial=0) > 1:
        g = g / 255.0
    return (p >= threshold), (g >= threshold), p, g

def iou_dice_fscore(pred, gt, threshold=0.5, eps=1e-8):
    p, g, _, _ = _binary_pair(pred, gt, threshold)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fscore = 2 * precision * recall / (precision + recall + eps)
    return {"IoU": float(iou), "Dice": float(dice),
            "Precision": float(precision), "Recall": float(recall),
            "F1": float(fscore)}

def mae(pred, gt):
    _, _, p, g = _binary_pair(pred, gt, threshold=0.5)
    return float(np.mean(np.abs(p - g)))

def s_alpha_approx(pred, gt, threshold=0.5):
    \"\"\"Practical approximate S-measure; validate against benchmark's official code.

    Combines object-aware and region-aware terms. This is not guaranteed bit-for-bit
    equivalent to the canonical S-measure implementation.
    \"\"\"
    p, g, _, _ = _binary_pair(pred, gt, threshold)
    if g.sum() == 0 or g.sum() == g.size:
        return float(1.0 - np.mean(np.abs(p.astype(float) - g.astype(float))))
    fg = p[g].mean() if g.any() else 0.0
    bg = 1.0 - p[~g].mean() if (~g).any() else 0.0
    object_score = 0.5 * (fg + bg)
    # Region term based on global binary agreement as a conservative approximation.
    region_score = float(np.mean(p == g))
    return float(0.5 * object_score + 0.5 * region_score)

def e_phi_approx(pred, gt):
    \"\"\"Enhanced-alignment E-measure approximation; use official benchmark code for papers.\"\"\"
    _, _, p, g = _binary_pair(pred, gt)
    if g.max() == g.min():
        return float(1.0 - np.mean(np.abs(p - g)))
    p = p - p.mean()
    g = g - g.mean()
    alignment = 2 * p * g / (p * p + g * g + 1e-8)
    enhanced = ((alignment + 1) ** 2) / 4
    return float(np.mean(enhanced))

def evaluate_pair(pred, gt, dataset="ecssd"):
    if dataset.lower() in ("pascal", "pascal_voc", "voc"):
        return iou_dice_fscore(pred, gt)
    if dataset.lower() == "ecssd":
        return {"S_alpha_approx": s_alpha_approx(pred, gt),
                "E_phi_approx": e_phi_approx(pred, gt),
                "MAE": mae(pred, gt)}
    raise ValueError("dataset must be 'pascal_voc' or 'ecssd'")
