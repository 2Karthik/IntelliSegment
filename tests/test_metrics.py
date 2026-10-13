import numpy as np
from intellisegment.metrics import iou_dice_fscore, mae

def test_perfect_mask():
    gt = np.array([[0, 1], [1, 0]], dtype=float)
    result = iou_dice_fscore(gt, gt)
    assert abs(result["IoU"] - 1.0) < 1e-6
    assert abs(result["Dice"] - 1.0) < 1e-6
    assert abs(mae(gt, gt)) < 1e-6
