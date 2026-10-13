import numpy as np

class SAM2BoxSegmenter:
    \"\"\"SAM2 image predictor adapter using verified XYXY bounding boxes.\"\"\"
    def __init__(self, config_path, checkpoint_path, device="cuda", multimask_output=False):
        if not config_path or not checkpoint_path:
            raise ValueError("Set sam2_config and sam2_checkpoint in the YAML config.")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError("Install SAM2 from its official repository and verify the selected release.") from exc
        model = build_sam2(config_path, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(model)
        self.multimask_output = bool(multimask_output)

    def set_image(self, image_rgb: np.ndarray):
        self.predictor.set_image(image_rgb)

    def predict(self, box_xyxy):
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(box_xyxy, dtype=np.float32),
            multimask_output=self.multimask_output
        )
        masks = np.asarray(masks)
        scores = np.asarray(scores)
        if masks.ndim == 2:
            return masks.astype(bool), float(scores.reshape(-1)[0]) if scores.size else 0.0
        idx = int(np.argmax(scores)) if scores.size else 0
        return masks[idx].astype(bool), float(scores[idx]) if scores.size else 0.0
