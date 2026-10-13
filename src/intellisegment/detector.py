from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class Detection:
    box: List[float]  # [x1, y1, x2, y2] in original-image pixels
    label: str
    score: float

class YoloNasDetector:
    \"\"\"Adapter around the SuperGradients YOLO-NAS prediction API.

    Validate output parsing against the exact SuperGradients version/checkpoint.
    \"\"\"
    def __init__(self, architecture="yolo_nas_l", checkpoint: Optional[str] = None,
                 confidence_threshold=0.35, class_names=None, device="cuda"):
        try:
            from super_gradients.training import models
        except ImportError as exc:
            raise ImportError("Install a compatible SuperGradients release.") from exc
        self.confidence_threshold = float(confidence_threshold)
        self.class_names = class_names or []
        self.model = models.get(architecture, pretrained_weights="coco" if checkpoint is None else None)
        if checkpoint:
            # Some SuperGradients versions expect checkpoint_path; verify for your release.
            self.model = models.get(architecture, checkpoint_path=checkpoint)
        self.model.to(device)
        self.model.eval()

    def predict(self, image_rgb: np.ndarray) -> List[Detection]:
        \"\"\"Run detection on an RGB uint8 image.

        SuperGradients prediction output varies by release. This adapter handles the
        common ImagePrediction/ prediction.bboxes_xyxy representation and fails loudly
        when it cannot safely interpret the output.
        \"\"\"
        result = self.model.predict(image_rgb, conf=self.confidence_threshold)
        predictions = getattr(result, "prediction", None)
        if predictions is None and isinstance(result, (list, tuple)) and result:
            predictions = getattr(result[0], "prediction", None)
        if predictions is None:
            raise RuntimeError("Unknown YOLO-NAS prediction format; adapt detector.py to your installed version.")
        boxes = getattr(predictions, "bboxes_xyxy", None)
        labels = getattr(predictions, "labels", None)
        confidences = getattr(predictions, "confidence", None)
        if boxes is None or labels is None or confidences is None:
            raise RuntimeError("Prediction lacks bboxes_xyxy, labels, or confidence fields.")
        detections = []
        for box, class_id, score in zip(boxes, labels, confidences):
            score = float(score)
            if score < self.confidence_threshold:
                continue
            idx = int(class_id)
            label = self.class_names[idx] if 0 <= idx < len(self.class_names) else str(idx)
            detections.append(Detection(list(map(float, box)), label, score))
        return detections
