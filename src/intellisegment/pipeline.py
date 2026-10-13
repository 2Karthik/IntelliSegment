import os
import cv2
import numpy as np
from .detector import YoloNasDetector
from .clip_verifier import CLIPVerifier
from .sam2_segmenter import SAM2BoxSegmenter

class IntelliSegment:
    def __init__(self, config):
        det = config["detector"]
        clip = config["clip"]
        sam = config["sam2"]
        device = config.get("device", "cuda")
        self.detector = YoloNasDetector(
            architecture=det.get("architecture", "yolo_nas_l"),
            checkpoint=det.get("checkpoint"),
            confidence_threshold=det.get("confidence_threshold", 0.35),
            class_names=det.get("class_names", []),
            device=device,
        )
        self.clip = CLIPVerifier(
            model_name=clip.get("model_name", "openai/clip-vit-base-patch32"),
            device=device,
            prompt_template=clip.get("prompt_template", "a photo of a {label}"),
        )
        self.clip_threshold = float(clip.get("similarity_threshold", 0.20))
        self.sam = SAM2BoxSegmenter(
            sam.get("sam2_config"), sam.get("sam2_checkpoint"), device,
            sam.get("multimask_output", False)
        )

    def run(self, image_rgb):
        detections = self.detector.predict(image_rgb)
        self.sam.set_image(image_rgb)
        outputs = []
        candidate_labels = self.detector.class_names
        for det in detections:
            labels = candidate_labels or [det.label]
            best_label, sim, scores = self.clip.verify(image_rgb, det.box, labels)
            if best_label is None or sim < self.clip_threshold:
                continue
            # Keep the detector box only if CLIP agrees with its class. If the
            # best candidate differs, record the verified semantic label explicitly.
            if best_label != det.label and candidate_labels:
                # Conservative policy: reject class disagreement. Change only if
                # your method explicitly allows CLIP to relabel detections.
                continue
            mask, sam_score = self.sam.predict(det.box)
            outputs.append({
                "box": det.box, "label": best_label, "detector_score": det.score,
                "clip_similarity": sim, "clip_scores": scores,
                "mask": mask, "sam_score": sam_score
            })
        return outputs

def save_outputs(image_rgb, outputs, output_dir, stem="result"):
    os.makedirs(output_dir, exist_ok=True)
    overlay = image_rgb.copy()
    for i, item in enumerate(outputs):
        mask = item["mask"]
        # Overlay with a simple fixed color for readability.
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0, 255, 80])).astype(np.uint8)
        x1, y1, x2, y2 = map(int, item["box"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 80, 0), 2)
        cv2.putText(overlay, f'{item["label"]} {item["clip_similarity"]:.2f}',
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 1)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_mask_{i:03d}.png"),
                    (mask.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(output_dir, f"{stem}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
