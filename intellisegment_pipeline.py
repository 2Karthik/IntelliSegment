import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import clip
from super_gradients.training import models
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class IntelliSegment(nn.Module):
    """
    IntelliSegment Framework:
    YOLO-NAS (Object Detection) -> CLIP (Semantic Verification) -> SAM2 (Instance Mask Generation)
    """
    def __init__(
        self,
        yolo_model: str = "yolo_nas_m",
        sam2_checkpoint: str = "sam2_hiera_large.pt",
        sam2_config: str = "sam2_hiera_l.yaml",
        use_clip: bool = True,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.use_clip = use_clip

        # 1. YOLO-NAS Detector (Backbone + FPN Neck)
        self.detector = models.get(yolo_model, pretrained_weights="coco").to(self.device)
        self.detector.eval()

        # 2. SAM2 Segmentor
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam_predictor = SAM2ImagePredictor(sam2_model)

        # 3. CLIP Verifier (Optional for Ablation Studies)
        if self.use_clip:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            self.clip_model.eval()

    def _verify_with_clip(self, image_rgb: np.ndarray, boxes: np.ndarray, classes: np.ndarray, class_names: list, conf_thresh: float = 0.20):
        """Filters false positive box predictions using CLIP visual-text cosine similarity."""
        text_tokens = clip.tokenize([f"a photo of a {c}" for c in class_names]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        verified_boxes, verified_scores, verified_classes = [], [], []
        img_h, img_w, _ = image_rgb.shape

        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            crop = image_rgb[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop)
            crop_tensor = self.clip_preprocess(crop_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                img_feat = self.clip_model.encode_image(crop_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                similarity = (100.0 * img_feat @ text_features.T).softmax(dim=-1)
                score, pred_cls = similarity.max(dim=-1)

            if score.item() >= conf_thresh:
                verified_boxes.append(box)
                verified_scores.append(score.item())
                verified_classes.append(pred_cls.item())

        return np.array(verified_boxes), np.array(verified_scores), np.array(verified_classes)

    def forward(self, image_rgb: np.ndarray, class_names: list, conf_threshold: float = 0.25):
        # Step 1: Detect bounding boxes with YOLO-NAS
        det_res = self.detector.predict(image_rgb, conf=conf_threshold)
        preds = det_res[0].prediction
        boxes, scores, classes = preds.bboxes_xyxy, preds.confidence, preds.labels.astype(int)

        if len(boxes) == 0:
            return [], [], [], []

        # Step 2: Semantic verification with CLIP (if enabled)
        if self.use_clip:
            boxes, scores, classes = self._verify_with_clip(image_rgb, boxes, classes, class_names)
            if len(boxes) == 0:
                return [], [], [], []

        # Step 3: Fine-grained mask generation with SAM2 using box prompts
        self.sam_predictor.set_image(image_rgb)
        masks = []
        for box in boxes:
            mask, _, _ = self.sam_predictor.predict(
                point_coords=None, point_labels=None, box=box[None, :], multimask_output=False
            )
            masks.append(mask[0])

        return boxes, np.array(masks), scores, classes
