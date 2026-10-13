from typing import List
import numpy as np
from PIL import Image

class CLIPVerifier:
    \"\"\"Compare detected image crops against candidate category text prompts.\"\"\"
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda",
                 prompt_template="a photo of a {label}"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.prompt_template = prompt_template

    @staticmethod
    def crop(image_rgb: np.ndarray, box):
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return Image.fromarray(image_rgb[y1:y2, x1:x2])

    def verify(self, image_rgb, box, candidate_labels: List[str]):
        \"\"\"Return (best_label, cosine_similarity, all_scores).

        The best label is chosen from candidate_labels. Threshold calibration is
        dataset/model-dependent; do not treat the default threshold as universal.
        \"\"\"
        crop = self.crop(image_rgb, box)
        if crop is None:
            return None, 0.0, {}
        texts = [self.prompt_template.format(label=x.replace("_", " ")) for x in candidate_labels]
        inputs = self.processor(text=texts, images=[crop] * len(texts),
                                return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            image_emb = outputs.image_embeds
            text_emb = outputs.text_embeds
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            # One crop is repeated per prompt, so compare aligned image/text embeddings.
            scores = (image_emb * text_emb).sum(dim=-1).detach().cpu().numpy()
        score_map = {label: float(score) for label, score in zip(candidate_labels, scores)}
        best = max(score_map, key=score_map.get)
        return best, score_map[best], score_map
