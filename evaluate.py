import os
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from intellisegment_pipeline import IntelliSegment

PASCAL_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def run_pascal_evaluation(gt_json_path: str, img_dir: str, use_clip: bool, output_json: str):
    coco_gt = COCO(gt_json_path)
    img_ids = coco_gt.getImgIds()
    pipeline = IntelliSegment(use_clip=use_clip)

    coco_results = []
    print(f"\nRunning Inference (use_clip={use_clip})...")

    for img_id in tqdm(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_info['height'], img_info['width']

        boxes, masks, scores, classes = pipeline(image_rgb, class_names=PASCAL_CLASSES)

        for box, mask, score, cls_id in zip(boxes, masks, scores, classes):
            mask_resized = cv2.resize(mask.astype(np.uint8), (img_w, img_h))
            x1, y1, x2, y2 = box
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            rle_enc = mask_utils.encode(np.asfortranarray(mask_resized))
            rle_enc['counts'] = rle_enc['counts'].decode('utf-8')

            coco_results.append({
                "image_id": int(img_id),
                "category_id": int(cls_id + 1),
                "bbox": bbox,
                "segmentation": rle_enc,
                "score": float(score)
            })

    with open(output_json, 'w') as f:
        json.dump(coco_results, f)

    # Compute COCOeval BBox and Mask Metrics
    coco_dt = coco_gt.loadRes(output_json)

    print("\n--- Detection BBox Metrics ---")
    eval_bbox = COCOeval(coco_gt, coco_dt, iouType='bbox')
    eval_bbox.evaluate(); eval_bbox.accumulate(); eval_bbox.summarize()

    print("\n--- Instance Mask Metrics ---")
    eval_mask = COCOeval(coco_gt, coco_dt, iouType='segm')
    eval_mask.evaluate(); eval_mask.accumulate(); eval_mask.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json", type=str, default="data/pascal_voc2012_val.json")
    parser.add_argument("--img_dir", type=str, default="data/VOCdevkit/VOC2012/JPEGImages")
    args = parser.parse_args()

    # 1. Baseline Evaluation
    run_pascal_evaluation(args.gt_json, args.img_dir, use_clip=False, output_json="artifacts/pascal_baseline.json")

    # 2. IntelliSegment Evaluation
    run_pascal_evaluation(args.gt_json, args.img_dir, use_clip=True, output_json="artifacts/pascal_intellisegment.json")
