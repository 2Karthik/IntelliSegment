# IntelliSegment: A Sequential Framework for Multi-Object Detection and Mask Generation using YOLO-NAS, CLIP and SAM2

Overview

IntelliSegment is a sequential computer vision framework designed to combine object detection, vision-language semantic verification, and prompt-based segmentation in a unified pipeline.

The framework integrates:

YOLO-NAS for object detection and bounding-box generation.

CLIP for semantic verification of detected image regions using category-based text prompts.

SAM2 (Segment Anything Model 2) for generating segmentation masks from verified bounding-box prompts.

The objective is to combine object localization, semantic consistency checking, and precise mask generation into a modular pipeline.

Datasets

The framework is intended to support evaluation using the following datasets:

PASCAL VOC: Commonly used for object detection and semantic segmentation evaluation.

ECSSD: A salient-object segmentation dataset used for evaluating foreground segmentation quality.

---

## 📌 Pipeline Architecture

```
Input Image ──► YOLO-NAS (Backbone + FPN) ──► Raw BBoxes & Scores
                                                  │
                                                  ▼
                                       CLIP Visual-Text Verifier
                                      (Filters False Positives)
                                                  │
                                                  ▼
SAM2 Mask Segmentor ◄────────────────── Verified Box Prompts
         │
         ▼
Final Instance Masks
```


## 🚀 Quick Start & Installation

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/anonymous-author/IntelliSegment.git
cd IntelliSegment
pip install -r requirements.txt
```

### 2. Download Pre-trained Checkpoints
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### 3. Generate Prediction Artifacts
The prediction outputs in `artifacts/`:
```bash
python generate_artifacts.py
```

### 4. Run Complete Evaluation
```bash
python evaluate.py --gt_json data/pascal_voc2012_val.json --img_dir data/VOCdevkit/VOC2012/JPEGImages
```

---

## 📁 Repository Structure
* `intellisegment_pipeline.py`: Core inference wrapper combining YOLO-NAS, CLIP, and SAM2.
* `metrics.py`: Standard instance segmentation ($AP$) and perceptual ($S_\alpha, E_\phi, MAE$) evaluation utilities.
* `evaluate.py`: Standard evaluation runner on PASCAL VOC validation benchmark.
* `generate_artifacts.py`: Serializes predictions (`pascal_baseline.json`, `pascal_intellisegment.json`) for verification.
* `artifacts/`: Pre-computed prediction files.
