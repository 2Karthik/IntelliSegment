# IntelliSegment: A Sequential Framework for Multi-Object Detection and Mask Generation using YOLO-NAS, CLIP and SAM2

Official repository for **IntelliSegment**, a hybrid visual framework integrating object detection (YOLO-NAS), zero-shot semantic verification (CLIP), and fine-grained promptable mask generation (SAM2).

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

---

## 📊 Experimental Results (PASCAL VOC 2012 Validation Set)

| Method | Component Breakdown | $AP_{50}^{box}$ (%) | $AP_{50:95}^{box}$ (%) | $AP_{50}^{mask}$ (%) | $AP_{50:95}^{mask}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLO-NAS Baseline** | YOLO-NAS | 81.4 | 58.6 | — | — |
| **Ablation Baseline** | YOLO-NAS + SAM2 | 81.4 | 58.6 | 76.8 | 52.3 |
| **IntelliSegment (Ours)** | **YOLO-NAS + CLIP + SAM2** | **84.2** | **61.8** | **80.3** | **56.1** |

---

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
To recreate the serialized COCO-formatted prediction outputs in `artifacts/`:
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
* `evaluate.py`: Standard evaluation runner on PASCAL VOC 2012 validation benchmark.
* `generate_artifacts.py`: Serializes predictions (`pascal_baseline.json`, `pascal_intellisegment.json`) for verification.
* `artifacts/`: Pre-computed prediction files submitted for reviewer validation.
