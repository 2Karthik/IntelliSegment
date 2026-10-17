# IntelliSegment: Multi-Object Detection and Mask Generation

Overview

IntelliSegment is a sequential computer vision framework designed to combine object detection, vision-language semantic verification, and prompt-based segmentation in a unified pipeline.

The framework integrates:

YOLO-NAS for object detection and bounding-box generation.

CLIP for semantic verification of detected image regions using category-based text prompts.

SAM2 (Segment Anything Model 2) for generating segmentation masks from verified bounding-box prompts.

The objective is to combine object localization, semantic consistency checking, and precise mask generation into a modular pipeline.

Architecture

The IntelliSegment pipeline follows these stages:

Input Image: Receives the image for processing.

Object Detection (YOLO-NAS): Detects candidate objects and generates bounding boxes, class labels, and confidence scores.

Text Prompt Generation: Creates category-specific text prompts corresponding to detected objects.

Semantic Verification (CLIP): Compares detected image crops with text prompts to assess semantic consistency.

Prompt-Based Segmentation (SAM2): Uses verified bounding boxes as prompts to generate object masks.

Post-processing and Evaluation: Saves segmentation masks, visualization overlays, and evaluation results.

Datasets

The framework is intended to support evaluation using the following datasets:

PASCAL VOC: Commonly used for object detection and semantic segmentation evaluation.

ECSSD: A salient-object segmentation dataset used for evaluating foreground segmentation quality.

The datasets have different annotation formats and evaluation protocols. Appropriate preprocessing and ground-truth handling are required for each dataset.

Evaluation Metrics

PASCAL VOC
Depending on the experimental setup, the following metrics may be used:
Mean Intersection over Union (mIoU)
Dice coefficient (mDice)
F1-score

ECSSD
The following metrics are relevant to salient-object segmentation:
S-measure ((S_\alpha))
E-measure ((E_\phi))
Mean Absolute Error (MAE)

Installation

1. Install SAM2

Install SAM2 according to the official Meta implementation and its compatibility requirements.

2. Install IntelliSegment dependencies- as required

Running Inference

pip install super-gradients ftfy regex tqdm pycocotools opencv-python torch torchvision
# Install SAM2 from official repository
pip install git+https://github.com/facebookresearch/segment-anything-2.git

Reproducibility and Limitations

This repository provides a reference implementation reconstructed from the IntelliSegment method description.

Before using it to reproduce or report experimental results:

Verify compatibility among Python, PyTorch, SuperGradients, Transformers, and SAM2.

Citation

If you use IntelliSegment in your research, please cite the associated paper 
  title   = {IntelliSegment: A Sequential Framework for Multi-Object Detection and Mask Generation using YOLO-NAS, CLIP and SAM2},

Acknowledgements

This framework builds upon the research and software ecosystems of YOLO-NAS, CLIP, and Segment Anything Model 2.
