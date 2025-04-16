# NYCU Computer Vision 2025 Spring HW2

**StudentID:** 313553058\
**Name:** Ian Tsai

---

## Introduction

This project addresses two core tasks in digit-level image understanding using object detection and recognition pipelines:

- **Task 1**: Detect all digits in an image (bounding boxes + class labels)
- **Task 2**: Assemble detected digits in left-to-right order as a number string

We employ a Faster R-CNN based pipeline to complete the tasks with optional evaluation utilities.

---

## Architecture Summary

The current solution uses the **Faster R-CNN** architecture with customizable backbones. The best-performing model weights are saved for inference and downstream recognition conversion.

---

## How to Install

Set up your environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate _FastRcnn
```

---

## How to Run

### Step 1. Train the Detection Model

```bash
python task1_train.py
```

- **Input**: `nycu-hw2-data/train/*.png` + `train.json`
- **Output**: `nycu_hw2_output/best_model.pth`

> Re-run this if you change the model backbone, head, or any training parameter.

---

### Step 2. Inference on Validation Set

```bash
python predict.py
```

- **Input**: `nycu-hw2-data/valid/*.png` + `best_model.pth`
- **Output**: `pred.json` (COCO-style, for validation set)

> This `pred.json` is used for offline mAP evaluation using `coco_eval.py`.

---

### Step 3. (Optional) Evaluate Detection Performance

```bash
python coco_eval.py
```

- **Input**: `valid.json` (ground truth) + `pred.json` (from validation set)
- **Output**: Print mAP / AP50 / AP75

---

### Step 4. Inference on Test Set

```bash
python JsonTest.py
```

- **Input**: `nycu-hw2-data/test/*.png` + `best_model.pth`
- **Output**: `pred.json` (for test set)

---

### Step 5. Convert to Recognition Format for Task 2

```bash
python Gen_pred-csv.py
```

- **Input**: `pred.json` (from test set inference)
- **Output**: `pred.csv`

> This CSV represents the digit string prediction for each image. If no digits are detected, `-1` is output.

---

## Pipeline Diagram

```text
           Train          --->     Inference (val/test)         --->     Recognition + Submission
      [task1_train.py]      [predict.py / JsonTest.py]                   [Gen_pred-csv.py]
             |                      |                                            |
     best_model.pth         →   pred.json (val/test)            → pred.csv → Task2 Submission
                                      |
                         [coco_eval.py]（optional self-check for Task1）
```

---

## Summary

| Script            | Purpose                          | Input                       | Output           |
| ----------------- | -------------------------------- | --------------------------- | ---------------- |
| `task1_train.py`  | Train detection model            | train images + annotations  | `best_model.pth` |
| `predict.py`      | Inference on validation images   | val images + model weights  | `pred.json`      |
| `coco_eval.py`    | Evaluate detection performance   | `valid.json` + `pred.json`  | printed metrics  |
| `JsonTest.py`     | Inference on test images         | test images + model weights | `pred.json`      |
| `Gen_pred-csv.py` | Convert detection to recognition | `pred.json`                 | `pred.csv`       |
