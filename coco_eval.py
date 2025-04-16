from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import random
import numpy as np
import torch


# Force reproducibility for all randomness
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using GPU, also set cudnn behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Apply seed setup
set_seed(42)


def main():
    # Ground truth annotations (validation set)
    ann_file = "nycu-hw2-data/valid.json"
    # Prediction results (from my model)
    pred_file = "pred.json"

    # Load ground truth annotations
    coco_gt = COCO(ann_file)

    # Load prediction results
    coco_dt = coco_gt.loadRes(pred_file)

    # Create COCO evaluator for bbox prediction
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    # Specify image IDs to evaluate (all images)
    coco_eval.params.imgIds = sorted(coco_gt.getImgIds())

    # Run evaluation (automatically computes IoU, precision, recall)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # Print mAP / AP50 / AP75, etc.


if __name__ == "__main__":
    main()
