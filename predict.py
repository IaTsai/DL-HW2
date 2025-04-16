import os
import torch
import json
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision
import transforms as T       # your custom transforms.py
import utils                 # your custom utils.py (includes collate_fn)
from tqdm import tqdm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import random
import numpy as np
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnext50_32x4d
from torchvision.models import ResNeXt50_32X4D_Weights


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

# Determine whether to use GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load validation dataset (COCO format)
def get_dataset(root, annFile, train=False):
    return CocoDetection(
        root=root,
        annFile=annFile,
        transform=T.get_transform(train=train)
    )


# Build the model and load trained weights
def get_model(num_classes, model_path):
    # BackBone1:
    # model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # BackBone2:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Replace classifier head to match num_classes
    # (background + digits 0~9 = 11 classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Inference and save pred.json
@torch.no_grad()
def predict(
    model,
    data_loader,
    output_file="pred.json",
    score_threshold=0.7
):
    results = []

    for imgs, targets in tqdm(data_loader, desc="Predicting-val"):
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        for output, target in zip(outputs, targets):
            if len(target) == 0:
                continue

            image_id = target[0].get("image_id", None)
            if image_id is None:
                continue

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue

                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                result = OrderedDict()
                result["image_id"] = int(image_id)
                result["bbox"] = [
                    float(x1),
                    float(y1),
                    float(width),
                    float(height)
                ]
                result["score"] = float(score)
                result["category_id"] = int(label)
                results.append(result)

    results = sorted(results, key=lambda x: x["image_id"])

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Saved predictions to {output_file}")


# Main entry point
def main():
    val_dir = "nycu-hw2-data/valid"
    val_json = "nycu-hw2-data/valid.json"
    model_path = "nycu_hw2_output/0416/Re2Col_ResNet50V2_model.pth"

    dataset = get_dataset(val_dir, val_json, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    model = get_model(num_classes=11, model_path=model_path)
    predict(model, data_loader)


if __name__ == "__main__":
    main()
