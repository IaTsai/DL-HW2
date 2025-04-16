import os
import torch
import json
from PIL import Image
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
import transforms as T
import utils
from tqdm import tqdm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import random
import numpy as np
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.imgs = sorted([f for f in os.listdir(root) if f.endswith(".png")])
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        image_id = int(os.path.splitext(self.imgs[idx])[0])
        if self.transform:
            img = self.transform(img)
        return img, image_id

    def __len__(self):
        return len(self.imgs)


def get_model(num_classes, model_path):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # BackBone 5: Use ResNeXt50_32x4d + FPN
    # backbone = resnet_fpn_backbone(
    #     'resnext50_32x4d',
    #     weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1
    # )
    # model = torchvision.models.detection.FasterRCNN(
    #     backbone, num_classes=num_classes
    # )

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


@torch.no_grad()
def predict(
    model,
    data_loader,
    output_file="pred.json",
    score_threshold=0.7
):
    results = []
    for imgs, image_ids in tqdm(data_loader, desc="Predicting-test/"):
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        for output, image_id in zip(outputs, image_ids):
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
                    float(x1), float(y1), float(width), float(height)
                ]
                result["score"] = float(score)
                result["category_id"] = int(label)
                results.append(result)

    # Sort results before saving
    results = sorted(results, key=lambda x: x["image_id"])

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"\u2705 Saved predictions to {output_file}")


def main():
    test_dir = "nycu-hw2-data/test"
    model_path = "nycu_hw2_output/0416/Re2Col_ResNet50V2_model.pth"

    transform = T.get_transform()
    dataset = TestDataset(test_dir, transform=transform)
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
