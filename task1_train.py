import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import utils
import transforms as T  # custom get_transform function from transforms.py
from torch.utils.data import Subset  # split sub-set for rapid testing
from tqdm import tqdm  # show progress bar during epoch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
import json
import tempfile
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import random
import numpy as np
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnext50_32x4d
from torchvision.models import ResNeXt50_32X4D_Weights


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Call seed setting
set_seed(42)

device = \
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ==== Dataset ====
def get_dataset(root, annFile, train=True):
    dataset = CocoDetection(
        root=root, annFile=annFile, transform=T.get_transform(train=train)
        )

    # Check transform (print log once)
    if not hasattr(get_dataset, "_checked"):
        img_sample, _ = dataset[0]
        print(
            f"[Transform Check] Image Type: {type(img_sample)}, "
            f"min: {img_sample.min():.2f}, max: {img_sample.max():.2f}"
        )
        get_dataset._checked = True

    return dataset


# ==== Model ====
def get_model(num_classes):
    # BackBone 2: ResNet50V2 with pretrained weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Replace classifier head to match num_classes
    # (background + digits 0~9 = 11 classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    return model


# ==== Training ====
def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(data_loader, desc="Epoch", leave=False)

    for i, (images, targets) in enumerate(loop):
        print(f"[train][Epoch {epoch}] Batch {i}, Images: {len(images)}")

        images = list(img.to(device) for img in images)
        new_targets = []
        for j, t in enumerate(targets):
            if len(t) == 0:
                print(f"[Warning] Image {j} has no annotations (skipped)")
                continue

            boxes = []
            labels = []
            for obj in t:
                x, y, w, h = obj['bbox']
                if w <= 0 or h <= 0:
                    continue
                if w > 3000 or h > 3000:
                    print(
                        f"[Warning] Image ID {t[0]['image_id']} "
                        f"has large box: w={w}, h={h}"
                        )
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                boxes.append([x1, y1, x2, y2])
                labels.append(obj['category_id'])
                if obj['category_id'] < 0 or obj['category_id'] > 10:
                    print(
                        f"[Error] Image ID {t[0]['image_id']} "
                        "has invalid label: "
                        f"{obj['category_id']}"
                    )

            if len(boxes) == 0:
                print(f"[Warning] Image {j} has no valid bbox (skipped)")
                continue

            image_id = torch.tensor(
                [t[0]["image_id"]],
                dtype=torch.int64
            ).to(device)

            d = {
                'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
                'labels': torch.tensor(labels, dtype=torch.int64).to(device),
                'image_id': image_id
            }

            new_targets.append(d)

        if len(new_targets) == 0:
            continue

        loss_dict = model(images, new_targets)
        losses = sum(loss for loss in loss_dict.values())

        all_boxes = [b for t in new_targets for b in t['boxes']]
        if all_boxes:
            areas = [
                (box[2] - box[0]) * (box[3] - box[1])
                for box in all_boxes
                ]
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        avg_loss = running_loss / (i + 1)
        loop.set_postfix(loss=avg_loss)
        print(
            f"[train] Batch {i}, loss: {losses.item():.4f}, "
            f"avg: {avg_loss:.4f}"
            )
    print(f"[train_one_epoch] Epoch avg loss: {avg_loss:.4f}")


# ==== Accuracy Evaluation ====
@torch.no_grad()
def evaluate(model, data_loader_val, device):
    model.eval()
    correct = 0
    total = 0
    for images, targets in data_loader_val:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            preds = output['labels'].cpu().numpy()
            gts = [obj['category_id'] for obj in target]
            total += len(gts)
            correct += sum([1 for p, g in zip(preds, gts) if p == g])

    acc = correct / total if total > 0 else 0
    return acc


# ==== mAP Evaluation ====
@torch.no_grad()
def evaluate_map(model, data_loader, device, score_threshold=0.7):
    model.eval()
    coco_gt = data_loader.dataset.coco

    coco_results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            image_id = target[0]["image_id"]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                print(
                    f"[evaluate_map] image_id={image_id}, boxes={len(boxes)}, "
                    f"max score={scores.max() if len(scores) > 0 else 'N/A'}"
                    )
                x1, y1, x2, y2 = box
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1)
                        ],
                    "score": float(score)
                })

    if len(coco_results) == 0:
        print("[evaluate_map] No predictions made! "
              "Model may have failed to detect any boxes.")
        return 0.0

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as f:
        json.dump(coco_results, f)
        f.flush()

        coco_dt = coco_gt.loadRes(f.name)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]


# ==== Main Function ====
def main():
    train_dir = 'nycu-hw2-data/train'
    train_json = 'nycu-hw2-data/train.json'
    val_dir = 'nycu-hw2-data/valid'
    val_json = 'nycu-hw2-data/valid.json'

    batch_size = 4
    dataset = get_dataset(train_dir, train_json)
    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn
        )
    dataset_val_full = get_dataset(val_dir, val_json, train=False)
    data_loader_val = DataLoader(
        dataset_val_full,
        batch_size,
        shuffle=False,
        collate_fn=utils.collate_fn
        )

    model = get_model(num_classes=11)
    model.to(device)

    print("\nTrainable Parameter Check:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[Trainable] {name}")
        else:
            print(f"[Frozen]    {name}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    patience = 10
    early_stop_counter = 0
    num_epochs = 30
    best_map = 0.0

    for epoch in range(num_epochs):
        print("=" * 40)
        print(f"🚀 Starting Epoch {epoch+1}/{num_epochs}")
        print(f"📦 Trainset size: {len(data_loader.dataset)}")
        print(f"📦 Valset size: {len(data_loader_val.dataset)}")

        train_one_epoch(model, data_loader, optimizer, device, epoch)
        print(f"[Epoch {epoch+1}] Finished training epoch.")

        map_score = evaluate_map(
            model,
            DataLoader(
                dataset_val_full,
                batch_size,
                shuffle=False,
                collate_fn=utils.collate_fn
            ),
            device)
        print(f"[Epoch {epoch+1}] Validation mAP: {map_score:.4f}")

        if map_score == 0.0:
            print(
                "mAP is 0. The model may have failed to predict boxes "
                "or dataset format is incorrect."
            )

        if map_score > best_map:
            best_map = map_score
            early_stop_counter = 0
            state_dict = model.state_dict()
            output_path = "nycu_hw2_output/Re2Col_ResNet50V2_model.pth"
            torch.save(state_dict, output_path)
            print(f"Saved best model with mAP {map_score:.4f}")
        else:
            early_stop_counter += 1
            print(
                "No mAP improvement, "
                f"early_stop_counter={early_stop_counter}/{patience}"
                )
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break

        scheduler.step()


if __name__ == '__main__':
    main()
