import json
import pandas as pd
from collections import defaultdict

# === Load predict.json ===
with open("pred.json", "r") as f:
    preds = json.load(f)

# Create mapping: image_id -> list of (x, category_id)
image_predictions = defaultdict(list)
for pred in preds:
    image_id = pred["image_id"]
    # originally 1~10, representing digits 0~9
    category_id = pred["category_id"]
    x_coord = pred["bbox"][0]  # x coordinate
    image_predictions[image_id].append((x_coord, category_id))

# Assume you have 3340 images in the valid set, image_id from 1 to 3340
results = []
for image_id in range(1, 3341):
    preds = image_predictions.get(image_id, [])
    if len(preds) == 0:
        pred_str = "-1"
    else:
        # Sort predictions by x coordinate
        preds.sort(key=lambda x: x[0])
        # category_id - 1 → convert to string
        digits = [str(p[1] - 1) for p in preds]
        pred_str = "".join(digits)

    results.append({"image_id": image_id, "pred_label": pred_str})

# Save as CSV
df = pd.DataFrame(results)
df.to_csv("pred.csv", index=False)
print("Done! Saved to pred.csv")
