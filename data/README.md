# Dataset Guide — LEGO Object Detection

## Directory Structure

```
data/
├── images/
│   ├── train/          ← Training images (80%)
│   ├── val/            ← Validation images (10%)
│   └── test/           ← Test images (10%)
├── labels/
│   ├── train/          ← YOLO format labels for training
│   ├── val/            ← YOLO format labels for validation
│   └── test/           ← YOLO format labels for testing
└── raw/                ← Your original, unlabeled photos
```

## How to Collect Training Data

### Step 1: Photograph Your Objects
Take **50–100+ photos per class** with your phone/camera:

| Class ID | Object | Tips |
|----------|--------|------|
| 0 | LEGO Block | Photograph each color (yellow, green, blue, white) separately AND mixed together |
| 1 | LEGO Rod | Scatter rods randomly, photograph from above, side, and angle |
| 2 | Barrier | Full assembled barriers from multiple distances and angles |
| 3 | Rectangular Trowel | Various orientations on different backgrounds |
| 4 | Cement Bowl | Top-down and angled views |
| 5 | Masonry Trowel | Various orientations |

### Photo Tips for Maximum Accuracy
- **Angles**: Photograph from above (bird's eye), 45°, and eye-level
- **Distances**: Close-up (30cm), medium (1m), and far (2–3m)
- **Lighting**: Indoor, outdoor, bright, dim, shadow conditions
- **Backgrounds**: Different surfaces (table, floor, carpet, road)
- **Multiple objects**: Include frames with 2–5 objects for multi-detection training
- **Motion blur**: Take some photos while moving the camera (simulates car movement)

### Step 2: Label Your Images
Use the built-in labeling tool:
```bash
python scripts/label_tool.py --images data/raw/ --output data/
```

Or use [LabelImg](https://github.com/HumanSignal/labelImg) (set to YOLO format).

### YOLO Label Format
Each image needs a matching `.txt` file with one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized (0.0 to 1.0). Example:
```
0 0.45 0.50 0.12 0.15
1 0.70 0.30 0.25 0.05
```

### Step 3: Augment Your Dataset
After labeling, multiply your dataset 20–50x:
```bash
python scripts/generate_augmented_data.py --input data/images/train --labels data/labels/train --output data/augmented --multiplier 30
```

### Step 4: Split Into Train/Val/Test
The augmented images will be auto-split 80/10/10 into train/val/test folders.
