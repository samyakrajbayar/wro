# LEGO Object Detection — Offline AI Model

A **YOLOv8-based object detection system** that runs fully offline on a car-mounted camera. Detects and labels 6 classes of LEGO objects with automatic color differentiation.

## 🎯 Detection Classes

| Class | Description | Color Detection |
|-------|-------------|-----------------|
| `lego_block` | Cubic/rectangular LEGO blocks | ✅ Yellow, Green, Blue, White |
| `lego_rod` | Long thin LEGO rods/beams | ✅ Yellow, Green, Blue, White |
| `barrier` | Assembled barrier structures | ❌ |
| `rectangular_trowel` | Red rectangular trowel tool | ❌ |
| `cement_bowl` | White/grey cement bowl | ❌ |
| `masonry_trowel` | White+red masonry trowel | ❌ |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect & Label Training Data
```bash
# Put your raw photos in data/raw/
# Then label them:
python scripts/label_tool.py --images data/raw/ --output data/
```

### 3. Augment Dataset (30x multiplier)
```bash
python scripts/generate_augmented_data.py \
    --input data/images/train \
    --labels data/labels/train \
    --output data/augmented \
    --multiplier 30
```

### 4. Train the Model
```bash
# Standard training (recommended)
python train.py --model yolov8s.pt --epochs 200 --batch 16

# For Raspberry Pi (faster, lighter model)
python train.py --model yolov8n.pt --epochs 200 --batch 16

# For powerful GPU (best accuracy)
python train.py --model yolov8m.pt --epochs 300 --batch 16
```

### 5. Run Detection
```bash
# Live camera
python detect.py

# Video file
python detect.py --source video.mp4

# Single image
python scripts/detect_image.py --source photo.jpg --save

# Custom model weights
python detect.py --weights runs/detect/lego_detector/weights/best.pt
```

## 📁 Project Structure

```
wro code/
├── config.yaml                    # YOLOv8 dataset config
├── requirements.txt               # Python dependencies
├── train.py                       # Training script
├── detect.py                      # Real-time detection (main)
├── color_classifier.py            # HSV color classification
├── data/
│   ├── README.md                  # Dataset instructions
│   ├── images/{train,val,test}/   # Training images
│   ├── labels/{train,val,test}/   # YOLO format labels
│   └── raw/                       # Your original photos
├── scripts/
│   ├── label_tool.py              # Bounding box annotation tool
│   ├── generate_augmented_data.py # Data augmentation pipeline
│   ├── validate.py                # Model evaluation
│   └── detect_image.py            # Single image detection
├── utils/
│   ├── visualization.py           # Drawing utilities
│   └── camera.py                  # Camera abstraction
├── tests/
│   └── test_color_classifier.py   # Color classifier tests
└── runs/                          # Training output (auto-created)
    └── detect/lego_detector/
        └── weights/best.pt        # Trained model
```

## 🎮 Detection Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `r` | Toggle recording |
| `c` | Toggle crosshair |
| `+/-` | Adjust confidence threshold |
| `SPACE` | Pause/resume |

## 🔧 Architecture

```
Camera → YOLOv8 Detection → Color Classifier (HSV) → Labeled Output
              │                       │
       (offline .pt)          (rule-based, no ML)
```

- **YOLOv8** handles object detection (shape/type recognition)
- **HSV Color Classifier** identifies block/rod colors deterministically
- **No internet required** at inference time — fully offline

## 📊 Tips for Maximum Accuracy

1. **Photograph from many angles** — especially from above, 45°, and side views
2. **Use varied lighting** — indoor, outdoor, bright, dim
3. **Include mixed scenes** — multiple objects per frame
4. **Augment heavily** — use 30-50x multiplier
5. **Train for 200+ epochs** — early stopping will find optimal point
6. **Keep HSV hue augmentation LOW** (0.02) — preserves color accuracy
