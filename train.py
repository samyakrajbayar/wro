"""
YOLOv8 Training Script — LEGO Object Detection

Trains a YOLOv8 model on your labeled LEGO dataset.
Uses transfer learning from COCO pretrained weights for fast convergence.

Usage:
    python train.py                          # Train with defaults
    python train.py --model yolov8n.pt       # Use nano model (faster, less accurate)
    python train.py --model yolov8m.pt       # Use medium model (slower, more accurate)
    python train.py --epochs 300 --batch 8   # Custom settings
    python train.py --resume                 # Resume interrupted training
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for LEGO Object Detection"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8s.pt",
        help="Base model: yolov8n.pt (nano/fast), yolov8s.pt (small/balanced), "
             "yolov8m.pt (medium/accurate). Default: yolov8s.pt"
    )
    parser.add_argument(
        "--data", type=str, default="config.yaml",
        help="Path to dataset config YAML. Default: config.yaml"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs. Default: 200"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size. Reduce if you run out of GPU memory. Default: 16"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image size. Default: 640"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device: '' (auto), 'cpu', '0' (GPU 0), '0,1' (multi-GPU). Default: auto"
    )
    parser.add_argument(
        "--name", type=str, default="lego_detector",
        help="Experiment name for saving results. Default: lego_detector"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Early stopping patience (epochs with no improvement). Default: 30"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of data loading workers. Default: 4"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export model to ONNX after training"
    )
    return parser.parse_args()


def check_dataset(data_yaml: str):
    """Verify dataset exists and has images."""
    if not os.path.isfile(data_yaml):
        print(f"\n❌ ERROR: Dataset config not found: {data_yaml}")
        print("Make sure you have created config.yaml and prepared your dataset.")
        print("See data/README.md for instructions.")
        sys.exit(1)

    import yaml
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg.get("path", ".")
    train_dir = os.path.join(data_root, cfg.get("train", "images/train"))

    if not os.path.isdir(train_dir):
        print(f"\n⚠️  WARNING: Training image directory not found: {train_dir}")
        print("Creating directory structure...")
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(data_root, "images", split), exist_ok=True)
            os.makedirs(os.path.join(data_root, "labels", split), exist_ok=True)
        print(f"✅ Created directories under {data_root}/")
        print("Add your labeled images before training. See data/README.md.\n")
        return False

    # Count images
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in Path(train_dir).iterdir() if f.suffix.lower() in img_exts]
    print(f"\n📊 Dataset: {len(images)} training images found in {train_dir}")

    if len(images) < 10:
        print("⚠️  WARNING: Very few training images. Accuracy will be low.")
        print("Recommend at least 50+ images per class (300+ total).")
        print("Use scripts/generate_augmented_data.py to multiply your dataset.\n")

    return len(images) > 0


def train(args):
    """Run YOLOv8 training."""
    from ultralytics import YOLO

    print("=" * 60)
    print("  LEGO Object Detection — YOLOv8 Training")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  Dataset:   {args.data}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch}")
    print(f"  Image size:{args.imgsz}")
    print(f"  Device:    {args.device or 'auto'}")
    print("=" * 60)

    # Check dataset
    has_data = check_dataset(args.data)
    if not has_data:
        print("❌ No training images found. Please add images first.")
        print("See data/README.md for instructions.")
        sys.exit(1)

    # Load model (downloads pretrained weights if not cached)
    if args.resume:
        # Resume from last checkpoint
        model = YOLO(f"runs/detect/{args.name}/weights/last.pt")
        print("\n🔄 Resuming from last checkpoint...")
    else:
        model = YOLO(args.model)
        print(f"\n📦 Loaded base model: {args.model}")

    # ============================================================
    # Training with aggressive augmentation for multi-angle robustness
    # ============================================================
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        exist_ok=True,

        # === Augmentation (critical for car-mounted camera) ===
        hsv_h=0.02,       # Hue augmentation (keep low for color accuracy)
        hsv_s=0.7,        # Saturation augmentation
        hsv_v=0.5,        # Value/brightness augmentation
        degrees=15.0,     # Random rotation ±15°
        translate=0.2,    # Random translation ±20%
        scale=0.5,        # Random scale ±50%
        shear=5.0,        # Shear ±5°
        perspective=0.001, # Perspective warp (simulates angle changes)
        flipud=0.2,       # Vertical flip probability
        fliplr=0.5,       # Horizontal flip probability
        mosaic=1.0,       # Mosaic augmentation (combines 4 images)
        mixup=0.15,       # MixUp augmentation
        copy_paste=0.1,   # Copy-paste augmentation

        # === Training settings ===
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        warmup_momentum=0.5,
        weight_decay=0.0005,
        cos_lr=True,      # Cosine learning rate scheduler

        # === Saving ===
        save=True,
        save_period=10,   # Save checkpoint every 10 epochs
        plots=True,       # Generate training plots
        val=True,         # Validate after each epoch
    )

    print("\n" + "=" * 60)
    print("  ✅ Training Complete!")
    print("=" * 60)
    print(f"  Best model:  runs/detect/{args.name}/weights/best.pt")
    print(f"  Last model:  runs/detect/{args.name}/weights/last.pt")
    print(f"  Results:     runs/detect/{args.name}/")
    print("=" * 60)

    # Export to ONNX for deployment
    if args.export:
        print("\n📦 Exporting to ONNX format...")
        best_model = YOLO(f"runs/detect/{args.name}/weights/best.pt")
        best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"✅ ONNX model saved to runs/detect/{args.name}/weights/best.onnx")

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
