"""
Data Augmentation Pipeline — Generate augmented training data.

Takes labeled images (YOLO format) and creates augmented versions with
correct bounding box transformations. Essential for training a robust
model that works from all angles.

Usage:
    python scripts/generate_augmented_data.py \\
        --input data/images/train \\
        --labels data/labels/train \\
        --output data/augmented \\
        --multiplier 30
"""

import argparse
import os
import sys
import random
import shutil
import cv2
import numpy as np
from pathlib import Path

try:
    import albumentations as A
    from albumentations import BboxParams
except ImportError:
    print("❌ albumentations not installed. Run: pip install albumentations")
    sys.exit(1)


# ============================================================
# Augmentation Pipeline — tuned for car-mounted camera robustness
# ============================================================

def create_augmentation_pipeline():
    """
    Creates a heavy augmentation pipeline optimized for:
    - Multi-angle robustness (rotation, perspective, flip)
    - Lighting variation (brightness, contrast, hue)
    - Motion artifacts (blur, noise)
    - Scale variation (resize, crop)
    """
    return A.Compose([
        # === Geometric transforms (multi-angle robustness) ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        A.Affine(
            scale=(0.7, 1.3),
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.5
        ),

        # === Color/lighting transforms ===
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=40, p=1.0
            ),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.7),

        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
        ], p=0.3),

        # === Noise & blur (simulating movement) ===
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.2),

        # === Occlusion simulation ===
        A.CoarseDropout(
            max_holes=3, max_height=30, max_width=30,
            min_holes=1, min_height=10, min_width=10,
            fill_value=0, p=0.15
        ),

        # === Weather/environmental ===
        A.OneOf([
            A.RandomShadow(p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
        ], p=0.1),

        # === Final resize (maintain consistency) ===
        A.LongestMaxSize(max_size=640, p=1.0),
        A.PadIfNeeded(
            min_height=640, min_width=640,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
        ),

    ], bbox_params=BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,  # Drop boxes that become mostly hidden
        min_area=100,        # Drop very small boxes
    ))


def read_yolo_labels(label_path: str):
    """
    Read YOLO format labels.

    Returns:
        bboxes: List of [x_center, y_center, width, height] (normalized)
        class_labels: List of class IDs (int)
    """
    bboxes = []
    class_labels = []

    if not os.path.isfile(label_path):
        return bboxes, class_labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Clamp values to valid range
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))
                w = max(0.001, min(0.999, w))
                h = max(0.001, min(0.999, h))
                # Ensure bbox doesn't exceed image bounds
                if x - w/2 < 0: w = 2 * x
                if y - h/2 < 0: h = 2 * y
                if x + w/2 > 1: w = 2 * (1 - x)
                if y + h/2 > 1: h = 2 * (1 - y)
                bboxes.append([x, y, w, h])
                class_labels.append(cls_id)

    return bboxes, class_labels


def write_yolo_labels(label_path: str, bboxes: list, class_labels: list):
    """Write YOLO format labels."""
    with open(label_path, "w") as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            x, y, w, h = bbox
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def split_dataset(images_dir: str, labels_dir: str, output_root: str,
                  train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Split augmented dataset into train/val/test sets.
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in Path(images_dir).iterdir() if f.suffix.lower() in img_exts]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        img_dir = os.path.join(output_root, "images", split_name)
        lbl_dir = os.path.join(output_root, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_path in split_images:
            # Copy image
            shutil.copy2(str(img_path), img_dir)

            # Copy label
            label_name = img_path.stem + ".txt"
            label_src = os.path.join(labels_dir, label_name)
            if os.path.isfile(label_src):
                shutil.copy2(label_src, lbl_dir)

        print(f"  {split_name}: {len(split_images)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented training data from labeled images"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input images directory (with YOLO labels in parallel labels/ dir)"
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Labels directory. Default: sibling 'labels/' matching 'images/' structure"
    )
    parser.add_argument(
        "--output", type=str, default="data/augmented",
        help="Output directory for augmented data. Default: data/augmented"
    )
    parser.add_argument(
        "--multiplier", type=int, default=30,
        help="Number of augmented copies per original image. Default: 30"
    )
    parser.add_argument(
        "--split", action="store_true", default=True,
        help="Auto-split into train/val/test after augmentation"
    )
    parser.add_argument(
        "--split-output", type=str, default="data",
        help="Root directory for split output. Default: data"
    )
    args = parser.parse_args()

    # Resolve labels directory
    if args.labels is None:
        # Infer: if input is data/images/train, labels = data/labels/train
        args.labels = str(Path(args.input).parent.parent / "labels" / Path(args.input).name)

    # Validate
    if not os.path.isdir(args.input):
        print(f"❌ Input directory not found: {args.input}")
        sys.exit(1)

    if not os.path.isdir(args.labels):
        print(f"⚠️  Labels directory not found: {args.labels}")
        print("Labels are required for augmentation with correct bbox transforms.")
        sys.exit(1)

    # Output dirs
    aug_img_dir = os.path.join(args.output, "images")
    aug_lbl_dir = os.path.join(args.output, "labels")
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_lbl_dir, exist_ok=True)

    # Find images
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in Path(args.input).iterdir() if f.suffix.lower() in img_exts]

    if not images:
        print(f"❌ No images found in {args.input}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  LEGO Data Augmentation Pipeline")
    print(f"{'='*50}")
    print(f"  Input:      {args.input} ({len(images)} images)")
    print(f"  Labels:     {args.labels}")
    print(f"  Multiplier: {args.multiplier}x")
    print(f"  Output:     {args.output}")
    print(f"  Expected:   ~{len(images) * (args.multiplier + 1)} total images")
    print(f"{'='*50}\n")

    # Create augmentation pipeline
    transform = create_augmentation_pipeline()

    total_generated = 0
    total_skipped = 0

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Cannot read: {img_path.name}")
            continue

        # Read labels
        label_path = os.path.join(args.labels, img_path.stem + ".txt")
        bboxes, class_labels = read_yolo_labels(label_path)

        # Copy original
        orig_name = f"{img_path.stem}_orig{img_path.suffix}"
        cv2.imwrite(os.path.join(aug_img_dir, orig_name), img)
        write_yolo_labels(
            os.path.join(aug_lbl_dir, f"{img_path.stem}_orig.txt"),
            bboxes, class_labels
        )
        total_generated += 1

        # Generate augmented copies
        for j in range(args.multiplier):
            try:
                if bboxes:
                    augmented = transform(
                        image=img,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                else:
                    # No labels — just augment the image
                    simple_transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.Rotate(limit=30, p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        A.LongestMaxSize(max_size=640),
                        A.PadIfNeeded(min_height=640, min_width=640,
                                      border_mode=cv2.BORDER_CONSTANT, value=0),
                    ])
                    augmented = simple_transform(image=img)
                    augmented["bboxes"] = []
                    augmented["class_labels"] = []

                # Save augmented image
                aug_name = f"{img_path.stem}_aug{j:03d}.jpg"
                cv2.imwrite(
                    os.path.join(aug_img_dir, aug_name),
                    augmented["image"],
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                # Save augmented labels
                write_yolo_labels(
                    os.path.join(aug_lbl_dir, f"{img_path.stem}_aug{j:03d}.txt"),
                    augmented["bboxes"],
                    augmented["class_labels"]
                )
                total_generated += 1

            except Exception as e:
                total_skipped += 1
                if total_skipped <= 5:
                    print(f"  ⚠️  Augmentation failed for {img_path.name} "
                          f"(attempt {j}): {e}")

        # Progress
        progress = (i + 1) / len(images) * 100
        print(f"  [{progress:5.1f}%] Processed {img_path.name} "
              f"→ {args.multiplier} augmentations "
              f"({total_generated} total)")

    print(f"\n✅ Augmentation complete!")
    print(f"   Generated: {total_generated} images")
    print(f"   Skipped:   {total_skipped} (failed augmentations)")

    # Auto-split
    if args.split:
        print(f"\n📂 Splitting dataset into train/val/test...")
        split_dataset(aug_img_dir, aug_lbl_dir, args.split_output)
        print(f"✅ Dataset split into: {args.split_output}/images/{{train,val,test}}/")


if __name__ == "__main__":
    main()
