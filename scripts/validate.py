"""
Validation Script — Evaluate trained model on test set.

Usage:
    python scripts/validate.py
    python scripts/validate.py --weights runs/detect/lego_detector/weights/best.pt
    python scripts/validate.py --data config.yaml --split test
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Validate LEGO detection model")
    parser.add_argument(
        "--weights", type=str,
        default="runs/detect/lego_detector/weights/best.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--data", type=str, default="config.yaml",
        help="Dataset config YAML"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["val", "test"],
        help="Which split to evaluate on. Default: test"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for evaluation"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device for evaluation"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-class metrics"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        print(f"❌ Model weights not found: {args.weights}")
        print("Train the model first: python train.py")
        sys.exit(1)

    from ultralytics import YOLO

    print("=" * 60)
    print("  LEGO Detection — Model Validation")
    print("=" * 60)

    model = YOLO(args.weights)

    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device if args.device else None,
        plots=True,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  📊 Validation Results")
    print("=" * 60)

    if hasattr(results, "box"):
        metrics = results.box
        print(f"  mAP@50:      {metrics.map50:.4f}")
        print(f"  mAP@50-95:   {metrics.map:.4f}")

        if hasattr(metrics, "maps") and metrics.maps is not None:
            print(f"\n  Per-class AP@50:")
            names = model.names
            for i, ap in enumerate(metrics.maps):
                class_name = names.get(i, f"class_{i}")
                print(f"    {class_name:25s} {ap:.4f}")

    print("=" * 60)
    print(f"  Results & plots saved to: runs/detect/")
    print("=" * 60)


if __name__ == "__main__":
    main()
