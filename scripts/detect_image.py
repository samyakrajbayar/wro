"""
Single Image Detection — For testing and debugging.

Usage:
    python scripts/detect_image.py --source photo.jpg
    python scripts/detect_image.py --source test_images/ --weights best.pt
    python scripts/detect_image.py --source photo.jpg --save --output results/
"""

import argparse
import cv2
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Detect LEGO objects in images")
    parser.add_argument(
        "--source", type=str, required=True,
        help="Image file or directory of images"
    )
    parser.add_argument(
        "--weights", type=str,
        default="runs/detect/lego_model_v1/weights/best.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated images"
    )
    parser.add_argument(
        "--output", type=str, default="detection_output",
        help="Output directory for saved images"
    )
    parser.add_argument(
        "--show", action="store_true", default=False,
        help="Display results in window"
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable color classification"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device for inference"
    )
    args = parser.parse_args()

    from color_classifier import ColorClassifier
    from utils.visualization import draw_detection, draw_info_panel
    from ultralytics import YOLO

    # Load model
    if not os.path.isfile(args.weights):
        print(f"❌ Model weights not found: {args.weights}")
        sys.exit(1)

    model = YOLO(args.weights)
    color_classifier = ColorClassifier() if not args.no_color else None

    # Find images
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.isfile(args.source):
        image_files = [args.source]
    elif os.path.isdir(args.source):
        image_files = sorted([
            str(f) for f in Path(args.source).iterdir()
            if f.suffix.lower() in img_exts
        ])
    else:
        print(f"❌ Source not found: {args.source}")
        sys.exit(1)

    if args.save:
        os.makedirs(args.output, exist_ok=True)

    print(f"\n📸 Processing {len(image_files)} image(s)...")

    color_classes = {"lego_block", "lego_rod"}

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️  Cannot read: {img_path}")
            continue

        # Run detection
        results = model.predict(
            source=img,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            verbose=False,
        )

        display = img.copy()
        detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")

                    color_label = None
                    if color_classifier and cls_name in color_classes:
                        color_label, _ = color_classifier.classify(
                            img, bbox=(x1, y1, x2, y2)
                        )

                    draw_detection(
                        display, (x1, y1, x2, y2),
                        cls_name, conf, color_label
                    )
                    detections.append({"class": cls_name})

        draw_info_panel(display, detections)

        filename = Path(img_path).name
        n_det = len(detections)
        print(f"  ✅ {filename}: {n_det} detection(s)")

        if args.save:
            out_path = os.path.join(args.output, f"det_{filename}")
            cv2.imwrite(out_path, display)
            print(f"     Saved: {out_path}")

        if args.show:
            cv2.imshow(f"Detection: {filename}", display)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    print(f"\n✅ Done. Processed {len(image_files)} images.")


if __name__ == "__main__":
    main()
