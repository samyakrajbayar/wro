"""
Real-Time LEGO Object Detection — Main Inference Script

Runs YOLOv8 detection + HSV color classification from camera/video/images.
Fully offline — no internet required after model is trained.

Usage:
    python detect.py                                        # USB camera (default)
    python detect.py --source 0                             # USB camera explicitly
    python detect.py --source video.mp4                     # Video file
    python detect.py --source images/                       # Image directory
    python detect.py --weights runs/detect/lego_detector/weights/best.pt
    python detect.py --conf 0.5 --show-fps --record output.avi

Controls:
    q     — Quit
    s     — Save screenshot
    r     — Toggle recording
    c     — Toggle crosshair
    +/-   — Adjust confidence threshold
    SPACE — Pause/resume (video only)
"""

import argparse
import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from color_classifier import ColorClassifier
from utils.visualization import draw_detection, draw_fps, draw_info_panel, draw_crosshair
from utils.camera import Camera


def parse_args():
    parser = argparse.ArgumentParser(
        description="LEGO Object Detection — Real-time Inference"
    )
    parser.add_argument(
        "--source", default=0,
        help="Input source: camera index (0,1,..), video path, or image directory. Default: 0"
    )
    parser.add_argument(
        "--weights", type=str, default="runs/detect/lego_model_v1/weights/best.pt",
        help="Path to trained YOLOv8 model weights (.pt file)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold for detections. Default: 0.35"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS. Default: 0.45"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size. Default: 640"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device: '' (auto), 'cpu', '0' (GPU). Default: auto"
    )
    parser.add_argument(
        "--show-fps", action="store_true", default=True,
        help="Show FPS counter"
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable color classification"
    )
    parser.add_argument(
        "--record", type=str, default=None,
        help="Record output to video file (e.g., output.avi)"
    )
    parser.add_argument(
        "--max-det", type=int, default=50,
        help="Maximum detections per frame. Default: 50"
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Camera resolution width"
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Camera resolution height"
    )
    parser.add_argument(
        "--save-dir", type=str, default="detection_output",
        help="Directory to save screenshots. Default: detection_output"
    )
    return parser.parse_args()


class LEGODetector:
    """
    Main detection pipeline combining YOLOv8 + Color Classification.
    """

    # Classes that should get color classification
    COLOR_CLASSES = {"lego_block", "lego_rod"}

    def __init__(self, weights: str, conf: float = 0.35, iou: float = 0.45,
                 imgsz: int = 640, device: str = "", max_det: int = 50,
                 enable_color: bool = True):
        """
        Initialize the detector.

        Args:
            weights:      Path to YOLOv8 .pt weight file
            conf:         Confidence threshold
            iou:          IoU threshold for NMS
            imgsz:        Inference image size
            device:       Compute device
            max_det:      Maximum detections per frame
            enable_color: Whether to run color classification
        """
        from ultralytics import YOLO

        # Verify weights file exists
        if not os.path.isfile(weights):
            print(f"\n❌ ERROR: Model weights not found: {weights}")
            print("\nPlease train the model first:")
            print("  python train.py --data config.yaml --epochs 200")
            print("\nOr specify the correct path:")
            print("  python detect.py --weights path/to/best.pt")
            sys.exit(1)

        print(f"Loading model: {weights}")
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.class_names = self.model.names  # {0: 'lego_block', 1: 'lego_rod', ...}

        # Color classifier
        self.enable_color = enable_color
        self.color_classifier = ColorClassifier() if enable_color else None

        print(f"✅ Model loaded — {len(self.class_names)} classes: "
              f"{list(self.class_names.values())}")
        print(f"   Confidence: {self.conf}, IoU: {self.iou}, "
              f"Image size: {self.imgsz}")

    def detect(self, frame: np.ndarray) -> list:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image

        Returns:
            List of detection dicts with keys:
              - bbox: (x1, y1, x2, y2)
              - class: class name string
              - class_id: class index
              - confidence: float 0-1
              - color: color name (if applicable)
              - color_conf: color classification confidence
        """
        # Run YOLOv8 inference
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device if self.device else None,
            max_det=self.max_det,
            verbose=False,
        )

        detections = []

        # Iterate through the generator (will yield 1 Results object for single frame)
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                    det = {
                        "bbox": (x1, y1, x2, y2),
                        "class": cls_name,
                        "class_id": cls_id,
                        "confidence": conf,
                        "color": None,
                        "color_conf": None,
                    }

                    # Classify color for blocks and rods
                    if (self.enable_color and self.color_classifier
                            and cls_name in self.COLOR_CLASSES):
                        color, color_conf = self.color_classifier.classify(
                            frame, bbox=(x1, y1, x2, y2)
                        )
                        det["color"] = color
                        det["color_conf"] = color_conf

                    detections.append(det)

        return detections

    def draw_results(self, frame: np.ndarray, detections: list,
                     show_fps: bool = True, fps: float = 0,
                     show_crosshair: bool = False) -> np.ndarray:
        """
        Draw all detections on the frame.

        Args:
            frame:          BGR image
            detections:     List of detection dicts from detect()
            show_fps:       Whether to show FPS counter
            fps:            Current FPS value
            show_crosshair: Whether to show center crosshair

        Returns:
            Annotated frame
        """
        display = frame.copy()

        # Draw each detection
        for det in detections:
            draw_detection(
                display,
                bbox=det["bbox"],
                class_name=det["class"],
                confidence=det["confidence"],
                color_label=det.get("color"),
                color_conf=det.get("color_conf"),
            )

        # Draw FPS
        if show_fps:
            draw_fps(display, fps)

        # Draw info panel
        draw_info_panel(display, detections)

        # Draw crosshair
        if show_crosshair:
            draw_crosshair(display)

        return display


def run_detection(args):
    """Main detection loop."""

    # Parse source (convert to int if it's a camera index)
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # Initialize detector
    detector = LEGODetector(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        enable_color=not args.no_color,
    )

    # Initialize camera
    camera = Camera(source=source, width=args.width, height=args.height)

    # Video recording setup
    recorder = None
    recording = False
    if args.record:
        recording = True

    # Screenshot directory
    os.makedirs(args.save_dir, exist_ok=True)
    screenshot_count = 0

    # State
    show_crosshair = False
    paused = False
    conf_threshold = args.conf

    print("\n" + "=" * 50)
    print("  LEGO Detection Running")
    print("=" * 50)
    print("  Controls:")
    print("    q       — Quit")
    print("    s       — Save screenshot")
    print("    r       — Toggle recording")
    print("    c       — Toggle crosshair")
    print("    +/-     — Adjust confidence")
    print("    SPACE   — Pause/resume")
    print("=" * 50 + "\n")

    # FPS tracking
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            if not paused:
                ret, frame = camera.read()
                if not ret or frame is None:
                    if isinstance(source, int):
                        print("⚠️  Frame dropped... retrying")
                        time.sleep(0.01)
                        continue
                    else:
                        print("✅ End of video/images.")
                        break

                # Run detection
                detections = detector.detect(frame)

                # Calculate FPS
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                # Draw results
                display = detector.draw_results(
                    frame, detections,
                    show_fps=args.show_fps,
                    fps=fps,
                    show_crosshair=show_crosshair,
                )

                # Record if enabled
                if recording and recorder is None:
                    h, w = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    recorder = cv2.VideoWriter(
                        args.record or "output.avi", fourcc, 20.0, (w, h)
                    )
                    print(f"🔴 Recording to: {args.record or 'output.avi'}")

                if recorder and recording:
                    recorder.write(display)

                # Show frame
                window_name = "LEGO Detection"
                if self.frame_count == 1:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 1280, 720)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                
                cv2.imshow(window_name, display)
                
                # First frame: handle window focus
                if self.frame_count == 1:
                    cv2.waitKey(100)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                # Screenshot
                screenshot_count += 1
                path = os.path.join(
                    args.save_dir, f"detection_{screenshot_count:04d}.jpg"
                )
                cv2.imwrite(path, display)
                print(f"📸 Screenshot saved: {path}")
            elif key == ord("r"):
                recording = not recording
                if recording:
                    print("🔴 Recording started")
                else:
                    print("⏹  Recording stopped")
                    if recorder:
                        recorder.release()
                        recorder = None
            elif key == ord("c"):
                show_crosshair = not show_crosshair
            elif key == ord("+") or key == ord("="):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                detector.conf = conf_threshold
                print(f"🎯 Confidence threshold: {conf_threshold:.2f}")
            elif key == ord("-"):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                detector.conf = conf_threshold
                print(f"🎯 Confidence threshold: {conf_threshold:.2f}")
            elif key == ord(" "):
                paused = not paused
                print("⏸  Paused" if paused else "▶  Resumed")

    except KeyboardInterrupt:
        print("\n\n⏹  Detection stopped by user.")
    finally:
        if recorder:
            recorder.release()
        camera.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete.")


if __name__ == "__main__":
    args = parse_args()
    run_detection(args)
