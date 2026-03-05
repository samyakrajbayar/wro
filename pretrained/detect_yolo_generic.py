"""
Generic YOLOv8 Detection — Local Pretrained Model

Uses the official YOLOv8m (Medium) model trained on COCO. 
This is a 'pretrained' baseline that doesn't require our LEGO data.

Usage:
    python pretrained/detect_yolo_generic.py --source photo.jpg
"""

import argparse
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    import sys
    import os
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.camera import Camera

    # Load official pretrained model
    model = YOLO("yolov8m.pt")
    
    # Initialize camera using our stable abstraction
    try:
        source = int(args.source) if args.source.isdigit() else args.source
        cam = Camera(source=source)
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return

    print("\n" + "=" * 50)
    print("  Generic YOLO Detection (Baseline)")
    print("  Press 'q' to quit")
    print("=" * 50 + "\n")

    window_name = "Generic YOLO Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("⚠️  Frame capture failed. Retrying...")
            time.sleep(0.1)
            continue
            
        # Run inference on single frame
        results = model.predict(frame, conf=args.conf, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        cv2.imshow(window_name, annotated_frame)
        
        # Force window to front on first frame
        if cam.frame_count == 1:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0) # Release topmost after focusing
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
