"""
Simple Bounding Box Labeling Tool — YOLO Format

An OpenCV-based annotation tool that lets you draw bounding boxes
on images and save them in YOLO format (.txt files).

Usage:
    python scripts/label_tool.py --images data/raw/ --output data/

Controls:
    Click+Drag  — Draw bounding box
    0-5         — Select class (shown in legend)
    z           — Undo last box
    n / →       — Next image
    p / ←       — Previous image
    s           — Save current labels
    d           — Delete all boxes on current image
    q / ESC     — Quit (auto-saves)
"""

import argparse
import cv2
import os
import sys
import glob
from pathlib import Path


# Class definitions (must match config.yaml)
CLASSES = {
    0: ("lego_block",         (0, 200, 255)),   # Gold
    1: ("lego_rod",           (255, 150, 0)),    # Cyan
    2: ("barrier",            (0, 0, 255)),      # Red
    3: ("rectangular_trowel", (255, 0, 200)),    # Magenta
    4: ("cement_bowl",        (200, 200, 0)),    # Teal
    5: ("masonry_trowel",     (0, 255, 100)),    # Green
}


class LabelTool:
    def __init__(self, images_dir: str, output_dir: str):
        self.images_dir = images_dir
        self.output_dir = output_dir

        # Find images
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        self.image_files = []
        for ext in exts:
            self.image_files.extend(
                sorted(glob.glob(os.path.join(images_dir, ext)))
            )
            self.image_files.extend(
                sorted(glob.glob(os.path.join(images_dir, ext.upper())))
            )
        self.image_files = sorted(set(self.image_files))

        if not self.image_files:
            print(f"❌ No images found in {images_dir}")
            sys.exit(1)

        print(f"📁 Found {len(self.image_files)} images in {images_dir}")

        # Create output directories
        self.img_out = os.path.join(output_dir, "images", "train")
        self.lbl_out = os.path.join(output_dir, "labels", "train")
        os.makedirs(self.img_out, exist_ok=True)
        os.makedirs(self.lbl_out, exist_ok=True)

        # State
        self.current_index = 0
        self.current_class = 0
        self.boxes = []           # List of (class_id, x1, y1, x2, y2) in pixels
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.image = None
        self.display = None
        self.window_name = "LEGO Label Tool"

    def load_image(self):
        """Load current image and any existing labels."""
        path = self.image_files[self.current_index]
        self.image = cv2.imread(path)
        self.boxes = []

        # Check for existing labels
        stem = Path(path).stem
        label_path = os.path.join(self.lbl_out, f"{stem}.txt")
        if os.path.isfile(label_path):
            h, w = self.image.shape[:2]
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = (float(parts[1]), float(parts[2]),
                                           float(parts[3]), float(parts[4]))
                        # Convert YOLO to pixel coords
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        self.boxes.append((cls_id, x1, y1, x2, y2))

    def save_labels(self):
        """Save current boxes to YOLO format label file."""
        path = self.image_files[self.current_index]
        stem = Path(path).stem
        h, w = self.image.shape[:2]

        # Save label file
        label_path = os.path.join(self.lbl_out, f"{stem}.txt")
        with open(label_path, "w") as f:
            for cls_id, x1, y1, x2, y2 in self.boxes:
                # Convert pixel to YOLO format
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Copy image to output
        import shutil
        img_dest = os.path.join(self.img_out, Path(path).name)
        if not os.path.isfile(img_dest):
            shutil.copy2(path, img_dest)

        print(f"  💾 Saved {len(self.boxes)} boxes → {label_path}")

    def draw_display(self):
        """Draw current state on display image."""
        self.display = self.image.copy()
        h, w = self.display.shape[:2]

        # Draw existing boxes
        for cls_id, x1, y1, x2, y2 in self.boxes:
            name, color = CLASSES.get(cls_id, (f"cls_{cls_id}", (128, 128, 128)))
            cv2.rectangle(self.display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.display, name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw current box being drawn
        if self.drawing and self.start_point and self.current_point:
            _, color = CLASSES.get(self.current_class, ("", (128, 128, 128)))
            cv2.rectangle(self.display, self.start_point, self.current_point,
                          color, 2)

        # Draw legend (top-left)
        y_offset = 25
        cv2.rectangle(self.display, (5, 5), (230, 25 + len(CLASSES) * 22), (0, 0, 0), -1)
        for cls_id, (name, color) in CLASSES.items():
            marker = "▶ " if cls_id == self.current_class else "  "
            text = f"{marker}[{cls_id}] {name}"
            cv2.putText(self.display, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y_offset += 22

        # Draw status bar
        status = (f"Image {self.current_index + 1}/{len(self.image_files)} | "
                  f"Boxes: {len(self.boxes)} | "
                  f"Class: [{self.current_class}] {CLASSES[self.current_class][0]}")
        cv2.rectangle(self.display, (0, h - 30), (w, h), (40, 40, 40), -1)
        cv2.putText(self.display, status, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                sx, sy = self.start_point
                ex, ey = x, y

                # Only add if box is reasonably sized
                if abs(ex - sx) > 5 and abs(ey - sy) > 5:
                    x1, y1 = min(sx, ex), min(sy, ey)
                    x2, y2 = max(sx, ex), max(sy, ey)
                    self.boxes.append((self.current_class, x1, y1, x2, y2))
                    print(f"  + Added {CLASSES[self.current_class][0]} "
                          f"box at ({x1},{y1})-({x2},{y2})")

    def run(self):
        """Main labeling loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.load_image()

        print("\n" + "=" * 50)
        print("  LEGO Label Tool")
        print("=" * 50)
        print("  Click+Drag: Draw box    0-5: Select class")
        print("  z: Undo    n/→: Next    p/←: Previous")
        print("  s: Save    d: Delete all    q/ESC: Quit")
        print("=" * 50 + "\n")

        while True:
            self.draw_display()
            cv2.imshow(self.window_name, self.display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                self.save_labels()
                break

            elif key in [ord(str(i)) for i in range(6)]:
                self.current_class = key - ord("0")
                print(f"  🎨 Class: [{self.current_class}] "
                      f"{CLASSES[self.current_class][0]}")

            elif key == ord("z"):
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"  ↩ Undid {CLASSES[removed[0]][0]} box")

            elif key == ord("n") or key == 83:  # n or →
                self.save_labels()
                self.current_index = min(self.current_index + 1,
                                         len(self.image_files) - 1)
                self.load_image()

            elif key == ord("p") or key == 81:  # p or ←
                self.save_labels()
                self.current_index = max(0, self.current_index - 1)
                self.load_image()

            elif key == ord("s"):
                self.save_labels()

            elif key == ord("d"):
                self.boxes = []
                print("  🗑  Deleted all boxes")

        cv2.destroyAllWindows()
        print("\n✅ Labeling complete.")
        print(f"   Images: {self.img_out}")
        print(f"   Labels: {self.lbl_out}")


def main():
    parser = argparse.ArgumentParser(description="LEGO Bounding Box Label Tool")
    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing images to label")
    parser.add_argument("--output", type=str, default="data",
                        help="Output root directory. Default: data")
    args = parser.parse_args()

    tool = LabelTool(args.images, args.output)
    tool.run()


if __name__ == "__main__":
    main()
