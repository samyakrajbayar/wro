"""
Synthetic Data Generator — LEGO Object Detection

Generates programmatically created training images of LEGO objects
with automatic YOLO-format annotations. This is used to train the 
model when real photos aren't yet available.

Objects:
- lego_block (4 colors)
- lego_rod (4 colors)
- barrier (Red/Black + balls)
- rectangular_trowel (Red)
- cement_bowl (White/Grey)
- masonry_trowel (White/Red)
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path

# Class IDs (match config.yaml)
CLASSES = {
    0: "lego_block",
    1: "lego_rod",
    2: "barrier",
    3: "rectangular_trowel",
    4: "cement_bowl",
    5: "masonry_trowel"
}

# Colors for shapes (BGR)
COLORS = {
    "yellow": (0, 220, 255),
    "blue": (255, 120, 0),
    "green": (0, 180, 0),
    "white": (240, 240, 240),
    "grey": (150, 150, 150),
    "red": (0, 0, 220),
    "black": (30, 30, 30)
}

def draw_lego_block(img, center, size, color):
    """Draw a 2D representation of a LEGO block with studs."""
    x, y = center
    w, h = size
    # Main body
    cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
    cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (0,0,0), 1)
    
    # Studs
    stud_r = w // 6
    for i in [-1, 1]:
        for j in [-1, 1]:
            cv2.circle(img, (x + i*w//4, y + j*h//4), stud_r, color, -1)
            cv2.circle(img, (x + i*w//4, y + j*h//4), stud_r, (0,0,0), 1)

def draw_lego_rod(img, center, size, color):
    """Draw a long LEGO rod (cement element)."""
    x, y = center
    w, h = size # h > w usually
    # Body
    cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
    cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (0,0,0), 1)
    # Studs along length
    stud_r = w // 3
    num_studs = 4
    for i in range(num_studs):
        sy = y - h//2 + (i+1)*h//(num_studs+1)
        cv2.circle(img, (x, sy), stud_r, color, -1)
        cv2.circle(img, (x, sy), stud_r, (0,0,0), 1)

def draw_barrier(img, center, size, type="red"):
    """Draw a barrier structure."""
    x, y = center
    w, h = size
    color = COLORS["red"] if type == "red" else COLORS["black"]
    ball_color = COLORS["blue"] if type == "red" else COLORS["red"]
    
    # Vertical post
    cv2.rectangle(img, (x-w//4, y-h//2), (x+w//4, y+h//2), color, -1)
    # Base
    cv2.rectangle(img, (x-w//2, y+h//4), (x+w//2, y+h//2), color, -1)
    # Top ball
    cv2.circle(img, (x, y-h//2), w//3, ball_color, -1)
    cv2.circle(img, (x, y-h//2), w//3, (0,0,0), 1)

def draw_trowel(img, center, size, masonry=False):
    """Draw a trowel."""
    x, y = center
    w, h = size
    if masonry:
        # Masonry trowel (White base, Red handle)
        cv2.rectangle(img, (x-w//2, y), (x+w//2, y+h//2), COLORS["white"], -1)
        cv2.rectangle(img, (x-w//4, y-h//2), (x+w//4, y), COLORS["red"], -1)
    else:
        # Rectangular trowel (Red)
        cv2.rectangle(img, (x-w//2, y-h//4), (x+w//2, y+h//4), COLORS["red"], -1)
        # Handle
        cv2.rectangle(img, (x-w//8, y-h//2), (x+w//8, y-h//4), COLORS["red"], -1)

def draw_bowl(img, center, size):
    """Draw a cement bowl."""
    x, y = center
    w, h = size
    # Outer white/grey circle
    cv2.circle(img, (x, y), w//2, COLORS["grey"], -1)
    # Inner circle (cement)
    cv2.circle(img, (x, y), w//3, (0,0,0), 1)
    cv2.ellipse(img, (x, y), (w//2, h//3), 0, 0, 360, COLORS["white"], -1)

def generate_image(img_size=(640, 640)):
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    # Random background color
    bg_color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
    img[:] = bg_color
    
    labels = []
    num_objects = random.randint(2, 6)
    
    for _ in range(num_objects):
        cls_id = random.randint(0, 5)
        cx = random.randint(100, 540)
        cy = random.randint(100, 540)
        
        # Color for block/rod
        l_color = random.choice(["yellow", "blue", "green", "white"])
        color_bgr = COLORS[l_color]
        
        w, h = 0, 0
        if cls_id == 0: # block
            w, h = random.randint(40, 80), random.randint(40, 80)
            draw_lego_block(img, (cx, cy), (w, h), color_bgr)
        elif cls_id == 1: # rod
            w, h = random.randint(30, 60), random.randint(100, 160)
            angle = random.randint(0, 360)
            draw_lego_rod(img, (cx, cy), (w, h), color_bgr) # simplify: no rotation in draw for now
        elif cls_id == 2: # barrier
            w, h = random.randint(60, 100), random.randint(80, 120)
            draw_barrier(img, (cx, cy), (w, h), type=random.choice(["red", "black"]))
        elif cls_id == 3: # rect trowel
            w, h = random.randint(80, 120), random.randint(60, 100)
            draw_trowel(img, (cx, cy), (w, h), masonry=False)
        elif cls_id == 4: # bowl
            w, h = random.randint(80, 120), random.randint(80, 120)
            draw_bowl(img, (cx, cy), (w, h))
        elif cls_id == 5: # masonry trowel
            w, h = random.randint(60, 100), random.randint(80, 120)
            draw_trowel(img, (cx, cy), (w, h), masonry=True)
            
        # Normalize labels
        labels.append(f"{cls_id} {cx/640:.6f} {cy/640:.6f} {w/640:.6f} {h/640:.6f}")
        
    return img, labels

def main(count=500):
    output_dir = Path("data")
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    val_images_dir = output_dir / "images" / "val"
    val_labels_dir = output_dir / "labels" / "val"
    
    for d in [images_dir, labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"Generating {count} synthetic training images...")
    for i in range(count):
        img, labels = generate_image()
        is_val = i < (count // 5)
        target_img_dir = val_images_dir if is_val else images_dir
        target_lbl_dir = val_labels_dir if is_val else labels_dir
        
        cv2.imwrite(str(target_img_dir / f"syn_{i:04d}.jpg"), img)
        with open(target_lbl_dir / f"syn_{i:04d}.txt", "w") as f:
            f.write("\n".join(labels))
            
    print("Generation complete!")

if __name__ == "__main__":
    main(1000)
