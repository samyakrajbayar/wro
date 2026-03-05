"""
Visualization Utilities for LEGO Object Detection.

Provides drawing functions for bounding boxes, labels, FPS counter,
and color-coded overlays.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


# Class-specific colors (BGR) — distinct for each class
CLASS_COLORS = {
    "lego_block":          (0, 200, 255),    # Gold
    "lego_rod":            (255, 150, 0),     # Cyan-ish
    "barrier":             (0, 0, 255),       # Red
    "rectangular_trowel":  (255, 0, 200),     # Magenta
    "cement_bowl":         (200, 200, 0),     # Teal
    "masonry_trowel":      (0, 255, 100),     # Spring green
}

# Color label colors (BGR)
LEGO_COLORS = {
    "yellow": (0, 230, 255),
    "green":  (0, 200, 0),
    "blue":   (255, 130, 0),
    "white":  (240, 240, 240),
    "red":    (0, 0, 240),
    "black":  (50, 50, 50),
    "unknown": (160, 160, 160),
}


def draw_detection(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    color_label: Optional[str] = None,
    color_conf: Optional[float] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw a single detection on the frame with label and optional color tag.

    Args:
        frame:       BGR image to draw on (modified in-place)
        bbox:        (x1, y1, x2, y2) bounding box coordinates
        class_name:  Detection class name
        confidence:  Detection confidence (0-1)
        color_label: Optional color classification result
        color_conf:  Optional color classification confidence
        thickness:   Line thickness
        font_scale:  Font scale for labels

    Returns:
        Modified frame
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = CLASS_COLORS.get(class_name, (128, 128, 128))

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Build label text
    label = f"{class_name} {confidence:.0%}"
    if color_label and color_label != "unknown":
        label = f"{color_label} {class_name} {confidence:.0%}"

    # Calculate label background size
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    label_h += baseline

    # Draw label background (filled rectangle)
    label_y1 = max(0, y1 - label_h - 8)
    label_y2 = y1
    cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 8, label_y2), color, -1)

    # Draw label text
    cv2.putText(
        frame, label,
        (x1 + 4, label_y2 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (0, 0, 0) if sum(color) > 400 else (255, 255, 255),
        1, cv2.LINE_AA
    )

    # Draw small color indicator dot
    if color_label and color_label != "unknown":
        dot_color = LEGO_COLORS.get(color_label, (128, 128, 128))
        dot_center = (x2 - 10, y1 + 10)
        cv2.circle(frame, dot_center, 6, dot_color, -1)
        cv2.circle(frame, dot_center, 6, (0, 0, 0), 1)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on top-right corner."""
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = frame.shape[1] - tw - 15
    y = 30

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # FPS text
    color = (0, 255, 0) if fps > 20 else (0, 200, 255) if fps > 10 else (0, 0, 255)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def draw_info_panel(
    frame: np.ndarray,
    detections: List[dict],
    model_name: str = "YOLOv8s"
) -> np.ndarray:
    """
    Draw a semi-transparent info panel at the bottom with detection summary.
    """
    h, w = frame.shape[:2]
    panel_h = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Count detections by class
    counts = {}
    for det in detections:
        cls = det.get("class", "unknown")
        counts[cls] = counts.get(cls, 0) + 1

    summary = f"Model: {model_name} | Detections: {len(detections)}"
    if counts:
        parts = [f"{name}: {count}" for name, count in counts.items()]
        summary += " | " + ", ".join(parts)

    cv2.putText(
        frame, summary, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
    )
    return frame


def draw_crosshair(frame: np.ndarray) -> np.ndarray:
    """Draw center crosshair (useful for aiming the car camera)."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size = 20
    color = (0, 255, 0)
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)
    return frame


def create_detection_grid(
    images: List[np.ndarray],
    cols: int = 3,
    cell_size: Tuple[int, int] = (320, 320)
) -> np.ndarray:
    """Create a grid of detection result images for visualization."""
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    resized = [cv2.resize(img, cell_size) for img in images]

    # Pad to fill grid
    rows = (len(resized) + cols - 1) // cols
    while len(resized) < rows * cols:
        resized.append(np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8))

    # Assemble grid
    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.hstack(resized[r * cols:(r + 1) * cols]))
    return np.vstack(row_imgs)
