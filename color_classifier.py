"""
Color Classifier — HSV-based color detection for LEGO objects.

After YOLOv8 detects an object (block/rod), this module classifies
its dominant color using HSV color space analysis. This is more robust
than training separate YOLO classes per color.

Supported colors: yellow, green, blue, white
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# ============================================================
# HSV Color Ranges (tuned for LEGO brick colors)
# ============================================================
# Format: (lower_bound, upper_bound) in HSV
# H: 0-179, S: 0-255, V: 0-255 in OpenCV

COLOR_RANGES = {
    "yellow": [
        (np.array([18, 80, 80]), np.array([35, 255, 255])),
    ],
    "green": [
        (np.array([36, 50, 50]), np.array([85, 255, 255])),
    ],
    "blue": [
        (np.array([90, 50, 50]), np.array([130, 255, 255])),
    ],
    "white": [
        (np.array([0, 0, 180]), np.array([179, 50, 255])),
    ],
    "red": [
        # Red wraps around 0 in HSV, so two ranges needed
        (np.array([0, 70, 50]), np.array([10, 255, 255])),
        (np.array([170, 70, 50]), np.array([179, 255, 255])),
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([179, 100, 60])),
    ],
}

# Display colors for visualization (BGR format for OpenCV)
DISPLAY_COLORS_BGR = {
    "yellow": (0, 230, 255),
    "green":  (0, 180, 0),
    "blue":   (255, 120, 0),
    "white":  (230, 230, 230),
    "red":    (0, 0, 230),
    "black":  (40, 40, 40),
    "unknown": (128, 128, 128),
}


class ColorClassifier:
    """
    Classifies the dominant color of a LEGO object ROI using HSV analysis.
    
    This works by:
    1. Converting the ROI to HSV color space
    2. Creating masks for each known color range
    3. Computing the percentage of pixels matching each color
    4. Returning the color with highest pixel coverage (above threshold)
    """

    def __init__(self, min_pixel_ratio: float = 0.15, blur_kernel: int = 5):
        """
        Args:
            min_pixel_ratio: Minimum fraction of pixels that must match a
                             color to be considered valid (0.0-1.0).
                             Lower = more lenient, higher = stricter.
            blur_kernel: Gaussian blur kernel size for noise reduction.
        """
        self.min_pixel_ratio = min_pixel_ratio
        self.blur_kernel = blur_kernel
        self.color_ranges = COLOR_RANGES.copy()

    def classify(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float]:
        """
        Classify the dominant color of an image region.

        Args:
            image: BGR image (full frame or cropped ROI)
            bbox:  Optional (x1, y1, x2, y2) bounding box to crop from image.
                   If None, uses the entire image.

        Returns:
            Tuple of (color_name, confidence) where confidence is 0.0-1.0
        """
        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Clamp coordinates to image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            roi = image[y1:y2, x1:x2]
        else:
            roi = image

        if roi.size == 0:
            return ("unknown", 0.0)

        # Shrink center crop to avoid bounding box edge noise
        rh, rw = roi.shape[:2]
        margin_x, margin_y = int(rw * 0.1), int(rh * 0.1)
        center_roi = roi[margin_y:rh - margin_y, margin_x:rw - margin_x]
        if center_roi.size == 0:
            center_roi = roi

        # Pre-process: blur to reduce noise
        blurred = cv2.GaussianBlur(center_roi, (self.blur_kernel, self.blur_kernel), 0)

        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        total_pixels = hsv.shape[0] * hsv.shape[1]
        if total_pixels == 0:
            return ("unknown", 0.0)

        # Score each color
        best_color = "unknown"
        best_ratio = 0.0

        for color_name, ranges in self.color_ranges.items():
            # Create combined mask for all ranges of this color
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)

            # Clean up mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Calculate coverage ratio
            pixel_count = cv2.countNonZero(mask)
            ratio = pixel_count / total_pixels

            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color_name

        # Only return a color if it exceeds the minimum threshold
        if best_ratio < self.min_pixel_ratio:
            return ("unknown", best_ratio)

        return (best_color, best_ratio)

    def classify_multiple(self, image: np.ndarray,
                          bboxes: list) -> list:
        """
        Classify colors for multiple bounding boxes in one image.

        Args:
            image:  BGR image
            bboxes: List of (x1, y1, x2, y2) bounding boxes

        Returns:
            List of (color_name, confidence) tuples
        """
        results = []
        for bbox in bboxes:
            color, conf = self.classify(image, bbox)
            results.append((color, conf))
        return results

    def get_display_color(self, color_name: str) -> Tuple[int, int, int]:
        """Get BGR color for drawing the label on screen."""
        return DISPLAY_COLORS_BGR.get(color_name, DISPLAY_COLORS_BGR["unknown"])

    def calibrate_from_sample(self, image: np.ndarray, color_name: str,
                               bbox: Tuple[int, int, int, int]):
        """
        Calibrate color ranges from a known sample. Useful for adjusting
        to specific lighting conditions.

        Args:
            image:      BGR image containing the sample
            color_name: Known color of the sample
            bbox:       (x1, y1, x2, y2) of the sample in the image
        """
        x1, y1, x2, y2 = bbox
        roi = image[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Compute mean and std of HSV channels
        h_mean, s_mean, v_mean = cv2.mean(hsv)[:3]
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])

        # Create range with 2-sigma spread
        lower = np.array([
            max(0, h_mean - 2 * h_std),
            max(0, s_mean - 2 * s_std),
            max(0, v_mean - 2 * v_std)
        ], dtype=np.uint8)
        upper = np.array([
            min(179, h_mean + 2 * h_std),
            min(255, s_mean + 2 * s_std),
            min(255, v_mean + 2 * v_std)
        ], dtype=np.uint8)

        self.color_ranges[color_name] = [(lower, upper)]
        print(f"[ColorClassifier] Calibrated '{color_name}': "
              f"H=[{lower[0]}-{upper[0]}], S=[{lower[1]}-{upper[1]}], V=[{lower[2]}-{upper[2]}]")


# Quick test
if __name__ == "__main__":
    import sys

    classifier = ColorClassifier()

    if len(sys.argv) < 2:
        print("Usage: python color_classifier.py <image_path>")
        print("Classifies the dominant color of the entire image.")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: Cannot load image '{sys.argv[1]}'")
        sys.exit(1)

    color, confidence = classifier.classify(img)
    print(f"Dominant color: {color} (confidence: {confidence:.2%})")
