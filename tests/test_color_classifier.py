"""
Tests for the Color Classifier module.

Run: python -m pytest tests/test_color_classifier.py -v
"""

import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from color_classifier import ColorClassifier


@pytest.fixture
def classifier():
    """Create a fresh color classifier instance."""
    return ColorClassifier()


def make_solid_image(bgr_color, size=(100, 100)):
    """Create a solid-colored test image."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = bgr_color
    return img


def make_noisy_image(bgr_color, size=(100, 100), noise_level=15):
    """Create a color image with Gaussian noise."""
    img = make_solid_image(bgr_color, size)
    noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class TestColorClassifier:
    """Tests for color classification accuracy."""

    def test_yellow_detection(self, classifier):
        """LEGO yellow (bright saturated yellow)."""
        # BGR for LEGO yellow
        img = make_solid_image((0, 220, 255))
        color, conf = classifier.classify(img)
        assert color == "yellow", f"Expected yellow, got {color}"
        assert conf > 0.5

    def test_green_detection(self, classifier):
        """LEGO green (bright saturated green)."""
        img = make_solid_image((0, 180, 0))
        color, conf = classifier.classify(img)
        assert color == "green", f"Expected green, got {color}"
        assert conf > 0.5

    def test_blue_detection(self, classifier):
        """LEGO blue (bright saturated blue)."""
        img = make_solid_image((230, 120, 0))
        color, conf = classifier.classify(img)
        assert color == "blue", f"Expected blue, got {color}"
        assert conf > 0.5

    def test_white_detection(self, classifier):
        """LEGO white (bright, low saturation)."""
        img = make_solid_image((240, 240, 240))
        color, conf = classifier.classify(img)
        assert color == "white", f"Expected white, got {color}"
        assert conf > 0.5

    def test_red_detection(self, classifier):
        """Red LEGO pieces (barriers, trowels)."""
        img = make_solid_image((0, 0, 220))
        color, conf = classifier.classify(img)
        assert color == "red", f"Expected red, got {color}"
        assert conf > 0.3

    def test_noisy_yellow(self, classifier):
        """Yellow detection with noise (simulated real conditions)."""
        img = make_noisy_image((0, 220, 255), noise_level=20)
        color, conf = classifier.classify(img)
        assert color == "yellow", f"Expected yellow, got {color}"

    def test_noisy_green(self, classifier):
        """Green detection with noise."""
        img = make_noisy_image((0, 180, 0), noise_level=20)
        color, conf = classifier.classify(img)
        assert color == "green", f"Expected green, got {color}"

    def test_noisy_blue(self, classifier):
        """Blue detection with noise."""
        img = make_noisy_image((230, 120, 0), noise_level=20)
        color, conf = classifier.classify(img)
        assert color == "blue", f"Expected blue, got {color}"

    def test_bbox_crop(self, classifier):
        """Test color classification with bounding box crop."""
        # Create image with yellow region in center
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = (0, 220, 255)  # Yellow center
        color, conf = classifier.classify(img, bbox=(50, 50, 150, 150))
        assert color == "yellow", f"Expected yellow, got {color}"

    def test_empty_bbox(self, classifier):
        """Test with zero-size bounding box."""
        img = make_solid_image((0, 220, 255))
        color, conf = classifier.classify(img, bbox=(50, 50, 50, 50))
        assert color == "unknown"

    def test_classify_multiple(self, classifier):
        """Test batch classification of multiple bboxes."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[0:100, 0:100] = (0, 220, 255)     # Yellow (top-left)
        img[0:100, 200:300] = (0, 180, 0)      # Green (top-right)
        img[200:300, 0:100] = (230, 120, 0)    # Blue (bottom-left)

        bboxes = [(0, 0, 100, 100), (200, 0, 300, 100), (0, 200, 100, 300)]
        results = classifier.classify_multiple(img, bboxes)

        assert len(results) == 3
        assert results[0][0] == "yellow"
        assert results[1][0] == "green"
        assert results[2][0] == "blue"

    def test_display_color(self, classifier):
        """Test display color retrieval."""
        color = classifier.get_display_color("yellow")
        assert isinstance(color, tuple)
        assert len(color) == 3

        unknown_color = classifier.get_display_color("nonexistent")
        assert unknown_color == (128, 128, 128)

    def test_calibration(self, classifier):
        """Test color range calibration from sample."""
        img = make_solid_image((0, 220, 255))  # Yellow
        classifier.calibrate_from_sample(img, "yellow", (10, 10, 90, 90))
        # After calibration, should still detect yellow
        color, conf = classifier.classify(img)
        assert color == "yellow", f"Expected yellow after calibration, got {color}"


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_very_dark_image(self, classifier):
        """Very dark image should return black or unknown."""
        img = make_solid_image((5, 5, 5))
        color, conf = classifier.classify(img)
        assert color in ("black", "unknown"), f"Unexpected: {color}"

    def test_medium_gray(self, classifier):
        """Medium gray should be difficult to classify."""
        img = make_solid_image((128, 128, 128))
        color, conf = classifier.classify(img)
        # Gray is ambiguous — any result is acceptable
        assert isinstance(color, str)

    def test_single_pixel(self, classifier):
        """Single pixel image."""
        img = np.array([[[0, 220, 255]]], dtype=np.uint8)
        color, conf = classifier.classify(img)
        assert isinstance(color, str)

    def test_large_bbox_clamping(self, classifier):
        """Bounding box larger than image should be clamped."""
        img = make_solid_image((0, 220, 255), size=(100, 100))
        color, conf = classifier.classify(img, bbox=(-50, -50, 200, 200))
        assert color == "yellow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
