"""
Camera Abstraction — Handles USB cameras, video files, and image directories.

Provides a unified interface for capturing frames regardless of source.
"""

import cv2
import os
import glob
import time
from typing import Optional, Tuple


class Camera:
    """
    Unified camera/video/image source abstraction.

    Usage:
        cam = Camera(source=0)              # USB webcam
        cam = Camera(source="video.mp4")    # Video file
        cam = Camera(source="images/")      # Image directory
        cam = Camera(source=0, width=1280, height=720)  # With resolution

        while True:
            ret, frame = cam.read()
            if not ret:
                break
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
    """

    def __init__(
        self,
        source=0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ):
        """
        Args:
            source: Camera index (int), video file path (str),
                    or image directory path (str ending with /)
            width:  Desired frame width (for cameras only)
            height: Desired frame height (for cameras only)
            fps:    Desired FPS (for cameras only)
        """
        self.source = source
        self.is_camera = isinstance(source, int)
        self.is_directory = isinstance(source, str) and os.path.isdir(source)
        self.cap = None
        self.image_files = []
        self.image_index = 0
        self.frame_count = 0
        self._fps_timer = time.time()
        self._fps = 0.0

        if self.is_directory:
            # Load images from directory
            exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
            for ext in exts:
                self.image_files.extend(
                    sorted(glob.glob(os.path.join(source, ext)))
                )
                self.image_files.extend(
                    sorted(glob.glob(os.path.join(source, ext.upper())))
                )
            self.image_files = sorted(set(self.image_files))
            print(f"[Camera] Loaded {len(self.image_files)} images from {source}")
        else:
            # Open video capture (try multiple backends on Windows)
            if self.is_camera and os.name == 'nt':
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
                success = False
                for b in backends:
                    print(f"[Camera] Trying backend: {b} ...")
                    self.cap = cv2.VideoCapture(source, b)
                    if self.cap and self.cap.isOpened():
                        # Test read to verify the backend is actually working
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ Camera started with backend {b}")
                            success = True
                            break
                        self.cap.release()
                
                if not success:
                    print("[Camera] Backend specific attempts failed. Trying default...")
                    self.cap = cv2.VideoCapture(source)
            else:
                self.cap = cv2.VideoCapture(source)

            if not self.cap or not self.cap.isOpened():
                # One last try with a different index
                if source == 0:
                    self.cap = cv2.VideoCapture(1)
                    if self.cap and self.cap.isOpened():
                         self.source = 1
                
                if not self.cap or not self.cap.isOpened():
                    raise RuntimeError(
                        f"Cannot open camera {source}. Check if it's plugged in or used by another app."
                    )

            # Set resolution for cameras
            if self.is_camera:
                if width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[Camera] {actual_w}x{actual_h} ready.")

    def read(self) -> Tuple[bool, Optional['cv2.Mat']]:
        """Read next frame. Returns (success, frame)."""
        self.frame_count += 1

        # Update FPS every 30 frames
        if self.frame_count % 30 == 0:
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed > 0:
                self._fps = 30.0 / elapsed
            self._fps_timer = now

        if self.is_directory:
            if self.image_index >= len(self.image_files):
                return False, None
            path = self.image_files[self.image_index]
            frame = cv2.imread(path)
            self.image_index += 1
            return frame is not None, frame
        else:
            return self.cap.read()

    @property
    def fps(self) -> float:
        """Current FPS (calculated from actual frame rate)."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Total frames for video/image sources, -1 for live camera."""
        if self.is_directory:
            return len(self.image_files)
        elif self.is_camera:
            return -1
        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        self.release()
