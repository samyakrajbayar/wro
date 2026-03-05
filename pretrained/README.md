# Pretrained LEGO Detection

This directory contains alternative detection solutions that use **pretrained models** instead of our custom-trained YOLOv8 model.

## 1. Online: Gemini 2.0 Flash (`detect_online.py`)
Uses Google's Gemini Vision API for zero-shot detection.
- **Pros**: Extremely accurate, identifies complex objects (barrier, trowel) without training.
- **Cons**: Requires internet, slower latency.
- **Usage**:
  ```bash
  export GOOGLE_API_KEY="your_api_key_here"
  python pretrained/detect_online.py --source photo.jpg
  ```

## 2. Local: YOLOv8m (Generic) (`detect_yolo_generic.py`)
Uses the standard YOLOv8 medium model trained on the COCO dataset.
- **Pros**: Runs offline, very fast.
- **Cons**: Not specifically trained for WRO LEGO pieces; will label many things as "sports ball" (for LEGO balls) or "bowl". 
- **Usage**:
  ```bash
  python pretrained/detect_yolo_generic.py --source photo.jpg
  ```

## Why use custom training instead?
While pretrained models are good, our custom training (in the main folder) is specifically tuned for the **exact geometry and colors** of the WRO 2026 LEGO pieces, ensuring maximum accuracy for the car-mounted camera.
