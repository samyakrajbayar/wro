# 🏗️ LEGO Detection — Deployment & Setup Guide

This guide explains how to set up and run this LEGO detection system on **any other PC**.

## 📋 Prerequisites
- **Python 3.10 or 3.11** installed. (Download from [python.org](https://www.python.org/downloads/))
- **Internet connection** for the first setup (to download libraries and models).
- **Webcam** (USB or built-in).

---

## ⚡ Option A: One-Click Setup & Train (New PC)
If you want to move this project to another PC and start training:

1. Copy the entire `wro code` folder to the new PC.
2. Double-click **`setup_and_train.bat`**.
3. It will automatically:
   - Create a Python virtual environment.
   - Install all AI libraries (`ultralytics`, `opencv`, etc.).
   - Generate 1,000 synthetic LEGO images.
   - Start training the YOLOv8 model for you.

---

## 🌐 Option B: Running Online Mode (Google Gemini)
This mode uses the **Gemini 2.0 Flash** model. It is the most accurate for identifying complex objects like the **Barrier**, **Trowel**, and **Bowl** without any training.

1. Ensure your webcam is connected.
2. Double-click **`run_online_mode.bat`**.
   - *It uses the API key you provided:* `AIzaSyAGFK-4068Wy_iwqyrNm6v6VceQAY2M4fk`
3. It will capture from your camera and print the detected objects to the console.

---

## 🚗 Option C: Running the Offline Model (Car Camera)
This is for high-speed, real-time detection on a car or laptop without internet.

1. To run with the **currently training** weights:
   ```bash
   python detect.py --weights runs/detect/lego_model_v1/weights/best.pt
   ```
2. **Key Controls:**
   - `q` — Quit
   - `s` — Save Screenshot
   - `c` — Show Crosshair (for aiming camera)
   - `+/-` — Adjust confidence threshold

---

## 🛠️ Troubleshooting

### 1. Camera not opening?
If you have multiple cameras, change the `--source` number.
- For most, it's `0`. For some laptops with multiple cams, try `1`.
- Edit `run_online_mode.bat` and change `--source 0` to `--source 1`.

### 2. "ModuleNotFoundError"?
Run this to fix:
```bash
pip install -r requirements.txt
pip install google-generativeai PyPDF2
```

### 3. Model file too big?
If deploying to a Raspberry Pi, use the **Nano** model:
```bash
python train.py --model yolov8n.pt --epochs 50
```
Then use `best.pt` from the `weights/` folder.
