"""
Online Pretrained LEGO Detection — Using Gemini 2.0 Flash

This script uses Google's Gemini 2.0 Flash model to detect LEGO objects
categorized in the WRO 2026 Senior rules. It is 'pretrained' on vast 
datasets and can identify custom objects zero-shot.

Usage:
    python pretrained/detect_online.py --source 0          # Capture from webcam
    python pretrained/detect_online.py --source image.jpg  # Detect from file
"""

import os
import cv2
import PIL.Image
import google.generativeai as genai
import argparse
from pathlib import Path

# Setup API
api_key = os.environ.get("GOOGLE_API_KEY") or "(removethisAIzaSyAGFK-4068Wy_iwqyrNm6v6VceQAY2M4fk)"
genai.configure(api_key=api_key)

def process_image(img):
    """Sends PIL image to Gemini and returns results."""
    # Try 2.0 first, then various 1.5 versions
    models_to_try = [
        'models/gemini-2.0-flash', 
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro',
        'models/gemini-1.5-flash-8b'
    ]
    
    last_error = ""
    for model_name in models_to_try:
        try:
            print(f"📡 Sending to {model_name}...")
            model = genai.GenerativeModel(model_name)
            prompt = """
            Detect the following LEGO objects in this image and return their bounding boxes:
            1. lego_block (yellow, blue, green, white)
            2. lego_rod (yellow, blue, green, white bricks)
            3. barrier (red/black structures with balls on top)
            4. rectangular_trowel (red tool)
            5. cement_bowl (white bowl)
            6. masonry_trowel (white and red tool)
            
            Return the response as a list of detection dictionaries:
            [{"box_2d": [ymin, xmin, ymax, xmax], "label": "class_name"}]
            The coordinates should be normalized (0-1000).
            """
            response = model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            last_error = str(e)
            # If quota hit OR model not found (404), try the next one
            if any(x in last_error for x in ["ResourceExhausted", "429", "404", "not found"]):
                print(f"⚠️  {model_name} unavailable (Quota/404). Trying next model...")
                continue
            else:
                return f"❌ API Error: {last_error}"
                
    return f"❌ All models failed. Last error: {last_error}\n\nTIP: This usually means your Google AI studio quota is full for today. Wait a while or try a different API key."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Camera index (0, 1) or path to image file")
    args = parser.parse_args()
    
    if not api_key:
        print("❌ Error: API Key not found.")
        return

    # Check if source is camera index
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {args.source}")
            return
        
        print(f"📸 Capturing from camera {args.source}...")
        # Buffer clear
        for _ in range(5): cap.read()
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Error: Could not read frame.")
            return
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb_frame)
        print(process_image(img))
        
    else:
        # Source is a file path
        if os.path.exists(args.source):
            print(f"🖼️  Processing image file: {args.source}")
            img = PIL.Image.open(args.source)
            print(process_image(img))
        else:
            print(f"❌ Error: File not found: {args.source}")

if __name__ == "__main__":
    main()
