@echo off
echo ============================================================
2: echo   LEGO Detection Project - One-Click Setup & Train
3: echo ============================================================
4: echo.
5: 
6: :: 1. Check if Python is installed
7: python --version >nul 2>&1
8: if %errorlevel% neq 0 (
9:     echo [X] Python not found! Please install Python 3.10+ from python.org
10:     pause
11:     exit /b
12: )
13: 
14: :: 2. Create Virtual Environment
15: echo [1/4] Creating virtual environment (venv)...
16: python -m venv venv
17: call venv\Scripts\activate
18: 
19: :: 3. Install Dependencies
20: echo [2/4] Installing requirements (this may take a few mins)...
21: pip install --upgrade pip
22: pip install -r requirements.txt
23: pip install PyPDF2
24: 
25: :: 4. Generate Synthetic Data
26: echo [3/4] Generating training data...
27: python scripts/generate_synthetic_data.py
28: 
29: :: 5. Start Training
30: echo [4/4] Starting YOLOv8 Training (50 epochs)...
31: echo Model will be saved to: runs/detect/lego_model_other_pc/
32: python train.py --epochs 50 --batch 16 --model yolov8s.pt --name lego_model_other_pc
33: 
34: echo.
35: echo ============================================================
36: echo   DONE! Trained model found in: runs/detect/lego_model_other_pc/weights/best.pt
37: echo ============================================================
38: pause
