@echo off
echo ============================================================
2: echo   LEGO Detection - Starting Online Google API Mode
3: echo ============================================================
4: echo.
5: 
6: set API_KEY=AIzaSyAGFK-4068Wy_iwqyrNm6v6VceQAY2M4fk
7: 
8: echo [!] Starting Gemini 2.0 Flash Detection...
9: echo [!] Using your provided API Key.
10: echo.
11: 
12: :: Check if virtual environment exists
13: if exist venv\Scripts\activate (
14:     call venv\Scripts\activate
15: )
16: 
17: python pretrained/detect_online.py --source 0
18: 
19: if %errorlevel% neq 0 (
20:     echo.
21:     echo [X] Error detected. Ensure your camera is connected and 
22:     echo     google-generativeai is installed (pip install google-generativeai).
23: )
24: 
25: pause
