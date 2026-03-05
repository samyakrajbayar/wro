import os
import google.generativeai as genai

api_key = os.environ.get("GOOGLE_API_KEY") or "AIzaSyAGFK-4068Wy_iwqyrNm6v6VceQAY2M4fk"
genai.configure(api_key=api_key)

print("--- EXHAUSTIVE MODEL LIST ---")
try:
    models = list(genai.list_models())
    for m in models:
        # Check if it supports generation and doesn't explicitly look like a tuning-only model
        if 'generateContent' in m.supported_generation_methods:
            print(f"NAME: {m.name} | DISPLAY: {m.display_name}")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
