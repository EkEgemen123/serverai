from flask import Flask, render_template, request, jsonify, session, Response
import google.generativeai as genai
from flask_cors import CORS
from google.generativeai import types
import os
from PIL import Image 
from io import BytesIO
import threading
import time
from datetime import datetime
import requests

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'mat-canavari-gizli-key-9988')
CORS(app)

# --- Gemini API Ayarları ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# --- Tek Model: Matematik Canavarı 1.0 Demo ---
MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = """Sen 'Matematik Canavarı 1.0 Demo' versiyonusun. 
LGS ve ortaokul matematik uzmanısın.
Sana gönderilen soruları (resim veya metin) adım adım, anlaşılır ve esprili bir dille çözersin.
Kısa, öz ve etkili konuş."""

def keep_alive():
    while True:
        time.sleep(600)
        try:
            app_url = os.environ.get('RENDER_EXTERNAL_URL') or 'http://localhost:5000'
            requests.get(f"{app_url}/ping", timeout=10)
        except: pass

@app.route("/ping")
def ping():
    return jsonify({"status": "alive"})

def create_multimodal_content(user_message, image_file):
    parts = []
    if image_file:
        try:
            img = Image.open(BytesIO(image_file.read()))
            parts.append(img)
        except Exception as e:
            print(f"Resim hatası: {e}")
    if user_message:
        parts.append(user_message)
    return parts

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get('message', '')
    image_file = request.files.get('image')

    if not user_message and not image_file:
        return Response("Soru nerede dostum?", status=400)

    try:
        content_parts = create_multimodal_content(user_message, image_file)
        model = genai.GenerativeModel(
            MODEL_NAME, 
            system_instruction=SYSTEM_INSTRUCTION
        )

        # Basit bir chat session başlat
        chat_session = model.start_chat(history=[])
        response_stream = chat_session.send_message_stream(content_parts)

        def generate():
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        return Response(f"Bir hata oluştu: {str(e)}", status=500)

if __name__ == "__main__":
    if os.environ.get('RENDER'):
        threading.Thread(target=keep_alive, daemon=True).start()
    
    port = int(os.environ.get("PORT", 5000))

    app.run(host='0.0.0.0', port=port)
