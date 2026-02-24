from flask import Flask, request, jsonify, Response
from flask_cors import CORS  # CORS kütüphanesi şart
import google.generativeai as genai
import os
from PIL import Image 
from io import BytesIO
import threading
import time
import requests

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'mat-canavari-gizli-key-9988')

# --- CORS AYARI (Hatanın Çözümü Burası) ---
CORS(app) 

# --- Gemini API Ayarları ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash" # Veya kullandığın model
SYSTEM_INSTRUCTION = "Sen Matematik Canavarı 1.0'sın. Kaya Studios tarafından geliştirildin. Soruları canavar gibi çöz ve karşındakine anlat!"

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

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get('message', '')
    image_file = request.files.get('image')

    try:
        parts = []
        if image_file:
            img = Image.open(BytesIO(image_file.read()))
            parts.append(img)
        if user_message:
            parts.append(user_message)

        model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION)
        chat_session = model.start_chat(history=[])
        response_stream = chat_session.send_message_stream(parts)

        def generate():
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        return Response(f"Hata: {str(e)}", status=500)

if __name__ == "__main__":
    if os.environ.get('RENDER'):
        threading.Thread(target=keep_alive, daemon=True).start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
