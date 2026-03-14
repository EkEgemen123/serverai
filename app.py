from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import google.generativeai as genai
import os
from PIL import Image 
from io import BytesIO

app = Flask(__name__)

# 1. flask-cors'u varsayılan haliyle başlatıyoruz
CORS(app)

# 2. CORS İÇİN KESİN ÇÖZÜM: Tüm yanıtlara zorla CORS başlıklarını ekliyoruz
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# API Key Kontrolü
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash" 

SYSTEM_INSTRUCTION = (
    "Sen Matematik Canavarı 1.0'sın. Kaya Studios tarafından geliştirildin. "
    "8. sınıf öğrencilerine matematik sorularında yardımcı oluyorsun. "
    "KESİNLİKLE sadece Türkçe konuşmalısın. "
    "Soruları kısa, öz ve anlaşılır bir şekilde çöz."
)

# 3. OPTIONS'ı sildik, sadece POST'a izin veriyoruz çünkü OPTIONS'ı Flask-CORS halledecek.
@app.route("/chat", methods=["POST"])
def chat():
    if not GEMINI_API_KEY:
        return Response("Hata: GEMINI_API_KEY bulunamadı!", status=500)

    user_message = request.form.get('message', '')
    image_file = request.files.get('image')

    try:
        parts = []
        if image_file:
            img_data = image_file.read()
            if img_data:
                img = Image.open(BytesIO(img_data))
                parts.append(img)
        
        if user_message:
            parts.append(user_message)

        if not parts:
            return Response("İçerik boş!", status=400)

        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        )
        
        response = model.generate_content(parts)
        
        return Response(response.text, mimetype='text/plain')

    except Exception as e:
        print(f"KRİTİK HATA: {str(e)}")
        return Response(f"Sunucu Hatası: {str(e)}", status=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
