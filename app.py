from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import google.generativeai as genai
import os
from PIL import Image 
from io import BytesIO

app = Flask(__name__)

# En geniş CORS izni
CORS(app, resources={r"/*": {"origins": "*"}})

# API Key'i doğrudan kontrol et
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Model adını 1.5-flash yapalım (Daha stabil ve yaygındır)
# Model adını ve sistem talimatını kontrol et
MODEL_NAME = "gemini-2.5-flash" 
SYSTEM_INSTRUCTION = "Sen Matematik Canavarı 1.0'sın. Kaya Studios tarafından geliştirildin. Matematik Sorularını sana ya resimle verecekler yada yazacaklar. Soruları kısa ve öz çöz!"

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if not GEMINI_API_KEY:
        return Response("Hata: GEMINI_API_KEY bulunamadı! Render panelinden ekle.", status=500)

    user_message = request.form.get('message', '')
    image_file = request.files.get('image')

    try:
        parts = []
        if image_file:
            # Resim işleme hatasını kontrol et
            img_data = image_file.read()
            if img_data:
                img = Image.open(BytesIO(img_data))
                parts.append(img)
        
        if user_message:
            parts.append(user_message)

        if not parts:
            return Response("İçerik boş!", status=400)

        model = genai.GenerativeModel(MODEL_NAME)
        # Stream olmadan dene (bazen stream 500 hatası verebilir)
        response = model.generate_content(parts)
        
        return Response(response.text, mimetype='text/plain')

    except Exception as e:
        # Hatayı terminalde görmek için yazdır
        print(f"KRİTİK HATA: {str(e)}")
        return Response(f"Sunucu Hatası: {str(e)}", status=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

