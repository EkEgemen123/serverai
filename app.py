from flask import Flask, request, Response
import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO
import traceback

app = Flask(__name__)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept, Origin, X-Requested-With'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept, Origin, X-Requested-With'
    return response

@app.errorhandler(Exception)
def handle_error(error):
    print(f"HATA: {str(error)}")
    traceback.print_exc()
    response = Response(f"Sunucu Hatasi: {str(error)}", status=500)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.errorhandler(404)
def not_found(error):
    response = Response("Endpoint bulunamadi", status=404)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.errorhandler(500)
def server_error(error):
    response = Response("Sunucu hatasi", status=500)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("BASARILI: Gemini API yapilandirildi")
else:
    print("HATA: GEMINI_API_KEY bulunamadi!")

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """Sen Matematik Canavarı'sin. Kaya Studios tarafindan geliştirildin.
8. sınıf ögrencilerine matematik sorularında yardımcı oluyorsun.
Sadece Türkçe konuş. Sorulari kısa ve anlasılır şekilde çöz."""

@app.route("/", methods=["GET"])
def index():
    return Response("Math Canavari API v2.0 - Aktif ve Calisiyor!", status=200, content_type='text/plain; charset=utf-8')

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", status=200)

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if not GEMINI_API_KEY:
        return Response("Hata: API anahtari yapilandirilmamis!", status=500)

    try:
        user_message = request.form.get('message', '').strip()
        image_file = request.files.get('image')

        print(f"Gelen mesaj: {user_message[:80] if user_message else 'BOS'}")

        if not user_message and not image_file:
            return Response("Mesaj veya gorsel gerekli!", status=400)

        parts = []

        if image_file:
            try:
                img_data = image_file.read()
                if img_data:
                    img = Image.open(BytesIO(img_data))
                    parts.append(img)
                    print("Gorsel eklendi")
            except Exception as e:
                print(f"Gorsel hatasi: {e}")

        if user_message:
            parts.append(user_message)

        if not parts:
            return Response("Icerik islenemedi!", status=400)

        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        )

        result = model.generate_content(parts)
        ai_text = result.text

        print(f"Yanit olusturuldu: {len(ai_text)} karakter")

        return Response(ai_text, status=200, content_type='text/plain; charset=utf-8')

    except Exception as e:
        error_msg = str(e)
        print(f"CHAT HATASI: {error_msg}")
        traceback.print_exc()
        return Response(f"AI Hatasi: {error_msg}", status=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Sunucu port {port} uzerinde basliyor...")
    app.run(host='0.0.0.0', port=port, debug=False)
