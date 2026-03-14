from flask import Flask, request, make_response
import google.generativeai as genai
import os
from PIL import Image 
from io import BytesIO

app = Flask(__name__)

# =============================================================
# CORS - TÜM YANITLARA HEADER EKLE (EN ÖNEMLİ KISIM)
# =============================================================
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

# =============================================================
# API KEY KONTROLÜ
# =============================================================
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "Sen Matematik Canavarı 1.0'sın. Kaya Studios tarafından geliştirildin. "
    "8. sınıf öğrencilerine matematik sorularında yardımcı oluyorsun. "
    "KESİNLİKLE sadece Türkçe konuşmalısın. "
    "Soruları kısa, öz ve anlaşılır bir şekilde çöz. "
    "Matematiksel ifadeleri LaTeX formatında yaz ($ ve $$ kullanarak)."
)

# =============================================================
# ANA SAYFA - SUNUCU KONTROL
# =============================================================
@app.route("/", methods=["GET"])
def index():
    return "Math Canavari API v2.0 - Aktif!"

# =============================================================
# CHAT ENDPOINT
# =============================================================
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # OPTIONS (Preflight) isteği için boş yanıt
    if request.method == "OPTIONS":
        return make_response("", 204)
    
    # API Key kontrolü
    if not GEMINI_API_KEY:
        return make_response("Hata: GEMINI_API_KEY sunucuda tanımlı değil!", 500)
    
    # Kullanıcı mesajını al
    user_message = request.form.get('message', '')
    image_file = request.files.get('image')
    
    # İçerik kontrolü
    if not user_message and not image_file:
        return make_response("Hata: Mesaj veya görsel gerekli!", 400)
    
    try:
        parts = []
        
        # Görsel varsa işle
        if image_file:
            try:
                img_data = image_file.read()
                if img_data:
                    img = Image.open(BytesIO(img_data))
                    parts.append(img)
            except Exception as img_error:
                print(f"Görsel işleme hatası: {str(img_error)}")
        
        # Mesaj varsa ekle
        if user_message:
            parts.append(user_message)
        
        # Parts boşsa hata ver
        if not parts:
            return make_response("Hata: İşlenecek içerik bulunamadı!", 400)
        
        # Gemini modeli oluştur
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        )
        
        # AI yanıtı al
        ai_response = model.generate_content(parts)
        
        # Başarılı yanıt
        return make_response(ai_response.text, 200)
    
    except Exception as e:
        error_message = str(e)
        print(f"SUNUCU HATASI: {error_message}")
        return make_response(f"Sunucu Hatası: {error_message}", 500)

# =============================================================
# SUNUCUYU BAŞLAT
# =============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Sunucu {port} portunda başlatılıyor...")
    app.run(host='0.0.0.0', port=port, debug=False)
