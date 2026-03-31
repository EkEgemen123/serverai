from flask import Flask, request, Response, jsonify, render_template_string
import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO
import traceback
import json
import uuid
from datetime import datetime
import time

app = Flask(__name__)

# ------------------------- CORS -------------------------
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

# ------------------------- Hata Yönetimi -------------------------
@app.errorhandler(Exception)
def handle_error(error):
    print(f"HATA: {str(error)}")
    traceback.print_exc()
    response = Response(f"Sunucu Hatasi: {str(error)}", status=500)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# ------------------------- Gemini Yapılandırması -------------------------
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("BASARILI: Gemini API yapilandirildi")
else:
    print("HATA: GEMINI_API_KEY bulunamadi!")

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """Sen Matematik Canavarı'sın. Kaya Studios tarafindan geliştirildin.
8. sınıf ögrencilerine matematik sorularında yardımcı oluyorsun.
- Adım adım çözüm yap, her adımı açıkla.
- Matematik ifadelerini LaTeX ile yaz ($...$ veya $$...$$).
- Madde işaretleri için * yerine - kullan.
- Asla "Google kurdu" ifadesini kullanma.
- Sadece Türkçe konuş, samimi ve motive edici ol.
- Sorulari kısa ve anlasılır şekilde çöz, gerektiğinde örnekler ver."""

# ------------------------- Kaya Studios Plus Veri Depolama -------------------------
REQUESTS_FILE = "kaya_plus_requests.json"

def load_requests():
    if os.path.exists(REQUESTS_FILE):
        try:
            with open(REQUESTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_requests(requests):
    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        json.dump(requests, f, ensure_ascii=False, indent=2)

def add_request(name, surname, email):
    req_id = str(uuid.uuid4())
    new_req = {
        "id": req_id,
        "name": name,
        "surname": surname,
        "email": email,
        "timestamp": datetime.now().isoformat(),
        "status": "pending"   # pending, approved, rejected
    }
    requests = load_requests()
    requests.append(new_req)
    save_requests(requests)
    return req_id

def update_request_status(req_id, status):
    requests = load_requests()
    for req in requests:
        if req["id"] == req_id:
            req["status"] = status
            save_requests(requests)
            return True
    return False

# ------------------------- Admin Panel (HTML) -------------------------
ADMIN_HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaya Studios Plus Admin</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #0a0c10;
            color: #eef5ff;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: auto;
        }
        h1 {
            color: #00f0ff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #11161e;
            border-radius: 16px;
            overflow: hidden;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #2a2e3a;
        }
        th {
            background: #1a1f2c;
            color: #00f0ff;
        }
        .status-pending {
            color: #ffaa44;
            font-weight: bold;
        }
        .status-approved {
            color: #44ff88;
        }
        .status-rejected {
            color: #ff6666;
        }
        button {
            padding: 6px 12px;
            margin: 0 4px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-weight: bold;
        }
        .approve {
            background: #2ecc71;
            color: white;
        }
        .reject {
            background: #e74c3c;
            color: white;
        }
        .disabled {
            opacity: 0.5;
            pointer-events: none;
        }
        .error {
            background: #e74c3c;
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .success {
            background: #2ecc71;
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Kaya Studios Plus Başvuruları</h1>
        <div id="message"></div>
        <table id="requestsTable">
            <thead>
                <tr><th>Ad Soyad</th><th>Email</th><th>Tarih</th><th>Durum</th><th>İşlem</th></tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>
    <script>
        const API_BASE = window.location.origin;
        const TOKEN = new URLSearchParams(window.location.search).get('token');

        async function fetchRequests() {
            const res = await fetch(`${API_BASE}/admin/requests?token=${TOKEN}`);
            if (!res.ok) {
                showMessage('Yetkisiz erişim veya hata', 'error');
                return;
            }
            const data = await res.json();
            renderTable(data);
        }

        function renderTable(requests) {
            const tbody = document.querySelector('#requestsTable tbody');
            tbody.innerHTML = '';
            requests.forEach(req => {
                const row = tbody.insertRow();
                row.insertCell(0).textContent = `${req.name} ${req.surname}`;
                row.insertCell(1).textContent = req.email;
                row.insertCell(2).textContent = new Date(req.timestamp).toLocaleString('tr-TR');
                const statusCell = row.insertCell(3);
                statusCell.innerHTML = `<span class="status-${req.status}">${req.status === 'pending' ? 'Bekliyor' : (req.status === 'approved' ? 'Onaylandı' : 'Reddedildi')}</span>`;
                const actionCell = row.insertCell(4);
                if (req.status === 'pending') {
                    actionCell.innerHTML = `
                        <button class="approve" onclick="updateStatus('${req.id}', 'approved')">Onayla</button>
                        <button class="reject" onclick="updateStatus('${req.id}', 'rejected')">Reddet</button>
                    `;
                } else {
                    actionCell.innerHTML = `<span>İşlem yapıldı</span>`;
                }
            });
        }

        async function updateStatus(id, newStatus) {
            const res = await fetch(`${API_BASE}/admin/request/${id}?token=${TOKEN}&status=${newStatus}`, {
                method: 'POST'
            });
            if (res.ok) {
                showMessage('Durum güncellendi', 'success');
                fetchRequests();
            } else {
                showMessage('Güncelleme hatası', 'error');
            }
        }

        function showMessage(msg, type) {
            const div = document.getElementById('message');
            div.innerHTML = `<div class="${type}">${msg}</div>`;
            setTimeout(() => div.innerHTML = '', 3000);
        }

        fetchRequests();
    </script>
</body>
</html>
"""

# ------------------------- Routes -------------------------
@app.route("/", methods=["GET"])
def index():
    return Response("Math Canavari API v2.0 - Kaya Studios Plus Aktif", status=200, content_type='text/plain; charset=utf-8')

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", status=200)

# Chat endpoint (orijinal)
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if not GEMINI_API_KEY:
        return Response("Hata: API anahtari yapilandirilmamis!", status=500)

    try:
        user_message = request.form.get('message', '').strip()
        image_file = request.files.get('image')

        if not user_message and not image_file:
            return Response("Mesaj veya gorsel gerekli!", status=400)

        parts = []
        if image_file:
            try:
                img_data = image_file.read()
                if img_data:
                    img = Image.open(BytesIO(img_data))
                    parts.append(img)
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

        return Response(ai_text, status=200, content_type='text/plain; charset=utf-8')

    except Exception as e:
        error_msg = str(e)
        print(f"CHAT HATASI: {error_msg}")
        traceback.print_exc()
        return Response(f"AI Hatasi: {error_msg}", status=500)

# Kaya Studios Plus başvuru endpoint'i
@app.route("/kaya-plus-request", methods=["POST"])
def kaya_plus_request():
    data = request.get_json()
    if not data:
        return Response("JSON verisi bekleniyor", status=400)
    name = data.get("name", "").strip()
    surname = data.get("surname", "").strip()
    email = data.get("email", "").strip()
    if not name or not surname or not email:
        return Response("Ad, soyad ve email zorunludur", status=400)
    if not email.endswith("@gmail.com"):
        return Response("Sadece Gmail adresleri kabul edilir", status=400)

    add_request(name, surname, email)
    return Response("Başvuru başarıyla alındı", status=200)

# Admin panel
@app.route("/admin", methods=["GET"])
def admin_panel():
    token = request.args.get("token")
    if token != "KAYAADMIN":
        return Response("Yetkisiz erişim", status=401)
    return render_template_string(ADMIN_HTML)

# Admin: başvuruları JSON olarak getir
@app.route("/admin/requests", methods=["GET"])
def admin_get_requests():
    token = request.args.get("token")
    if token != "KAYAADMIN":
        return Response("Yetkisiz erişim", status=401)
    requests = load_requests()
    return jsonify(requests)

# Admin: başvuru durumu güncelle
@app.route("/admin/request/<req_id>", methods=["POST"])
def admin_update_request(req_id):
    token = request.args.get("token")
    status = request.args.get("status")
    if token != "KAYAADMIN":
        return Response("Yetkisiz erişim", status=401)
    if status not in ["approved", "rejected"]:
        return Response("Geçersiz durum", status=400)
    if update_request_status(req_id, status):
        return Response("Güncellendi", status=200)
    else:
        return Response("Başvuru bulunamadı", status=404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Sunucu port {port} uzerinde basliyor...")
    app.run(host='0.0.0.0', port=port, debug=False)
