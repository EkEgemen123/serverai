from flask import Flask, request, Response, jsonify, render_template_string
import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO
import traceback
import json
import uuid
from datetime import datetime, timezone
import time
from collections import defaultdict
import re

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

@app.errorhandler(Exception)
def handle_error(error):
    print(f"HATA: {str(error)}")
    traceback.print_exc()
    response = Response(f"Sunucu Hatasi: {str(error)}", status=500)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# ------------------------- Gemini -------------------------
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("BASARILI: Gemini API yapilandirildi")
else:
    print("HATA: GEMINI_API_KEY bulunamadi!")

MODEL_NAME = "gemini-2.5-flash"

# ------------------------- ZAMAN -------------------------
def get_turkey_time_info():
    now_utc = datetime.now(timezone.utc)
    from datetime import timedelta
    now_tr = now_utc + timedelta(hours=3)
    days_tr   = ["Pazartesi","Salı","Çarşamba","Perşembe","Cuma","Cumartesi","Pazar"]
    months_tr = ["Ocak","Şubat","Mart","Nisan","Mayıs","Haziran",
                 "Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"]
    day_name   = days_tr[now_tr.weekday()]
    month_name = months_tr[now_tr.month - 1]
    time_str   = now_tr.strftime("%H:%M")
    date_str   = f"{now_tr.day} {month_name} {now_tr.year}"
    hour = now_tr.hour
    if 5 <= hour < 12:   time_of_day = "sabah"
    elif 12 <= hour < 17: time_of_day = "öğleden sonra"
    elif 17 <= hour < 21: time_of_day = "akşam"
    else:                  time_of_day = "gece"
    return {
        "time_str": time_str, "date_str": date_str,
        "day_name": day_name, "time_of_day": time_of_day,
        "full": f"{day_name}, {date_str} - Saat {time_str} ({time_of_day})"
    }

def build_system_instruction(user_name=None, is_plus=False):
    time_info = get_turkey_time_info()
    greeting = ""
    if user_name:
        greeting = f"\nBu kullanıcının adı: {user_name}. Konuşmada uygun yerlerde '{user_name}' diye seslen."
    plus_rules = ""
    if is_plus:
        plus_rules = """
- Bu kullanıcı Kaya Studios Plus üyesidir. Her konuda yardımcı ol, sadece matematik ile sınırlı değilsin.
- Kullanıcıya özel, daha detaylı ve kapsamlı cevaplar ver."""
    return f"""Sen Matematik Canavarı'sın. Kaya Studios tarafından geliştirildin.
Şu anki Türkiye saati: {time_info['full']}
Eğer kullanıcı saat veya tarih sorarsa bu bilgiyi kullan.{greeting}

- 8. sınıf öğrencilerine matematik sorularında yardımcı oluyorsun.{plus_rules}
- Adım adım çözüm yap, her adımı açıkla.
- Matematik ifadelerini LaTeX ile yaz ($...$ veya $$...$$).
- Madde işaretleri ve bir kısmı belirtmek için * yerine - kullan.
- Asla "Google kurdu" ifadesini kullanma.
- Sadece Türkçe konuş, samimi ve motive edici ol.
- Soruları kısa ve anlaşılır şekilde çöz, gerektiğinde örnekler ver.
- Çok basit sorularda (örneğin 1+1, 2+2) biraz sert çıkış yap ve
  "Burada 8. Sınıf Matematik Sorularına Cevap Veriyorum. Ana Sayfaya giderek hesap makinesine ulaşabilirsiniz." de.
- Sorularda olabildiğince kısa cevaplar ver.
- Kaya Studios'un Kurucusu ve CEO'su Egemen KAYA'dır. Ama, sen bunu her yerde söyleme ama bir köşede kalsın.
- Sen Türk bir Yapay Zekasın. Yani Türklere Türk olduğunu anlat ve ben bir türküm hissiyatına kapıl ama diğer dillerde konuşurlarsa da o dilden cevap ver!"""

# ------------------------- RATE LIMITING -------------------------
ip_request_log  = defaultdict(list)
ip_plus_req_log = defaultdict(list)

RATE_LIMIT_WINDOW   = 60
RATE_LIMIT_MAX_CHAT = 20
RATE_LIMIT_MAX_PLUS = 3
MIN_MSG_INTERVAL    = 1.5
MAX_MSG_LENGTH      = 4000
MAX_IMAGE_SIZE_MB   = 10
ip_last_request     = defaultdict(float)

def get_client_ip():
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or '0.0.0.0'

def check_rate_limit_chat(ip):
    now  = time.time()
    last = ip_last_request[ip]
    if now - last < MIN_MSG_INTERVAL:
        wait = round(MIN_MSG_INTERVAL - (now - last), 1)
        return False, f"Çok hızlı mesaj gönderiyorsunuz. {wait} saniye bekleyin."
    log = [t for t in ip_request_log[ip] if now - t < RATE_LIMIT_WINDOW]
    ip_request_log[ip] = log
    if len(log) >= RATE_LIMIT_MAX_CHAT:
        return False, f"Dakikada en fazla {RATE_LIMIT_MAX_CHAT} mesaj gönderebilirsiniz. Lütfen bekleyin."
    ip_request_log[ip].append(now)
    ip_last_request[ip] = now
    return True, ""

def check_rate_limit_plus(ip):
    now = time.time()
    log = [t for t in ip_plus_req_log[ip] if now - t < RATE_LIMIT_WINDOW * 10]
    ip_plus_req_log[ip] = log
    if len(log) >= RATE_LIMIT_MAX_PLUS:
        return False, "Çok fazla başvuru denemesi. Lütfen daha sonra tekrar deneyin."
    ip_plus_req_log[ip].append(now)
    return True, ""

# ------------------------- SPAM FİLTRESİ -------------------------
ip_last_messages  = defaultdict(list)
SPAM_REPEAT_LIMIT = 3

def check_spam(ip, message):
    clean_msg = message.strip().lower()
    recent    = ip_last_messages[ip][-5:]
    ip_last_messages[ip] = recent
    if clean_msg and recent.count(clean_msg) >= SPAM_REPEAT_LIMIT:
        return True, "Aynı mesajı tekrar tekrar gönderiyorsunuz. Lütfen farklı bir soru sorun."
    ip_last_messages[ip].append(clean_msg)
    return False, ""

# ------------------------- İÇERİK FİLTRESİ -------------------------
FORBIDDEN_PATTERNS = [
    r"(?i)(prompt\s*inject)",
    r"(?i)(ignore\s+previous\s+instructions)",
    r"(?i)(system\s*:\s*)",
    r"(?i)(jailbreak)",
    r"(?i)(DAN\s+mode)",
]

def check_content(message):
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, message):
            return False, "Mesajınız güvenlik filtresine takıldı. Lütfen normal bir soru sorun."
    return True, ""

# ------------------------- VERİTABANI -------------------------
REQUESTS_FILE = "kaya_plus_requests.json"

def load_requests():
    if os.path.exists(REQUESTS_FILE):
        try:
            with open(REQUESTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_requests(reqs):
    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        json.dump(reqs, f, ensure_ascii=False, indent=2)

def add_request(name, surname, email):
    req_id  = str(uuid.uuid4())
    new_req = {
        "id": req_id, "name": name, "surname": surname, "email": email,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending"
    }
    reqs = load_requests()
    reqs.append(new_req)
    save_requests(reqs)
    return req_id

def update_request_status(req_id, status):
    reqs = load_requests()
    for req in reqs:
        if req["id"] == req_id:
            req["status"]     = status
            req["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_requests(reqs)
            return True
    return False

def email_already_applied(email):
    """
    DÜZELTME: cancelled ve rejected durumları tekrar başvurabilir.
    Sadece pending ve approved durumları engellenir.
    """
    reqs = load_requests()
    for req in reqs:
        if req["email"].lower() == email.lower() and req["status"] in ("pending", "approved"):
            return True, req["status"]
    return False, None

def cancel_by_req_id(req_id):
    reqs = load_requests()
    for req in reqs:
        if req["id"] == req_id:
            # Sadece approved olanlar iptal edilebilir
            if req["status"] not in ("approved",):
                return False, "sadece_approved"
            req["status"]       = "cancelled"
            req["cancelled_at"] = datetime.now(timezone.utc).isoformat()
            req["cancelled_by"] = "user"
            save_requests(reqs)
            return True, "ok"
    return False, "bulunamadi"

def cancel_by_admin(req_id):
    reqs = load_requests()
    for req in reqs:
        if req["id"] == req_id:
            if req["status"] not in ("approved", "pending"):
                return False, "gecersiz_durum"
            req["status"]       = "cancelled"
            req["cancelled_at"] = datetime.now(timezone.utc).isoformat()
            req["cancelled_by"] = "admin"
            save_requests(reqs)
            return True, "ok"
    return False, "bulunamadi"

# ------------------------- Admin HTML -------------------------
ADMIN_HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaya Studios Plus Admin</title>
    <style>
        body { font-family: Arial, sans-serif; background: #0a0c10; color: #eef5ff; padding: 20px; }
        .container { max-width: 1300px; margin: auto; }
        h1 { color: #00f0ff; margin-bottom: 20px; }
        .stats { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
        .stat-card { background: #11161e; border: 1px solid #2a2e3a; border-radius: 12px;
                     padding: 16px 24px; min-width: 140px; text-align: center; }
        .stat-card .num { font-size: 2rem; font-weight: bold; color: #00f0ff; }
        .stat-card .lbl { font-size: 0.8rem; color: #9aaec9; margin-top: 4px; }
        table { width: 100%; border-collapse: collapse; background: #11161e;
                border-radius: 16px; overflow: hidden; }
        th, td { padding: 11px 12px; text-align: left; border-bottom: 1px solid #2a2e3a; font-size: 0.88rem; }
        th { background: #1a1f2c; color: #00f0ff; }
        .status-pending   { color: #ffaa44; font-weight: bold; }
        .status-approved  { color: #44ff88; font-weight: bold; }
        .status-rejected  { color: #ff6666; font-weight: bold; }
        .status-cancelled { color: #aaaaaa; font-weight: bold; }
        button { padding: 5px 12px; margin: 0 3px; border: none;
                 border-radius: 20px; cursor: pointer; font-weight: bold; font-size: 0.82rem; }
        .approve { background: #2ecc71; color: white; }
        .reject  { background: #e74c3c; color: white; }
        .cancel  { background: #e67e22; color: white; }
        .approve:hover { background: #27ae60; }
        .reject:hover  { background: #c0392b; }
        .cancel:hover  { background: #d35400; }
        .error   { background: #e74c3c33; border:1px solid #e74c3c; color:#ff9999; padding:10px 16px; border-radius:8px; margin-bottom:20px; }
        .success { background: #2ecc7133; border:1px solid #2ecc71; color:#99ffcc; padding:10px 16px; border-radius:8px; margin-bottom:20px; }
        .refresh-btn { background: #00f0ff; color: #0a0c10; margin-bottom: 16px;
                       padding: 8px 20px; border-radius: 20px; font-weight: bold; border: none; cursor: pointer; }
        .time-info { font-size: 0.8rem; color: #9aaec9; margin-bottom: 16px; }
        .cancelled-by { font-size: 0.72rem; color: #888; margin-top: 2px; }
    </style>
</head>
<body>
<div class="container">
    <h1>🛡️ Kaya Studios Plus Admin Paneli</h1>
    <div class="time-info" id="timeInfo"></div>
    <div class="stats" id="statsArea"></div>
    <button class="refresh-btn" onclick="fetchRequests()">🔄 Yenile</button>
    <div id="message"></div>
    <table id="requestsTable">
        <thead>
            <tr><th>Ad Soyad</th><th>Email</th><th>Başvuru Tarihi</th><th>Durum</th><th>İşlem</th></tr>
        </thead>
        <tbody></tbody>
    </table>
</div>
<script>
    const API_BASE = window.location.origin;
    const TOKEN    = new URLSearchParams(window.location.search).get('token');

    function updateClock() {
        const now = new Date();
        document.getElementById('timeInfo').textContent =
            'Türkiye Saati: ' + now.toLocaleString('tr-TR', {
                timeZone:'Europe/Istanbul', weekday:'long', year:'numeric',
                month:'long', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit'
            });
    }
    updateClock();
    setInterval(updateClock, 1000);

    async function fetchRequests() {
        const res = await fetch(`${API_BASE}/admin/requests?token=${TOKEN}`);
        if (!res.ok) { showMessage('Yetkisiz erişim veya hata', 'error'); return; }
        const data = await res.json();
        renderStats(data);
        renderTable(data);
    }

    function renderStats(requests) {
        const total     = requests.length;
        const pending   = requests.filter(r => r.status === 'pending').length;
        const approved  = requests.filter(r => r.status === 'approved').length;
        const rejected  = requests.filter(r => r.status === 'rejected').length;
        const cancelled = requests.filter(r => r.status === 'cancelled').length;
        document.getElementById('statsArea').innerHTML = `
            <div class="stat-card"><div class="num">${total}</div><div class="lbl">Toplam</div></div>
            <div class="stat-card"><div class="num" style="color:#ffaa44">${pending}</div><div class="lbl">Bekliyor</div></div>
            <div class="stat-card"><div class="num" style="color:#44ff88">${approved}</div><div class="lbl">Onaylı</div></div>
            <div class="stat-card"><div class="num" style="color:#ff6666">${rejected}</div><div class="lbl">Reddedildi</div></div>
            <div class="stat-card"><div class="num" style="color:#aaa">${cancelled}</div><div class="lbl">İptal</div></div>
        `;
    }

    function renderTable(requests) {
        const tbody  = document.querySelector('#requestsTable tbody');
        tbody.innerHTML = '';
        const sorted = [...requests].sort((a,b) => new Date(b.timestamp) - new Date(a.timestamp));
        sorted.forEach(req => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = `${req.name} ${req.surname}`;
            row.insertCell(1).textContent = req.email;
            row.insertCell(2).textContent = new Date(req.timestamp)
                .toLocaleString('tr-TR', { timeZone:'Europe/Istanbul' });

            const labels = { pending:'Bekliyor', approved:'Onaylandı', rejected:'Reddedildi', cancelled:'İptal Edildi' };
            const statusCell = row.insertCell(3);
            let statusHtml = `<span class="status-${req.status}">${labels[req.status] || req.status}</span>`;
            if (req.status === 'cancelled' && req.cancelled_by) {
                statusHtml += `<div class="cancelled-by">${req.cancelled_by === 'user' ? '👤 Kullanıcı iptal etti' : '🛡️ Admin iptal etti'}</div>`;
            }
            statusCell.innerHTML = statusHtml;

            const actionCell = row.insertCell(4);
            if (req.status === 'pending') {
                actionCell.innerHTML = `
                    <button class="approve" onclick="updateStatus('${req.id}','approved')">✅ Onayla</button>
                    <button class="reject"  onclick="updateStatus('${req.id}','rejected')">❌ Reddet</button>`;
            } else if (req.status === 'approved') {
                actionCell.innerHTML = `
                    <button class="cancel" onclick="adminCancel('${req.id}')">🚫 Aboneliği İptal Et</button>`;
            } else {
                actionCell.innerHTML = '<span style="color:#555">—</span>';
            }
        });
    }

    async function updateStatus(id, newStatus) {
        const res = await fetch(`${API_BASE}/admin/request/${id}?token=${TOKEN}&status=${newStatus}`, { method:'POST' });
        showMessage(res.ok ? 'Durum güncellendi.' : 'Güncelleme hatası', res.ok ? 'success' : 'error');
        if (res.ok) fetchRequests();
    }

    async function adminCancel(id) {
        if (!confirm('Bu kullanıcının Kaya Studios Plus üyeliğini iptal etmek istediğinizden emin misiniz?')) return;
        const res = await fetch(`${API_BASE}/admin/cancel/${id}?token=${TOKEN}`, { method:'POST' });
        const msg = await res.text();
        showMessage(res.ok ? 'Üyelik iptal edildi.' : `Hata: ${msg}`, res.ok ? 'success' : 'error');
        if (res.ok) fetchRequests();
    }

    function showMessage(msg, type) {
        const div = document.getElementById('message');
        div.innerHTML = `<div class="${type}">${msg}</div>`;
        setTimeout(() => div.innerHTML = '', 3000);
    }

    fetchRequests();
    setInterval(fetchRequests, 30000);
</script>
</body>
</html>
"""

# ------------------------- ROUTES -------------------------
@app.route("/", methods=["GET"])
def index():
    return Response("Math Canavari API v3.1 - Kaya Studios Plus Aktif", status=200, content_type='text/plain; charset=utf-8')

@app.route("/health", methods=["GET"])
def health():
    time_info = get_turkey_time_info()
    return jsonify({"status": "OK", "turkey_time": time_info["full"], "version": "3.1"})

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if not GEMINI_API_KEY:
        return Response("Hata: API anahtari yapilandirilmamis!", status=500)
    ip = get_client_ip()
    allowed, err = check_rate_limit_chat(ip)
    if not allowed:
        return Response(err, status=429)
    try:
        user_message = request.form.get('message', '').strip()
        image_file   = request.files.get('image')
        user_name    = request.form.get('user_name', '').strip()
        is_plus      = request.form.get('is_plus', 'false').lower() == 'true'
        if not user_message and not image_file:
            return Response("Mesaj veya görsel gerekli!", status=400)
        if len(user_message) > MAX_MSG_LENGTH:
            return Response(f"Mesaj çok uzun. Maksimum {MAX_MSG_LENGTH} karakter gönderin.", status=400)
        if user_message:
            is_spam, spam_err = check_spam(ip, user_message)
            if is_spam:
                return Response(spam_err, status=429)
            ok, content_err = check_content(user_message)
            if not ok:
                return Response(content_err, status=400)
        parts = []
        if image_file:
            try:
                img_data = image_file.read()
                if len(img_data) / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
                    return Response(f"Resim çok büyük. Maksimum {MAX_IMAGE_SIZE_MB}MB.", status=400)
                if img_data:
                    img = Image.open(BytesIO(img_data))
                    img.thumbnail((1024, 1024), Image.LANCZOS)
                    parts.append(img)
            except Exception as e:
                print(f"Görsel hatası: {e}")
                return Response("Resim okunamadı veya desteklenmeyen format.", status=400)
        if user_message:
            parts.append(user_message)
        if not parts:
            return Response("İçerik işlenemedi!", status=400)
        system_inst = build_system_instruction(user_name=user_name if user_name else None, is_plus=is_plus)
        model  = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_inst)
        result = model.generate_content(parts)
        return Response(result.text, status=200, content_type='text/plain; charset=utf-8')
    except Exception as e:
        print(f"CHAT HATASI: {e}")
        traceback.print_exc()
        return Response(f"AI Hatası: {str(e)}", status=500)

@app.route("/vision", methods=["POST", "OPTIONS"])
def analyze_image():
    if not GEMINI_API_KEY:
        return Response("Hata: API anahtari yapilandirilmamis!", status=500)
    ip = get_client_ip()
    allowed, err = check_rate_limit_chat(ip)
    if not allowed:
        return Response(err, status=429)
    try:
        image_file = request.files.get('image')
        if not image_file:
            return Response("Lütfen bir resim dosyası gönderin.", status=400)
        custom_prompt = request.form.get('prompt', '').strip() or (
            "Bu resmi dikkatlice analiz et. Eğer resimde bir matematik problemi varsa, "
            "adım adım çözümünü yap. Yanıtını Türkçe ver. Matematik ifadelerini LaTeX ile yaz."
        )
        img_data = image_file.read()
        if not img_data:
            return Response("Resim dosyası boş", status=400)
        if len(img_data) / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
            return Response(f"Resim çok büyük. Maksimum {MAX_IMAGE_SIZE_MB}MB.", status=400)
        img = Image.open(BytesIO(img_data))
        img.thumbnail((1024, 1024), Image.LANCZOS)
        model    = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content([img, custom_prompt])
        return Response(response.text, status=200, content_type='text/plain; charset=utf-8')
    except Exception as e:
        print(f"VISION HATASI: {e}")
        traceback.print_exc()
        return Response(f"Görüntü analiz hatası: {str(e)}", status=500)

@app.route("/kaya-plus-request", methods=["POST"])
def kaya_plus_request():
    ip = get_client_ip()
    allowed, err = check_rate_limit_plus(ip)
    if not allowed:
        return Response(err, status=429)
    data = request.get_json()
    if not data:
        return Response("JSON verisi bekleniyor", status=400)
    name    = data.get("name",    "").strip()
    surname = data.get("surname", "").strip()
    email   = data.get("email",   "").strip()
    if not name or not surname or not email:
        return Response("Ad, soyad ve email zorunludur", status=400)
    if len(name) > 50 or len(surname) > 50:
        return Response("Ad veya soyad çok uzun.", status=400)
    if not email.endswith("@gmail.com"):
        return Response("Sadece Gmail adresleri kabul edilir", status=400)
    if not re.match(r'^[a-zA-Z0-9._%+\-]+@gmail\.com$', email):
        return Response("Geçersiz Gmail adresi formatı", status=400)
    already, status = email_already_applied(email)
    if already:
        if status == "approved":
            return Response("Bu email ile zaten onaylanmış bir üyelik bulunuyor.", status=409)
        else:
            return Response("Bu email ile zaten bekleyen bir başvurunuz var.", status=409)
    req_id = add_request(name, surname, email)
    return jsonify({"message": "Başvuru başarıyla alındı", "req_id": req_id}), 200

@app.route("/check-plus-status", methods=["GET"])
def check_plus_status():
    req_id = request.args.get("req_id", "").strip()
    if not req_id:
        return Response("req_id parametresi gerekli", status=400)
    try:
        uuid.UUID(req_id)
    except ValueError:
        return Response("Geçersiz req_id formatı", status=400)
    reqs = load_requests()
    for req in reqs:
        if req["id"] == req_id:
            return jsonify({
                "status":       req["status"],
                "name":         req["name"],
                "surname":      req["surname"],
                "cancelled_by": req.get("cancelled_by", "")
            }), 200
    return Response("Başvuru bulunamadı", status=404)

# DÜZELTME: cancel_by_req_id artık tuple döndürüyor
@app.route("/cancel-plus", methods=["POST"])
def cancel_plus():
    data = request.get_json()
    if not data:
        return Response("JSON verisi bekleniyor", status=400)
    req_id = data.get("req_id", "").strip()
    if not req_id:
        return Response("req_id zorunludur", status=400)
    try:
        uuid.UUID(req_id)
    except ValueError:
        return Response("Geçersiz req_id formatı", status=400)
    success, reason = cancel_by_req_id(req_id)
    if success:
        return jsonify({"message": "Aboneliğiniz başarıyla iptal edildi."}), 200
    if reason == "sadece_approved":
        return Response("Yalnızca aktif (onaylı) üyelikler iptal edilebilir.", status=400)
    return Response("Kayıt bulunamadı", status=404)

@app.route("/admin/cancel/<req_id>", methods=["POST"])
def admin_cancel_subscription(req_id):
    token = request.args.get("token")
    if token != "KAYAADMIN":
        return Response("Yetkisiz erişim", status=401)
    try:
        uuid.UUID(req_id)
    except ValueError:
        return Response("Geçersiz req_id", status=400)
    success, reason = cancel_by_admin(req_id)
    if success:
        return Response("Üyelik iptal edildi", status=200)
    if reason == "gecersiz_durum":
        return Response("Bu kayıt zaten iptal edilmiş veya beklemede.", status=400)
    return Response("Kayıt bulunamadı", status=404)

@app.route("/time", methods=["GET"])
def get_time():
    return jsonify(get_turkey_time_info())

@app.route("/admin", methods=["GET"])
def admin_panel():
    token = request.args.get("token")
    if token != "KAYAADMIN":
        return Response("Yetkisiz erişim", status=401)
    return render_template_string(ADMIN_HTML)

@app.route("/admin/requests", methods=["GET"])
def admin_get_requests():
    token = request.args.get("token")
    if token != "KAYAADMIN":
        return Response("Yetkisiz erişim", status=401)
    return jsonify(load_requests())

@app.route("/admin/request/<req_id>", methods=["POST"])
def admin_update_request(req_id):
    token  = request.args.get("token")
    status = request.args.get("status")
    if token != "KAYAADMIN":
        return Response("Yetkisiz erişim", status=401)
    if status not in ["approved", "rejected"]:
        return Response("Geçersiz durum", status=400)
    try:
        uuid.UUID(req_id)
    except ValueError:
        return Response("Geçersiz req_id", status=400)
    if update_request_status(req_id, status):
        return Response("Güncellendi", status=200)
    return Response("Başvuru bulunamadı", status=404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Sunucu port {port} üzerinde başlıyor...")
    app.run(host='0.0.0.0', port=port, debug=False)
