from flask import Flask, request, Response, jsonify, render_template_string
import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO
import traceback
import json
import uuid
from datetime import datetime, timezone, timedelta
import time
from collections import defaultdict
import re
import requests as http_requests
from urllib.parse import urlparse

app = Flask(__name__)

# ========================= CORS =========================
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

# ========================= GEMİNİ =========================
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API yapılandırıldı")
else:
    print("❌ HATA: GEMINI_API_KEY bulunamadı!")

MODEL_NAME        = "gemini-2.5-flash"
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

# ========================= SABİTLER =========================
MAX_MSG_LENGTH    = 4000
MAX_IMAGE_SIZE_MB = 10

# Kaynakları body'ye gömmek için ayraç
SOURCES_SEPARATOR = "|||SOURCES|||"

# ========================= ARAŞTIRMA TESPİT =========================
RESEARCH_PATTERNS = [
    (r"(kimdir|kimdi|kim\s+o|hakkında\s+bilgi|hayatı\s+hakkında|biyografisi)", "person_query"),
    (r"ne\s*zaman\s*(doğdu|öldü|doğmuş|ölmüş|vefat\s*etti|kuruldu|keşfedildi|icat\s*edildi|bulundu|başladı|bitti|oldu|yapıldı|açıldı|kapandı)", "event"),
    (r"(doğum|ölüm|vefat|kuruluş)\s*(tarihi|günü|yılı|senesi)", "event"),
    (r"hangi\s*(tarih|yıl|gün|ay|dönem|çağ|yüzyıl)\s*(de|da|te|ta)?", "event"),
    (r"\b(atatürk|mustafa\s*kemal|einstein|newton|tesla|edison|mozart|beethoven|"
     r"da\s*vinci|leonardo|picasso|shakespeare|fatih\s*sultan|kanuni|yavuz\s*sultan|"
     r"mimar\s*sinan|nazım\s*hikmet|yunus\s*emre|mehmet\s*akif|baris\s*manço|"
     r"zeki\s*müren|tarkan|elon\s*musk|steve\s*jobs|bill\s*gates|mark\s*zuckerberg|"
     r"jeff\s*bezos|alan\s*turing|marie\s*curie|nikola\s*tesla|stephen\s*hawking|"
     r"galileo|kopernik|kepler|pythagoras|pisagor|arsimed|archimedes|öklid|euclid|"
     r"euler|gauss|fibonacci|fermat|pascal|descartes|leibniz|riemann|hilbert|"
     r"ramanujan|emmy\s*noether|ada\s*lovelace|al-?harizmi|harezmi|ali\s*kuscu|"
     r"ulug\s*bey|ibn-?i?\s*sina|farabi|biruni|hayyam|ömer\s*hayyam|cahit\s*arf|"
     r"enes\s*batur|burak\s*dogan|jahrein|irem\s*derici|selin\s*cigerci|"
     r"erdogan|demirel|özal)\b", "famous_person"),
    (r"(anneler\s*günü|babalar\s*günü|sevgililer\s*günü|ögretmenler\s*günü|"
     r"dünya\s*\w+\s*günü|cumhuriyet\s*bayrami|zafer\s*bayrami|19\s*mayis|"
     r"23\s*nisan|30\s*agustos|29\s*ekim|ramazan\s*bayrami|kurban\s*bayrami|"
     r"yilbasi|noel|nevruz|hidrellez|kadinlar\s*günü|çocuk\s*bayrami|"
     r"isci\s*bayrami|1\s*mayis|pi\s*günü|matematik\s*günü)", "special_day"),
    (r"(nüfusu|baskenti|para\s*birimi|yüzölçümü|en\s*büyük\s*sehri|resmi\s*dili)\s*(ne|kaç|nedir|hakkında)", "world_info"),
    (r"dünya\w*\s*(en\s*büyük|en\s*küçük|en\s*uzun|en\s*kisa|en\s*yüksek|en\s*derin|en\s*genis|en\s*hizli|en\s*agir|en\s*sicak|en\s*soguk|en\s*kalabalik)", "world_record"),
    (r"\b(kim\s*tarafindan|kimin\s*eseri|kim\s*yazdi|kim\s*buldu|kim\s*kesfetti|kim\s*gelistirdi|kim\s*icat\s*etti|kim\s*besteledi|kim\s*tasarladi|kim\s*kurdu)\b", "who"),
    (r"kaç\s*(yilinda|senesinde|tarihinde)", "year"),
    (r"(tarihi|tarihçesi)\s*(nedir|ne|hakkinda)", "history"),
    (r"(hangi\s*bilim\s*insani|hangi\s*matematikçi|hangi\s*fizikçi|hangi\s*kimyager|hangi\s*mühendis|hangi\s*mimar|hangi\s*sanatçi|hangi\s*yazar|hangi\s*sair)", "scientist"),
    (r"(su\s*an|güncel|son\s*durum|günümüzde|bu\s*yil\s*kaç)", "current"),
    (r"(kaç\s*yasinda|yasiyor\s*mu|hayatta\s*mi|sag\s*mi|ne\s*zaman\s*öldü)", "alive"),
    (r"(nerede\s*dogdu|nerede\s*öldü|nerede\s*yasiyor|mezari\s*nerede)", "location"),
    (r"(formül|teorem|kural|yasa|kanun)\w*\s*(kimin|kim\s*tarafindan|ne\s*zaman|hangi\s*yil)", "formula"),
    (r"(pi\s*sayisi|euler\s*sayisi|altin\s*oran|fibonacci)\w*\s*(ne|nedir|kim|tarih)", "math_concept"),
    (r"(\d{1,2})\s*(ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylül|ekim|kasim|aralik)\s*(ne\s*oldu|nedir|önemi)", "date_specific"),
    (r"(youtuber|oyuncu|sarkici|futbolcu|sporcu|siyasetçi|bilim\s*insani|yazar|sair|ressam|müzisyen)\s*(kimdir|nedir|hakkinda)", "celebrity"),
    (r"(savasi|depremi|felaketi|olayi|harekat|operasyon)\w*\s*(nedir|ne|hakkinda|ne\s*zaman)", "event_query"),
]

PURE_MATH_PATTERNS = [
    r"^\s*[\d\s\+\-\*\/\(\)\^\.\,\=\<\>]+\s*$",
    r"^(hesapla|çöz|bul|basitleştir|sadeleştir|türev\s+al|integral\s+al|limit\s+bul|matris|denklem\s+çöz|esitsizlik\s+çöz)\s+[\d\(]",
    r"^(sin|cos|tan|cot|log|ln|sqrt|karekök)\s*[\(\d]",
    r"^\d+\s*[\+\-\*\/\^]\s*\d+\s*$",
    r"^(türev|integral|limit|matris|determinant|faktöriyel)\s+[\d\(xa-z]",
]

MATH_ONLY_KEYWORDS = [
    "çarpanlarına ayır", "sadeleştir", "denklem çöz", "eşitsizlik",
    "koordinat", "fonksiyon çiz", "grafik çiz", "olasılık hesapla",
    "permütasyon", "kombinasyon", "logaritma hesapla",
]


def clean_for_header(text: str, max_len: int = 100) -> str:
    """HTTP header için güvenli string — newline, CR ve non-ASCII kaldır."""
    cleaned = re.sub(r'[\r\n\t]', ' ', text)
    cleaned = cleaned.encode('ascii', 'ignore').decode('ascii')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned[:max_len]


def needs_research(text: str):
    lower = text.lower().strip()
    if len(lower) < 5:
        return False, ""

    # 1. Tamamen matematiksel bir ifade veya işlem komutu mu?
    for pattern in PURE_MATH_PATTERNS:
        if re.match(pattern, lower, re.IGNORECASE):
            return False, ""
    for kw in MATH_ONLY_KEYWORDS:
        if kw in lower:
            return False, ""

    # 2. Sohbet/Selamlaşma ise araştırma yapma
    chat_patterns = [r"^(merhaba|selam|nasılsın|naber|iyi\s*günler|iyi\s*akşamlar|günaydın|hey|alo)$", r"^teşekkür", r"^sağol", r"^ok(ey)?$"]
    for cp in chat_patterns:
        if re.search(cp, lower):
            return False, ""

    query = text.strip()
    query = re.sub(
        r"\b(lütfen|acaba|bana\s*söyle|söyler\s*misin|öğrenebilir\s*miyim|merak\s*ediyorum|bana\s*anlat)\b",
        "", query, flags=re.IGNORECASE
    ).strip()
    query = re.sub(r'[\r\n]+', ' ', query).strip()

    # 3. Bilinen özel araştırma şablonlarından biri mi?
    for pattern, ptype in RESEARCH_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            print(f"[RESEARCH DETECT] type='{ptype}' query='{query}'")
            return True, query

    # 4. Genel soru kalıplarına uyuyorsa (kim, nerede, kaç, nedir vb.) araştır.
    general_question_words = r"\b(kim|nedir|ne|nerede|nasıl|ne\s*zaman|kaç|hangisi|hakkında|kimdir|neden|niçin|fiyatı|son\s*durum|maçı)\b"
    if re.search(general_question_words, lower):
        print(f"[RESEARCH DETECT] type='general_question' query='{query}'")
        return True, query

    # İçinde soru işareti varsa araştır
    if "?" in lower:
        print(f"[RESEARCH DETECT] type='question_mark' query='{query}'")
        return True, query

    return False, ""


def google_search(query: str, num_results: int = 5) -> list[dict]:
    api_key = os.environ.get('GOOGLE_SEARCH_API_KEY', '').strip()
    cx      = os.environ.get('GOOGLE_SEARCH_CX', '').strip()

    print(f"[GOOGLE] key={'VAR(' + api_key[:8] + '...)' if api_key else 'YOK'} cx='{cx}' q='{query[:60]}'")

    if not api_key:
        print("[GOOGLE] ❌ API KEY eksik!")
        return []
    if not cx:
        print("[GOOGLE] ❌ CX eksik!")
        return []

    try:
        params = {
            "key": api_key,
            "cx":  cx,
            "q":   query,
            "num": min(num_results, 10),
            "lr":  "lang_tr",
            "hl":  "tr",
        }
        resp = http_requests.get(GOOGLE_SEARCH_URL, params=params, timeout=10)
        print(f"[GOOGLE] HTTP {resp.status_code}")

        if resp.status_code == 400:
            print(f"[GOOGLE] 400: {resp.text[:300]}")
            return []
        if resp.status_code == 403:
            print(f"[GOOGLE] 403 - Kota veya key sorunu: {resp.text[:200]}")
            return []
        if resp.status_code == 429:
            print("[GOOGLE] 429 - Kota aşıldı")
            return []
        if resp.status_code != 200:
            print(f"[GOOGLE] Hata {resp.status_code}: {resp.text[:200]}")
            return []

        data  = resp.json()
        items = data.get("items", [])
        print(f"[GOOGLE] ✅ {len(items)} sonuç")

        results = []
        for item in items:
            try:
                domain = urlparse(item.get("link", "")).netloc.replace("www.", "")
            except Exception:
                domain = ""
                
            snippet = item.get("snippet", "")
            pagemap = item.get("pagemap", {})
            metatags = pagemap.get("metatags", [])
            if metatags and isinstance(metatags, list) and len(metatags) > 0:
                og_desc = metatags[0].get("og:description", "")
                if og_desc and len(og_desc) > len(snippet):
                    snippet = og_desc

            results.append({
                "title":   item.get("title", "")[:150],
                "link":    item.get("link", ""),
                "snippet": snippet[:600].replace('\n', ' '),
                "domain":  domain,
            })
        return results

    except http_requests.exceptions.Timeout:
        print("[GOOGLE] ⏱ Timeout!")
        return []
    except http_requests.exceptions.ConnectionError:
        print("[GOOGLE] 🔌 Bağlantı hatası!")
        return []
    except Exception as e:
        print(f"[GOOGLE] Hata: {e}")
        traceback.print_exc()
        return []


def format_search_results_for_ai(results: list[dict], query: str) -> str:
    if not results:
        return ""
    lines = [
        f"## Google Araştırma Sonuçları ({len(results)} kaynak)",
        f"Arama: {query}",
        "",
        "Aşağıdaki güncel Google arama sonuçları, senin bilgini güncellemek ve desteklemek içindir:",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"**Kaynak {i}: {r['title']}**")
        lines.append(f"  URL: {r['link']}")
        if r.get("snippet"):
            lines.append(f"  Özet: {r['snippet']}")
        lines.append("")
    lines.append("---")
    lines.append("LÜTFEN ŞUNLARA DİKKAT ET:")
    lines.append("1. Bu özetlerdeki verileri GÜNCEL BİLGİ olarak kabul et ve cevabını buna göre oluştur.")
    lines.append("2. Sadece özetlerdeki kısa cümlelerle yetinme; KENDİ ENGİN BİLGİ BİRİKİMİNİ DE KULLANARAK kullanıcıya kapsamlı, detaylı, profesyonel ve tatmin edici bir yanıt ver.")
    lines.append("3. Cevabının sonuna '📚 Kaynaklar: [kullandığın kaynakların isimleri]' şeklinde bir ibare ekle.")
    return "\n".join(lines)


# ========================= ZAMAN =========================
def get_turkey_time_info():
    now_tr    = datetime.now(timezone.utc) + timedelta(hours=3)
    days_tr   = ["Pazartesi","Salı","Çarşamba","Perşembe","Cuma","Cumartesi","Pazar"]
    months_tr = ["Ocak","Şubat","Mart","Nisan","Mayıs","Haziran",
                 "Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"]
    hour = now_tr.hour
    if 5 <= hour < 12:    tod = "sabah"
    elif 12 <= hour < 17: tod = "öğleden sonra"
    elif 17 <= hour < 21: tod = "akşam"
    else:                  tod = "gece"
    return {
        "time_str":    now_tr.strftime("%H:%M"),
        "date_str":    f"{now_tr.day} {months_tr[now_tr.month-1]} {now_tr.year}",
        "day_name":    days_tr[now_tr.weekday()],
        "time_of_day": tod,
        "full":        f"{days_tr[now_tr.weekday()]}, {now_tr.day} {months_tr[now_tr.month-1]} {now_tr.year} - Saat {now_tr.strftime('%H:%M')} ({tod})",
    }


def build_system_instruction(user_name=None, is_plus=False, research_context=""):
    time_info  = get_turkey_time_info()
    greeting   = f"\nBu kullanıcının adı: {user_name}. Uygun yerlerde '{user_name}' diye seslen." if user_name else ""
    plus_rules = "\n- Bu kullanıcı Kaya Studios Plus üyesidir. Her konuda yardımcı ol.\n- Daha detaylı ve kapsamlı cevaplar ver." if is_plus else ""

    research_block = ""
    if research_context:
        research_block = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GÜNCEL GOOGLE ARAŞTIRMA SONUÇLARI
{research_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    return f"""Sen Math Canavarı'sın — Kaya Studios tarafından geliştirilen, son derece akıllı, güncel bilgilere sahip ve detaylı cevaplar veren bir yapay zeka asistanısın.
Şu anki Türkiye saati: {time_info['full']}{greeting}

KURALLAR:
- Matematik ve eğitim konularında uzmansın. 8. sınıf öğrencilerine de uygun, anlaşılır açıklamalar yaparsın.{plus_rules}
- Matematik sorularını adım adım çöz, LaTeX kullan ($...$ veya $$...$$).
- Madde işareti olarak * yerine - kullan.
- "Google kurdu" veya "Büyük dil modeli" gibi ifadeleri ASLA kullanma.
- Yalnızca Türkçe konuş (başka dilde sorulursa o dilde cevapla).
- Kullanıcının sorusunu EN YÜKSEK KALİTEDE, doyurucu ve detaylı bir şekilde yanıtla. Kısa, baştan savma ve robotik cevaplardan kaçın.
- Kaya Studios kurucusu Egemen KAYA'dır.
- Sen Türk bir yapay zekasın.
{research_block}"""


# ========================= RATE LİMİTİNG =========================
ip_request_log  = defaultdict(list)
ip_plus_req_log = defaultdict(list)
ip_last_request = defaultdict(float)
ip_last_msgs    = defaultdict(list)

RATE_LIMIT_WINDOW   = 60
RATE_LIMIT_MAX_CHAT = 20
RATE_LIMIT_MAX_PLUS = 3
MIN_MSG_INTERVAL    = 1.5
SPAM_REPEAT_LIMIT   = 3

FORBIDDEN_PATTERNS = [
    r"(?i)(prompt\s*inject)",
    r"(?i)(ignore\s+previous\s+instructions)",
    r"(?i)(system\s*:\s*)",
    r"(?i)(jailbreak)",
    r"(?i)(DAN\s+mode)",
]


def get_client_ip():
    fwd = request.headers.get('X-Forwarded-For')
    return fwd.split(',')[0].strip() if fwd else (request.remote_addr or '0.0.0.0')


def check_rate_limit_chat(ip):
    now  = time.time()
    last = ip_last_request[ip]
    if now - last < MIN_MSG_INTERVAL:
        wait = round(MIN_MSG_INTERVAL - (now - last), 1)
        return False, f"Çok hızlı mesaj gönderiyorsunuz. {wait} saniye bekleyin."
    log = [t for t in ip_request_log[ip] if now - t < RATE_LIMIT_WINDOW]
    ip_request_log[ip] = log
    if len(log) >= RATE_LIMIT_MAX_CHAT:
        return False, f"Dakikada en fazla {RATE_LIMIT_MAX_CHAT} mesaj gönderebilirsiniz."
    ip_request_log[ip].append(now)
    ip_last_request[ip] = now
    return True, ""


def check_rate_limit_plus(ip):
    now = time.time()
    log = [t for t in ip_plus_req_log[ip] if now - t < RATE_LIMIT_WINDOW * 10]
    ip_plus_req_log[ip] = log
    if len(log) >= RATE_LIMIT_MAX_PLUS:
        return False, "Çok fazla başvuru denemesi. Daha sonra tekrar deneyin."
    ip_plus_req_log[ip].append(now)
    return True, ""


def check_spam(ip, message):
    clean = message.strip().lower()
    recent = ip_last_msgs[ip][-5:]
    if clean and recent.count(clean) >= SPAM_REPEAT_LIMIT:
        return True, "Aynı mesajı tekrar tekrar gönderiyorsunuz."
    ip_last_msgs[ip].append(clean)
    if len(ip_last_msgs[ip]) > 20:
        ip_last_msgs[ip] = ip_last_msgs[ip][-20:]
    return False, ""


def check_content(message):
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, message):
            return False, "Mesajınız güvenlik filtresine takıldı."
    return True, ""


# ========================= VERİTABANI =========================
REQUESTS_FILE = "kaya_plus_requests.json"


def load_requests():
    if os.path.exists(REQUESTS_FILE):
        try:
            with open(REQUESTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_requests(reqs):
    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        json.dump(reqs, f, ensure_ascii=False, indent=2)


def add_request(name, surname, email):
    req_id  = str(uuid.uuid4())
    reqs    = load_requests()
    reqs.append({
        "id": req_id, "name": name, "surname": surname, "email": email,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    })
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
    reqs = load_requests()
    for req in reqs:
        if req["email"].lower() == email.lower() and req["status"] in ("pending", "approved"):
            return True, req["status"]
    return False, None


def cancel_by_req_id(req_id):
    reqs = load_requests()
    for req in reqs:
        if req["id"] == req_id:
            if req["status"] != "approved":
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


# ========================= ADMİN HTML =========================
ADMIN_HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaya Studios Plus Admin</title>
    <style>
        body{font-family:Arial,sans-serif;background:#0a0c10;color:#eef5ff;padding:20px;}
        .container{max-width:1300px;margin:auto;}
        h1{color:#00f0ff;margin-bottom:20px;}
        .stats{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;}
        .stat-card{background:#11161e;border:1px solid #2a2e3a;border-radius:12px;padding:16px 24px;min-width:140px;text-align:center;}
        .stat-card .num{font-size:2rem;font-weight:bold;color:#00f0ff;}
        .stat-card .lbl{font-size:.8rem;color:#9aaec9;margin-top:4px;}
        table{width:100%;border-collapse:collapse;background:#11161e;border-radius:16px;overflow:hidden;}
        th,td{padding:11px 12px;text-align:left;border-bottom:1px solid #2a2e3a;font-size:.88rem;}
        th{background:#1a1f2c;color:#00f0ff;}
        .status-pending{color:#ffaa44;font-weight:bold;}
        .status-approved{color:#44ff88;font-weight:bold;}
        .status-rejected{color:#ff6666;font-weight:bold;}
        .status-cancelled{color:#aaa;font-weight:bold;}
        button{padding:5px 12px;margin:0 3px;border:none;border-radius:20px;cursor:pointer;font-weight:bold;font-size:.82rem;}
        .approve{background:#2ecc71;color:white;}.reject{background:#e74c3c;color:white;}.cancel-btn{background:#e67e22;color:white;}
        .approve:hover{background:#27ae60;}.reject:hover{background:#c0392b;}.cancel-btn:hover{background:#d35400;}
        .error{background:#e74c3c33;border:1px solid #e74c3c;color:#ff9999;padding:10px 16px;border-radius:8px;margin-bottom:20px;}
        .success{background:#2ecc7133;border:1px solid #2ecc71;color:#99ffcc;padding:10px 16px;border-radius:8px;margin-bottom:20px;}
        .refresh-btn{background:#00f0ff;color:#0a0c10;margin-bottom:16px;padding:8px 20px;border-radius:20px;font-weight:bold;border:none;cursor:pointer;}
        .time-info{font-size:.8rem;color:#9aaec9;margin-bottom:16px;}
        .cancelled-by{font-size:.72rem;color:#888;margin-top:2px;}
        .search-status{background:#1a2a3a;border:1px solid #00f0ff33;border-radius:8px;padding:8px 14px;font-size:.8rem;color:#60a5fa;margin-bottom:16px;}
    </style>
</head>
<body>
<div class="container">
    <h1>🛡️ Kaya Studios Plus Admin Paneli</h1>
    <div class="time-info" id="timeInfo"></div>
    <div class="search-status" id="searchStatus">Google Arama Durumu kontrol ediliyor...</div>
    <div class="stats" id="statsArea"></div>
    <button class="refresh-btn" onclick="fetchRequests()">🔄 Yenile</button>
    <div id="message"></div>
    <table id="requestsTable">
        <thead><tr><th>Ad Soyad</th><th>Email</th><th>Başvuru Tarihi</th><th>Durum</th><th>İşlem</th></tr></thead>
        <tbody></tbody>
    </table>
</div>
<script>
const API_BASE=window.location.origin;
const TOKEN=new URLSearchParams(window.location.search).get('token');
function updateClock(){
    document.getElementById('timeInfo').textContent='Türkiye Saati: '+
    new Date().toLocaleString('tr-TR',{timeZone:'Europe/Istanbul',weekday:'long',
    year:'numeric',month:'long',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
updateClock();setInterval(updateClock,1000);
async function checkSearchStatus(){
    try{
        const res=await fetch(`${API_BASE}/search-status?token=${TOKEN}`);
        const data=await res.json();
        const el=document.getElementById('searchStatus');
        if(data.configured){
            el.style.borderColor='#44ff8844';el.style.color='#44ff88';
            el.textContent='✅ Google Custom Search API aktif — Gerçek araştırma çalışıyor';
        }else{
            el.style.borderColor='#ff666644';el.style.color='#ff9999';
            el.textContent='⚠️ Google API yapılandırılmamış! key='+data.has_api_key+' cx='+data.has_cx;
        }
    }catch(e){}
}
async function fetchRequests(){
    const res=await fetch(`${API_BASE}/admin/requests?token=${TOKEN}`);
    if(!res.ok){showMessage('Yetkisiz erişim veya hata','error');return;}
    const data=await res.json();renderStats(data);renderTable(data);
}
function renderStats(r){
    const t=r.length,p=r.filter(x=>x.status==='pending').length,
    a=r.filter(x=>x.status==='approved').length,
    rj=r.filter(x=>x.status==='rejected').length,
    c=r.filter(x=>x.status==='cancelled').length;
    document.getElementById('statsArea').innerHTML=
    `<div class="stat-card"><div class="num">${t}</div><div class="lbl">Toplam</div></div>
     <div class="stat-card"><div class="num" style="color:#ffaa44">${p}</div><div class="lbl">Bekliyor</div></div>
     <div class="stat-card"><div class="num" style="color:#44ff88">${a}</div><div class="lbl">Onaylı</div></div>
     <div class="stat-card"><div class="num" style="color:#ff6666">${rj}</div><div class="lbl">Reddedildi</div></div>
     <div class="stat-card"><div class="num" style="color:#aaa">${c}</div><div class="lbl">İptal</div></div>`;
}
function renderTable(requests){
    const tbody=document.querySelector('#requestsTable tbody');tbody.innerHTML='';
    [...requests].sort((a,b)=>new Date(b.timestamp)-new Date(a.timestamp)).forEach(req=>{
        const row=tbody.insertRow();
        row.insertCell(0).textContent=`${req.name} ${req.surname}`;
        row.insertCell(1).textContent=req.email;
        row.insertCell(2).textContent=new Date(req.timestamp).toLocaleString('tr-TR',{timeZone:'Europe/Istanbul'});
        const labels={pending:'Bekliyor',approved:'Onaylandı',rejected:'Reddedildi',cancelled:'İptal Edildi'};
        const sc=row.insertCell(3);
        let sh=`<span class="status-${req.status}">${labels[req.status]||req.status}</span>`;
        if(req.status==='cancelled'&&req.cancelled_by)
            sh+=`<div class="cancelled-by">${req.cancelled_by==='user'?'👤 Kullanıcı':'🛡️ Admin'} iptal etti</div>`;
        sc.innerHTML=sh;
        const ac=row.insertCell(4);
        if(req.status==='pending')
            ac.innerHTML=`<button class="approve" onclick="updateStatus('${req.id}','approved')">✅ Onayla</button>
                          <button class="reject" onclick="updateStatus('${req.id}','rejected')">❌ Reddet</button>`;
        else if(req.status==='approved')
            ac.innerHTML=`<button class="cancel-btn" onclick="adminCancel('${req.id}')">🚫 İptal Et</button>`;
        else ac.innerHTML='<span style="color:#555">—</span>';
    });
}
async function updateStatus(id,s){
    const res=await fetch(`${API_BASE}/admin/request/${id}?token=${TOKEN}&status=${s}`,{method:'POST'});
    showMessage(res.ok?'Durum güncellendi.':'Hata',res.ok?'success':'error');
    if(res.ok)fetchRequests();
}
async function adminCancel(id){
    if(!confirm('Üyeliği iptal etmek istediğinizden emin misiniz?'))return;
    const res=await fetch(`${API_BASE}/admin/cancel/${id}?token=${TOKEN}`,{method:'POST'});
    showMessage(res.ok?'İptal edildi.':`Hata: ${await res.text()}`,res.ok?'success':'error');
    if(res.ok)fetchRequests();
}
function showMessage(msg,type){
    const d=document.getElementById('message');
    d.innerHTML=`<div class="${type}">${msg}</div>`;
    setTimeout(()=>d.innerHTML='',3000);
}
checkSearchStatus();fetchRequests();setInterval(fetchRequests,30000);
</script>
</body>
</html>
"""


# ========================= ROUTES =========================
@app.route("/", methods=["GET"])
def index():
    gk = bool(os.environ.get('GOOGLE_SEARCH_API_KEY', '').strip())
    cx = bool(os.environ.get('GOOGLE_SEARCH_CX', '').strip())
    return Response(
        f"Math Canavari API v4.2\nGemini: {'OK' if GEMINI_API_KEY else 'MISSING'}\n"
        f"Google Search: {'OK' if (gk and cx) else 'MISSING'}",
        status=200, content_type='text/plain; charset=utf-8'
    )


@app.route("/health", methods=["GET"])
def health():
    gk = bool(os.environ.get('GOOGLE_SEARCH_API_KEY', '').strip())
    cx = bool(os.environ.get('GOOGLE_SEARCH_CX', '').strip())
    return jsonify({
        "status":         "OK",
        "turkey_time":    get_turkey_time_info()["full"],
        "version":        "4.2",
        "gemini":         bool(GEMINI_API_KEY),
        "google_search":  gk and cx,
        "google_key_set": gk,
        "google_cx_set":  cx,
    })


@app.route("/debug-env", methods=["GET"])
def debug_env():
    gk = os.environ.get('GOOGLE_SEARCH_API_KEY', '')
    cx = os.environ.get('GOOGLE_SEARCH_CX', '')
    return jsonify({
        "GEMINI_API_KEY_set":           bool(GEMINI_API_KEY),
        "GOOGLE_SEARCH_API_KEY_set":    bool(gk),
        "GOOGLE_SEARCH_API_KEY_prefix": (gk[:10] + "...") if gk else "YOK",
        "GOOGLE_SEARCH_CX_set":         bool(cx),
        "GOOGLE_SEARCH_CX_value":       cx if cx else "YOK",
        "all_env_keys": [k for k in os.environ.keys() if "GOOGLE" in k or "GEMINI" in k],
    })


@app.route("/search-status", methods=["GET"])
def search_status():
    if request.args.get("token") != "KAYAADMIN":
        return Response("Yetkisiz erişim", status=401)
    gk = os.environ.get('GOOGLE_SEARCH_API_KEY', '').strip()
    cx = os.environ.get('GOOGLE_SEARCH_CX', '').strip()
    return jsonify({
        "configured":  bool(gk and cx),
        "has_api_key": bool(gk),
        "has_cx":      bool(cx),
        "cx_value":    cx or "YOK",
    })


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if not GEMINI_API_KEY:
        return Response("Hata: GEMINI_API_KEY yapılandırılmamış!", status=500)

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
        if user_message and len(user_message) > MAX_MSG_LENGTH:
            return Response(f"Mesaj çok uzun. Maksimum {MAX_MSG_LENGTH} karakter.", status=400)

        if user_message:
            is_spam, spam_err = check_spam(ip, user_message)
            if is_spam:
                return Response(spam_err, status=429)
            ok, content_err = check_content(user_message)
            if not ok:
                return Response(content_err, status=400)

        # ─── GOOGLE ARAŞTIRMASI ───
        # ─── GOOGLE ARAŞTIRMASI ───
        research_context = ""
        search_results   = []
        search_performed = False
        search_query     = ""

        if user_message and not image_file:
            # ⭐ Önce ayrı search_query parametresine bak, yoksa user_message'dan çıkar
            raw_search_query = request.form.get('search_query', '').strip()
            if raw_search_query:
                do_research, query = needs_research(raw_search_query)
            else:
                do_research, query = needs_research(user_message)
            
            if do_research:
                print(f"[CHAT] Araştırma: '{query[:60]}'")
                search_results   = google_search(query, num_results=5)
                search_performed = True
                search_query     = query
                if search_results:
                    research_context = format_search_results_for_ai(search_results, query)
                    print(f"[CHAT] {len(search_results)} sonuç AI'a verildi")
                else:
                    print("[CHAT] Sonuç yok — AI kendi bilgisinden yanıtlar")
        # ─── GÖRSEL İŞLEME ───
        parts = []
        if image_file:
            try:
                img_data = image_file.read()
                if not img_data:
                    return Response("Resim dosyası boş.", status=400)
                if len(img_data) / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
                    return Response(f"Resim çok büyük. Maks {MAX_IMAGE_SIZE_MB}MB.", status=400)
                img = Image.open(BytesIO(img_data))
                img.thumbnail((1024, 1024), Image.LANCZOS)
                parts.append(img)
            except Exception as e:
                print(f"[CHAT] Görsel hatası: {e}")
                return Response("Resim okunamadı.", status=400)

        if user_message:
            parts.append(user_message)
        if not parts:
            return Response("İçerik işlenemedi!", status=400)

        # ─── AI YANITI ───
        system_inst = build_system_instruction(
            user_name=user_name or None,
            is_plus=is_plus,
            research_context=research_context,
        )
        model   = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_inst)
        result  = model.generate_content(parts)
        ai_text = result.text

        # ─── KAYNAKLARI BODY'YE GÖM — header sorunu YOK ───
        # Format: <AI_YANITI>|||SOURCES|||<JSON>
        sources_payload = {
            "performed": search_performed and len(search_results) > 0,
            "query":     search_query,
            "count":     len(search_results),
            "sources": [
                {
                    "title":  r["title"],
                    "domain": r["domain"],
                    "link":   r["link"],
                }
                for r in search_results[:5]
            ],
        }
        sources_json  = json.dumps(sources_payload, ensure_ascii=False)
        full_response = ai_text + SOURCES_SEPARATOR + sources_json

        return Response(full_response, status=200, content_type='text/plain; charset=utf-8')

    except Exception as e:
        print(f"[CHAT] HATA: {e}")
        traceback.print_exc()
        return Response(f"AI Hatası: {str(e)}", status=500)


@app.route("/search", methods=["GET", "POST"])
def manual_search():
    if request.method == "GET":
        query = request.args.get("q", "").strip()
        num   = min(int(request.args.get("num", 5)), 10)
    else:
        data  = request.get_json() or {}
        query = data.get("q", "").strip()
        num   = min(int(data.get("num", 5)), 10)

    if not query:
        return jsonify({"error": "q parametresi gerekli"}), 400

    ip = get_client_ip()
    allowed, err = check_rate_limit_chat(ip)
    if not allowed:
        return Response(err, status=429)

    results = google_search(query, num_results=num)
    gk = bool(os.environ.get('GOOGLE_SEARCH_API_KEY', '').strip())
    cx = bool(os.environ.get('GOOGLE_SEARCH_CX', '').strip())
    return jsonify({
        "query":          query,
        "count":          len(results),
        "results":        results,
        "api_configured": gk and cx,
    })


@app.route("/vision", methods=["POST", "OPTIONS"])
def analyze_image():
    if not GEMINI_API_KEY:
        return Response("Hata: GEMINI_API_KEY yapılandırılmamış!", status=500)
    ip = get_client_ip()
    allowed, err = check_rate_limit_chat(ip)
    if not allowed:
        return Response(err, status=429)
    try:
        image_file = request.files.get('image')
        if not image_file:
            return Response("Resim dosyası gerekli.", status=400)
        custom_prompt = request.form.get('prompt', '').strip() or (
            "Bu resmi dikkatlice analiz et. Matematik problemi varsa adım adım çöz. Türkçe yanıtla. LaTeX kullan."
        )
        img_data = image_file.read()
        if not img_data:
            return Response("Resim dosyası boş.", status=400)
        if len(img_data) / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
            return Response(f"Resim çok büyük. Maks {MAX_IMAGE_SIZE_MB}MB.", status=400)
        img = Image.open(BytesIO(img_data))
        img.thumbnail((1024, 1024), Image.LANCZOS)
        model    = genai.GenerativeModel(model_name=MODEL_NAME)
        response = model.generate_content([img, custom_prompt])
        return Response(response.text, status=200, content_type='text/plain; charset=utf-8')
    except Exception as e:
        print(f"[VISION] HATA: {e}")
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
        return Response("JSON verisi bekleniyor.", status=400)
    name    = data.get("name",    "").strip()
    surname = data.get("surname", "").strip()
    email   = data.get("email",   "").strip()
    if not name or not surname or not email:
        return Response("Ad, soyad ve email zorunludur.", status=400)
    if len(name) > 50 or len(surname) > 50:
        return Response("Ad veya soyad çok uzun.", status=400)
    if not email.endswith("@gmail.com"):
        return Response("Sadece Gmail adresleri kabul edilir.", status=400)
    if not re.match(r'^[a-zA-Z0-9._%+\-]+@gmail\.com$', email):
        return Response("Geçersiz Gmail formatı.", status=400)
    already, status = email_already_applied(email)
    if already:
        msg = "Bu email ile zaten onaylanmış bir üyelik var." if status == "approved" else "Bu email ile bekleyen bir başvurunuz var."
        return Response(msg, status=409)
    req_id = add_request(name, surname, email)
    return jsonify({"message": "Başvuru alındı.", "req_id": req_id}), 200


@app.route("/check-plus-status", methods=["GET"])
def check_plus_status():
    req_id = request.args.get("req_id", "").strip()
    if not req_id:
        return Response("req_id gerekli.", status=400)
    try:
        uuid.UUID(req_id)
    except ValueError:
        return Response("Geçersiz req_id.", status=400)
    reqs = load_requests()
    for req in reqs:
        if req["id"] == req_id:
            return jsonify({
                "status":       req["status"],
                "name":         req["name"],
                "surname":      req["surname"],
                "cancelled_by": req.get("cancelled_by", ""),
            }), 200
    return Response("Başvuru bulunamadı.", status=404)


@app.route("/cancel-plus", methods=["POST"])
def cancel_plus():
    data = request.get_json()
    if not data:
        return Response("JSON verisi bekleniyor.", status=400)
    req_id = data.get("req_id", "").strip()
    if not req_id:
        return Response("req_id zorunludur.", status=400)
    try:
        uuid.UUID(req_id)
    except ValueError:
        return Response("Geçersiz req_id.", status=400)
    success, reason = cancel_by_req_id(req_id)
    if success:
        return jsonify({"message": "Abonelik iptal edildi."}), 200
    if reason == "sadece_approved":
        return Response("Yalnızca aktif üyelikler iptal edilebilir.", status=400)
    return Response("Kayıt bulunamadı.", status=404)


@app.route("/admin/cancel/<req_id>", methods=["POST"])
def admin_cancel_subscription(req_id):
    if request.args.get("token") != "KAYAADMIN":
        return Response("Yetkisiz erişim.", status=401)
    try:
        uuid.UUID(req_id)
    except ValueError:
        return Response("Geçersiz req_id.", status=400)
    success, reason = cancel_by_admin(req_id)
    if success:
        return Response("Üyelik iptal edildi.", status=200)
    if reason == "gecersiz_durum":
        return Response("Zaten iptal edilmiş veya beklemede.", status=400)
    return Response("Kayıt bulunamadı.", status=404)


@app.route("/time", methods=["GET"])
def get_time():
    return jsonify(get_turkey_time_info())


@app.route("/admin", methods=["GET"])
def admin_panel():
    if request.args.get("token") != "KAYAADMIN":
        return Response("Yetkisiz erişim.", status=401)
    return render_template_string(ADMIN_HTML)


@app.route("/admin/requests", methods=["GET"])
def admin_get_requests():
    if request.args.get("token") != "KAYAADMIN":
        return Response("Yetkisiz erişim.", status=401)
    return jsonify(load_requests())


@app.route("/admin/request/<req_id>", methods=["POST"])
def admin_update_request(req_id):
    token  = request.args.get("token")
    status = request.args.get("status")
    if token != "KAYAADMIN":
        return Response("Yetkisiz erişim.", status=401)
    if status not in ("approved", "rejected"):
        return Response("Geçersiz durum.", status=400)
    try:
        uuid.UUID(req_id)
    except ValueError:
        return Response("Geçersiz req_id.", status=400)
    if update_request_status(req_id, status):
        return Response("Güncellendi.", status=200)
    return Response("Başvuru bulunamadı.", status=404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print(f"Math Canavari API v4.2 — Port {port}")
    print(f"Gemini:     {'✅' if GEMINI_API_KEY else '❌ YOK'}")
    gk = os.environ.get('GOOGLE_SEARCH_API_KEY', '').strip()
    cx = os.environ.get('GOOGLE_SEARCH_CX', '').strip()
    print(f"Google Key: {'✅ (' + gk[:8] + '...)' if gk else '❌ YOK'}")
    print(f"Google CX:  {'✅ (' + cx + ')' if cx else '❌ YOK'}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)
