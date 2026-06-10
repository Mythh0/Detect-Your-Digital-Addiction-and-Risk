from flask import Flask, jsonify, request
from flask_cors import CORS
from model import train_model, predict
import threading

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
clf, scaler = train_model()

laptop_data  = {"source": "demo"}
phone_data   = {"source": "demo"}
emotion_data = {"source": "none"}
lock = threading.Lock()

DEMO_LAPTOP = {
    "screen_time": 5.2, "notifications": 95, "night_usage": 0.5,
    "app_opens": 18, "social_media": 1.8, "longest_session": 48.0,
    "top_apps": [
        {"app": "Chrome",  "mins": 92},
        {"app": "VSCode",  "mins": 74},
        {"app": "Slack",   "mins": 45},
        {"app": "Spotify", "mins": 22},
        {"app": "Zoom",    "mins": 18},
    ],
    "source": "demo",
}
DEMO_PHONE = {
    "screen_time": 4.1, "screen_unlocks": 67, "notifications": 112,
    "night_usage": 0.3, "app_opens": 145, "social_media": 2.1,
    "top_apps": [
        {"app": "Instagram", "opens": 34},
        {"app": "WhatsApp",  "opens": 28},
        {"app": "YouTube",   "opens": 15},
        {"app": "Snapchat",  "opens": 12},
        {"app": "Chrome",    "opens": 8},
    ],
    "source": "demo",
}

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

@app.route("/api/laptop-data", methods=["POST","OPTIONS"])
def recv_laptop():
    if request.method == "OPTIONS": return jsonify({}), 200
    body = request.get_json(force=True, silent=True) or {}
    with lock:
        laptop_data.clear()
        laptop_data.update(body)
        laptop_data["source"] = "live"
    return jsonify({"status": "ok"})

@app.route("/api/phone-data", methods=["POST","OPTIONS"])
def recv_phone():
    if request.method == "OPTIONS": return jsonify({}), 200
    body = request.get_json(force=True, silent=True) or {}
    with lock:
        phone_data.clear()
        phone_data.update(body)
        phone_data["source"] = "live"
    return jsonify({"status": "ok"})

# ── NEW: Emotion data endpoint ────────────────────────────────────────────────
@app.route("/api/emotion-data", methods=["POST","OPTIONS"])
def recv_emotion():
    if request.method == "OPTIONS": return jsonify({}), 200
    body = request.get_json(force=True, silent=True) or {}
    with lock:
        emotion_data.clear()
        emotion_data.update(body)
        emotion_data["source"] = "live"
    return jsonify({"status": "ok"})

@app.route("/api/emotion-status", methods=["GET","OPTIONS"])
def emotion_status():
    if request.method == "OPTIONS": return jsonify({}), 200
    with lock:
        return jsonify(dict(emotion_data))

# ── NEW: Dark-mode schedule config ───────────────────────────────────────────
# Stored in memory; POST to update, GET to read
darkmode_config = {
    "enabled": True,
    "dark_start": "20:00",   # 8 PM — switch to dark mode
    "dark_end":   "07:00",   # 7 AM — switch back to light mode
}

@app.route("/api/darkmode-schedule", methods=["GET","POST","OPTIONS"])
def darkmode_schedule():
    if request.method == "OPTIONS": return jsonify({}), 200
    if request.method == "GET":
        return jsonify(darkmode_config)
    body = request.get_json(force=True, silent=True) or {}
    darkmode_config.update({k: v for k, v in body.items() if k in darkmode_config})
    return jsonify({"status": "ok", "config": darkmode_config})

# ── Existing status / predict endpoints ──────────────────────────────────────
@app.route("/api/status", methods=["GET","OPTIONS"])
def status():
    if request.method == "OPTIONS": return jsonify({}), 200
    with lock:
        lap = dict(laptop_data)
        pho = dict(phone_data)
        emo = dict(emotion_data)

    l = lap if lap.get("source") == "live" else DEMO_LAPTOP
    p = pho if pho.get("source") == "live" else DEMO_PHONE

    combined_screen = float(l.get("screen_time",0)) + float(p.get("screen_time",0))
    features = [
        combined_screen,
        float(p.get("screen_unlocks", 50)),
        float(l.get("notifications",0)) + float(p.get("notifications",0)),
        float(l.get("night_usage",0))   + float(p.get("night_usage",0)),
        float(l.get("app_opens",0))     + float(p.get("app_opens",0)),
        float(l.get("social_media",0))  + float(p.get("social_media",0)),
        float(l.get("longest_session",0)),
    ]
    pred = predict(clf, scaler, features)
    return jsonify({
        "laptop": l, "phone": p,
        "prediction": pred,
        "combined_screen": round(combined_screen, 2),
        "emotion": emo,
    })

@app.route("/api/predict", methods=["POST","OPTIONS"])
def api_predict():
    if request.method == "OPTIONS": return jsonify({}), 200
    body = request.get_json(force=True, silent=True) or {}
    try:
        features = [
            float(body.get("screen_time",     5)),
            float(body.get("screen_unlocks",  50)),
            float(body.get("notifications",   80)),
            float(body.get("night_usage",     0)),
            float(body.get("app_opens",       100)),
            float(body.get("social_media",    2)),
            float(body.get("longest_session", 40)),
        ]
        return jsonify(predict(clf, scaler, features))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("🚀 DigitDetox backend on http://localhost:5000")
    app.run(debug=True, port=5000, host="0.0.0.0")
