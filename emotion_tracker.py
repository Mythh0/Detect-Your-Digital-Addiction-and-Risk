"""
DigitDetox — Cognitive Drift Detector
Fixed: sensitive thresholds, debug print so you can see raw signal values
Detects: happy, sad, angry, neutral
"""

import cv2
import threading
import time
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
import urllib.request

from flask import Flask, jsonify
from flask_cors import CORS

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_OK = True
except Exception as e:
    MP_OK = False
    print(f"❌ mediapipe error: {e}")

BACKEND_URL  = "http://localhost:5000/api/emotion-data"
CAMERA_INDEX = 0
MODEL_PATH   = Path("face_landmarker.task")
MODEL_URL    = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ── colours ──────────────────────────────────────────────────────
COLORS_BGR = {
    "happy":   ( 40, 200,  80),
    "sad":     (200, 120,  40),
    "angry":   ( 40,  40, 220),
    "neutral": (160, 160, 160),
}
EMOJIS = {"happy":"😄","sad":"😢","angry":"😠","neutral":"😐"}

# ── landmark indices ──────────────────────────────────────────────
# mouth
MOUTH_LEFT        = 61
MOUTH_RIGHT       = 291
MOUTH_TOP         = 13
MOUTH_BOTTOM      = 14
# lip corners vs lip center
LIP_CORNER_L      = 61
LIP_CORNER_R      = 291
LIP_TOP_CENTER    = 0
LIP_BOTTOM_CENTER = 17
# upper lip
UPPER_LIP_L       = 38
UPPER_LIP_R       = 268
# lower lip
LOWER_LIP_L       = 95
LOWER_LIP_R       = 325
# cheek points (for smile width)
CHEEK_L           = 116
CHEEK_R           = 345
# brows
BROW_L_INNER      = 65
BROW_R_INNER      = 295
BROW_L_MID        = 63
BROW_R_MID        = 293
BROW_L_OUTER      = 46
BROW_R_OUTER      = 276
# eyes
EYE_L_TOP         = 159
EYE_L_BOT         = 145
EYE_L_LEFT        = 33
EYE_L_RIGHT       = 133
EYE_R_TOP         = 386
EYE_R_BOT         = 374
EYE_R_LEFT        = 362
EYE_R_RIGHT       = 263
# face reference
FACE_L            = 234
FACE_R            = 454
FACE_TOP          = 10
FACE_BOT          = 152
# chin
CHIN              = 152
# nose tip
NOSE_TIP          = 1


def pt(lms, i, W, H):
    return np.array([lms[i].x * W, lms[i].y * H])

def d(a, b):
    return float(np.linalg.norm(a - b))


# ════════════════════════════════════════════════════════════════
#  SIGNAL EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_signals(lms, W, H):
    """
    Returns a dict of raw geometric signals — all normalised 0..1 or signed.
    Print these while calibrating to find your personal thresholds.
    """
    fw = d(pt(lms, FACE_L,   W, H), pt(lms, FACE_R,   W, H))
    fh = d(pt(lms, FACE_TOP, W, H), pt(lms, FACE_BOT, W, H))
    if fw < 1 or fh < 1:
        return None

    # ── 1. SMILE RATIO ───────────────────────────────────────────
    # Positive  = corners above lip midpoint  = SMILE
    # Negative  = corners below lip midpoint  = FROWN / SAD
    lc_y   = (pt(lms, LIP_CORNER_L, W, H)[1] + pt(lms, LIP_CORNER_R, W, H)[1]) / 2
    mid_y  = (pt(lms, LIP_TOP_CENTER, W, H)[1] + pt(lms, LIP_BOTTOM_CENTER, W, H)[1]) / 2
    smile  = (mid_y - lc_y) / fh    # +ve = smile, -ve = frown

    # ── 2. MOUTH OPEN RATIO ──────────────────────────────────────
    m_vert  = d(pt(lms, MOUTH_TOP, W, H), pt(lms, MOUTH_BOTTOM, W, H))
    m_horiz = d(pt(lms, MOUTH_LEFT, W, H), pt(lms, MOUTH_RIGHT, W, H))
    mar     = m_vert / (m_horiz + 1e-6)   # 0 = closed, >0.3 = open

    # ── 3. BROW FURROW ───────────────────────────────────────────
    # Smaller = brows closer together = ANGRY
    brow_gap = d(pt(lms, BROW_L_INNER, W, H), pt(lms, BROW_R_INNER, W, H))
    furrow   = brow_gap / fw   # ~0.35 normal | <0.28 furrowed

    # ── 4. BROW HEIGHT ───────────────────────────────────────────
    # Larger = brows higher = SURPRISED / SAD inner brow raise
    lb_y    = (pt(lms, BROW_L_MID, W, H)[1] + pt(lms, BROW_R_MID, W, H)[1]) / 2
    le_y    = (pt(lms, EYE_L_TOP,  W, H)[1] + pt(lms, EYE_R_TOP,  W, H)[1]) / 2
    brow_h  = (le_y - lb_y) / fh    # larger = brows higher

    # ── 5. EYE ASPECT RATIO ──────────────────────────────────────
    ear_l   = d(pt(lms, EYE_L_TOP, W, H), pt(lms, EYE_L_BOT, W, H)) / \
              (d(pt(lms, EYE_L_LEFT, W, H), pt(lms, EYE_L_RIGHT, W, H)) + 1e-6)
    ear_r   = d(pt(lms, EYE_R_TOP, W, H), pt(lms, EYE_R_BOT, W, H)) / \
              (d(pt(lms, EYE_R_LEFT, W, H), pt(lms, EYE_R_RIGHT, W, H)) + 1e-6)
    ear     = (ear_l + ear_r) / 2   # ~0.25 normal | <0.18 squint | >0.32 wide

    # ── 6. LIP CORNER PULL (smile width) ────────────────────────
    # when smiling, corners pull outward toward cheeks
    corner_spread = d(pt(lms, LIP_CORNER_L, W, H), pt(lms, LIP_CORNER_R, W, H))
    lip_width     = corner_spread / fw   # larger = wider smile

    return {
        "smile":      smile,
        "mar":        mar,
        "furrow":     furrow,
        "brow_h":     brow_h,
        "ear":        ear,
        "lip_width":  lip_width,
        "fw":         fw,
        "fh":         fh,
    }


# ════════════════════════════════════════════════════════════════
#  EMOTION CLASSIFIER
# ════════════════════════════════════════════════════════════════

def classify(sig):
    smile     = sig["smile"]
    mar       = sig["mar"]
    furrow    = sig["furrow"]
    brow_h    = sig["brow_h"]
    ear       = sig["ear"]
    lip_width = sig["lip_width"]

    # ── HAPPY ────────────────────────────────────────────────────
    # Primary: lip corners raised above midpoint
    # Secondary: wide lip spread
    happy_score  = 0.0
    if smile > 0.015:                          # very sensitive threshold
        happy_score += min(smile / 0.05, 1.0) * 0.70
    if lip_width > 0.42:                       # wide smile
        happy_score += min((lip_width - 0.42) / 0.10, 1.0) * 0.30
    happy_score = min(happy_score, 1.0)

    # ── ANGRY ────────────────────────────────────────────────────
    # Primary: brows pulled together
    # Secondary: squinting eyes + no smile
    angry_score = 0.0
    if furrow < 0.38 and smile < 0.02:        # wider threshold for furrow
        angry_score += min((0.38 - furrow) / 0.12, 1.0) * 0.70
    if ear < 0.22 and smile < 0.02:           # squinting
        angry_score += min((0.22 - ear) / 0.06, 1.0) * 0.30
    angry_score = min(angry_score, 1.0)

    # ── SAD ──────────────────────────────────────────────────────
    # Primary: lip corners dropped below midpoint
    # Secondary: droopy eyes (low EAR)
    sad_score = 0.0
    if smile < -0.005:                         # corners dropped
        sad_score += min(abs(smile + 0.005) / 0.03, 1.0) * 0.65
    if ear < 0.20 and furrow > 0.32:           # droopy eyes but not furrowed
        sad_score += min((0.20 - ear) / 0.05, 1.0) * 0.25
    if brow_h > 0.20 and furrow > 0.35:        # inner brow raise (sad brow)
        sad_score += 0.10
    sad_score = min(sad_score, 1.0)

    # ── NEUTRAL ──────────────────────────────────────────────────
    # Default when no strong signal
    neutral_score = max(0.0, 1.0 - max(happy_score, angry_score, sad_score) * 1.4)

    raw = {
        "happy":   happy_score,
        "angry":   angry_score,
        "sad":     sad_score,
        "neutral": neutral_score,
    }

    # normalise to sum = 1
    total = sum(raw.values()) or 1
    pct   = {k: round(v / total, 3) for k, v in raw.items()}

    dominant = max(pct, key=pct.get)
    return dominant, pct


# ── smoothing (weighted recent) ──────────────────────────────────
SMOOTH_N  = 6
_emo_buf  = []
_pct_buf  = []

def smooth(emo, pct):
    _emo_buf.append(emo)
    _pct_buf.append(pct)
    if len(_emo_buf) > SMOOTH_N: _emo_buf.pop(0)
    if len(_pct_buf) > SMOOTH_N: _pct_buf.pop(0)

    # smooth pct by weighted average (recent = higher weight)
    out = {k: 0.0 for k in pct}
    total_w = 0
    for i, p in enumerate(_pct_buf):
        w = i + 1
        total_w += w
        for k in out:
            out[k] += p.get(k, 0.0) * w
    out = {k: round(v / total_w, 3) for k, v in out.items()}
    dom = max(out, key=out.get)
    return dom, out


# ── distance ─────────────────────────────────────────────────────
DIST_CLOSE, DIST_FAR   = 290, 100
DIST_OK_LO, DIST_OK_HI = 140, 260

def dist_info(fw):
    if fw <= 0:                        return "no face",        (100,100,100)
    if fw > DIST_CLOSE:                return "too close",      ( 40, 40,220)
    if fw < DIST_FAR:                  return "too far",        ( 40,180,220)
    if DIST_OK_LO <= fw <= DIST_OK_HI: return "good",          ( 50,200, 80)
    if fw < DIST_OK_LO:                return "move closer",    ( 40,200,200)
    return                                    "move back",      ( 40,100,220)


# ── shared state ─────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "emotion": "neutral",
    "pct":     {"happy":0.0,"sad":0.0,"angry":0.0,"neutral":1.0},
    "face": False, "face_w": 0,
    "distance": "no face", "landmarks": None,
    "signals": {},
}
_fq, _fq_l = [None], threading.Lock()

app = Flask(__name__)
CORS(app)

@app.route("/emotion")
def api():
    with _lock:
        return jsonify({k:v for k,v in _state.items() if k!="landmarks"})

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/signals")   # debug endpoint — open in browser to see raw values
def signals():
    with _lock:
        return jsonify(_state.get("signals", {}))


# ════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ════════════════════════════════════════════════════════════════
def download_model():
    if MODEL_PATH.exists():
        return True
    print(f"  📥 Downloading face landmark model (~30MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  ✅ Model saved: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False


# ════════════════════════════════════════════════════════════════
#  ANALYSIS THREAD
# ════════════════════════════════════════════════════════════════
def analysis_loop():
    if not download_model():
        return

    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    det = mp_vision.FaceLandmarker.create_from_options(opts)
    print("  ✅ Face Landmarker loaded\n")

    last_post = 0
    ts        = 0

    while True:
        with _fq_l:
            frame = _fq[0]
        if frame is None:
            time.sleep(0.02); continue

        H, W = frame.shape[:2]

        # CLAHE contrast boost — helps in dim light
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        l     = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(l)
        enh   = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
        rgb   = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)

        ts += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = det.detect_for_video(mp_img, ts)
        except Exception:
            time.sleep(0.03); continue

        emotion = "neutral"
        pct     = {"happy":0.0,"sad":0.0,"angry":0.0,"neutral":1.0}
        face    = False
        lms     = None
        fw      = 0.0
        dlabel  = "no face"
        sigs    = {}

        if result.face_landmarks:
            face  = True
            lms   = result.face_landmarks[0]
            sigs  = extract_signals(lms, W, H) or {}

            if sigs:
                fw     = sigs["fw"]
                dlabel, _ = dist_info(fw)
                raw_emo, raw_pct = classify(sigs)
                emotion, pct     = smooth(raw_emo, raw_pct)

        with _lock:
            _state.update({
                "emotion":emotion,"pct":pct,
                "face":face,"face_w":fw,
                "distance":dlabel,"landmarks":lms,
                "signals":sigs,
            })

        # debug print — shows raw signal values every frame
        if sigs:
            print(
                f"\r  {EMOJIS.get(emotion,'?')} {emotion:<8} "
                f"smile={sigs.get('smile', 0):+.3f}  "
                f"furrow={sigs.get('furrow', 0):.3f}  "
                f"brow_h={sigs.get('brow_h', 0):.3f}  "
                f"ear={sigs.get('ear', 0):.3f}  "
                f"lip_w={sigs.get('lip_width', 0):.3f}  "
                f"dist={dlabel:<12}",
                end="", flush=True
            )
        else:
            print(f"\r  😐 no face detected — face camera directly",
                  end="", flush=True)

        now = time.time()
        if now - last_post >= 2:
            last_post = now
            try:
                requests.post(BACKEND_URL, json={
                    "emotion":emotion,"scores":pct,
                    "face_found":face,
                    "timestamp":datetime.now().isoformat(),
                }, timeout=2)
            except Exception:
                pass

        time.sleep(0.03)


# ════════════════════════════════════════════════════════════════
#  CAMERA / DISPLAY THREAD
# ════════════════════════════════════════════════════════════════
def draw_bars(frame, pct, W, H):
    order  = ["happy","sad","angry","neutral"]
    BW,BH,PAD = 160, 24, 7
    sx = W - BW - 12
    sy = 12
    for i,emo in enumerate(order):
        val = pct.get(emo, 0.0)
        col = COLORS_BGR[emo]
        bx  = sx
        by  = sy + i*(BH+PAD)
        # track
        cv2.rectangle(frame,(bx,by),(bx+BW,by+BH),(30,30,30),-1)
        # fill
        fill = max(0,int(val*BW))
        if fill>0:
            cv2.rectangle(frame,(bx,by),(bx+fill,by+BH),col,-1)
        # border
        cv2.rectangle(frame,(bx,by),(bx+BW,by+BH),(80,80,80),1)
        # label
        cv2.putText(frame,f"{EMOJIS[emo]} {emo:<7} {val*100:4.0f}%",
                    (bx-122,by+BH-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,col,1,cv2.LINE_AA)


def draw_dist_bar(frame, fw, W, H):
    bx,by = 10, H-34
    bw,bh = W-20, 16
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(25,25,25),-1)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(70,70,70),1)
    lo = int(DIST_OK_LO/DIST_CLOSE*bw)
    hi = int(DIST_OK_HI/DIST_CLOSE*bw)
    cv2.rectangle(frame,(bx+lo,by+2),(bx+hi,by+bh-2),(20,90,20),-1)
    cv2.putText(frame,"FAR", (bx+4,by-5),    cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,80,80),1)
    cv2.putText(frame,"NEAR",(bx+bw-36,by-5),cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,80,80),1)
    if fw>0:
        pos      = int(min(fw,DIST_CLOSE)/DIST_CLOSE*bw)
        lbl,col  = dist_info(fw)
        cv2.circle(frame,(bx+pos,by+bh//2),8,col,-1)
        cv2.putText(frame,f"Distance: {lbl}",
                    (bx,by-8),cv2.FONT_HERSHEY_SIMPLEX,0.44,col,1)


def camera_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("  ❌ No camera found"); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    print("  ✅ Camera ready")
    print("  📊 Debug values shown in terminal — use these to fine-tune thresholds\n")

    fc = 0
    while True:
        ret,frame = cap.read()
        if not ret:
            time.sleep(0.02); continue

        frame = cv2.flip(frame,1)
        H,W   = frame.shape[:2]
        fc   += 1

        if fc%2==0:
            with _fq_l:
                _fq[0]=frame.copy()

        with _lock:
            emotion = _state["emotion"]
            pct     = dict(_state["pct"])
            face    = _state["face"]
            lms     = _state["landmarks"]
            fw      = _state["face_w"]

        col = COLORS_BGR.get(emotion,(160,160,160))

        if lms:
            all_x=[int(lms[i].x*W) for i in range(min(468,len(lms)))]
            all_y=[int(lms[i].y*H) for i in range(min(468,len(lms)))]
            x1=max(0, min(all_x)-14)
            y1=max(0, min(all_y)-14)
            x2=min(W, max(all_x)+14)
            y2=min(H, max(all_y)+14)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            banner=f"{EMOJIS.get(emotion,'')} {emotion.upper()}  {pct.get(emotion,0)*100:.0f}%"
            cv2.rectangle(frame,(x1,y1-32),(x2,y1),col,-1)
            cv2.putText(frame,banner,(x1+6,y1-9),
                        cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,255,255),1,cv2.LINE_AA)
            # key landmark dots
            for idx in [61,291,65,295,159,386,0,17,63,293]:
                if idx<len(lms):
                    cv2.circle(frame,(int(lms[idx].x*W),int(lms[idx].y*H)),
                               2,(0,220,180),-1)
        else:
            cv2.rectangle(frame,(0,0),(W,58),(20,20,20),-1)
            cv2.putText(frame,"NO FACE — face camera, better light, move closer",
                        (10,36),cv2.FONT_HERSHEY_SIMPLEX,0.52,(60,80,255),2,cv2.LINE_AA)

        draw_bars(frame,pct,W,H)
        draw_dist_bar(frame,fw,W,H)

        cv2.imshow("DigitDetox — Cognitive Drift  (Q to quit)",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    print("\n  Done.")
    cap.release()
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__=="__main__":
    if not MP_OK:
        print("Run:  pip install mediapipe"); exit(1)

    print("╔══════════════════════════════════════════════════════╗")
    print("║   DigitDetox — Cognitive Drift Detector             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("  😄 happy   — lip corners raised")
    print("  😢 sad     — lip corners dropped / droopy eyes")
    print("  😠 angry   — brows pulled together / squinting")
    print("  😐 neutral — no strong signal\n")
    print("  📊 Terminal shows raw signal values for debugging")
    print("  🌐 Debug signals: http://localhost:5001/signals\n")

    threading.Thread(target=analysis_loop, daemon=True).start()
    threading.Thread(target=camera_loop,   daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)