import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import base64
import cv2
import librosa
import soundfile as sf
import tensorflow as tf
import pickle
import tempfile, io, json
app = FastAPI(title="Tone AI")
# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE      = 48
FACE_MODEL_DIR = "models"

# ─── CLAHE Histogram Equalization ─────────────────────────────────────────────
def apply_clahe(frame: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Enhance face contrast via CLAHE on the L channel of LAB colorspace."""
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab_eq  = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# ─── Load CNN face model ──────────────────────────────────────────────────────
face_cnn_model  = None
cnn_label_info  = None   # {"label_to_idx": ..., "idx_to_label": ..., "emotions": [...]}

def _load_cnn(path):
    """
    Try loaders in order from newest Keras API to oldest.
    The model was saved by train.py which uses tensorflow.keras (Keras 3 / 2.x).
    tf_keras (the legacy compat shim) uses an older InputLayer that rejects
    'batch_shape', so we try it LAST.
    """
    # 1. Standalone keras package (keras 3.x — matches train.py environment)
    try:
        import keras
        m = keras.models.load_model(path, compile=False)
        print("✅ CNN face model loaded via keras (standalone)")
        return m
    except Exception as e:
        print(f"   keras standalone failed: {e}")

    # 2. tf.keras (bundled with TensorFlow — usually same version as training)
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print("✅ CNN face model loaded via tf.keras")
        return m
    except Exception as e:
        print(f"   tf.keras failed: {e}")

    # 3. tf_keras legacy shim — last resort
    try:
        import tf_keras
        m = tf_keras.models.load_model(path, compile=False)
        print("✅ CNN face model loaded via tf_keras (legacy)")
        return m
    except Exception as e:
        print(f"❌ CNN face model failed to load: {e}")
        return None

face_cnn_model = _load_cnn(f"{FACE_MODEL_DIR}/face_emotion_cnn.h5")

try:
    with open(f"{FACE_MODEL_DIR}/cnn_label_encoder.pkl", "rb") as f:
        cnn_label_info = pickle.load(f)
    # Read img_size saved by train.py (96 for MobileNetV2, 48 for old CNN)
    IMG_SIZE = cnn_label_info.get("img_size", IMG_SIZE)
    print(f"✅ CNN label encoder loaded — classes: {cnn_label_info['emotions']}, img_size: {IMG_SIZE}")
except Exception as e:
    print(f"❌ CNN label encoder failed to load: {e}")

# ─── OpenCV face detector (Haar cascade — no extra deps) ─────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(frame_bgr: np.ndarray):
    """
    Detect the largest face using OpenCV Haar cascade.
    Returns:
        face_bgr : (IMG_SIZE, IMG_SIZE, 3) BGR crop
        face_box : dict with x, y, w, h in pixels
    or (None, None) if no face found.
    """
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None, None

    x, y, w, h   = max(faces, key=lambda r: r[2] * r[3])
    face_roi      = frame_bgr[y:y+h, x:x+w]
    face_resized  = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face_box      = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    return face_resized, face_box


# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Voice model ──────────────────────────────────────────────────────────────
VOICE_MODEL_PATH = "models/Emotion_Voice_Detection_Model.h5"
VOICE_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOJI_MAP = {
    "neutral":   "😐",
    "calm":      "😌",
    "happy":     "😊",
    "sad":       "😢",
    "angry":     "😠",
    "fearful":   "😨",
    "fear":      "😨",
    "disgust":   "🤢",
    "surprised": "😲",
}

voice_model = None
try:
    import tf_keras
    voice_model = tf_keras.models.load_model(VOICE_MODEL_PATH, compile=False)
    print("✅ Voice model loaded via tf_keras")
except Exception as e1:
    try:
        voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH, compile=False)
        print("✅ Voice model loaded via tf.keras")
    except Exception as e2:
        print(f"❌ Voice model failed: {e2}")


def extract_mfcc(audio_bytes: bytes) -> np.ndarray:
    """Try multiple methods to load audio and extract MFCC."""
    try:
        buf   = io.BytesIO(audio_bytes)
        X, sr = librosa.load(buf, sr=22050, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs.reshape(1, 40, 1)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        X, sr = librosa.load(tmp, sr=22050, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs.reshape(1, 40, 1)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/analyze/face")
async def analyze_face(image: str = Form(...)):
    if face_cnn_model is None or cnn_label_info is None:
        return JSONResponse({
            "success": False,
            "error": "CNN face model not loaded. Check models/face_emotion_cnn.h5 and cnn_label_encoder.pkl."
        })
    try:
        # Decode base64 → BGR frame
        if "," in image:
            image = image.split(",", 1)[1]
        img_bytes = base64.b64decode(image)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"success": False, "error": "Could not decode image"})

        # CLAHE preprocessing on full frame
        frame = apply_clahe(frame)

        # Detect and crop face
        face_bgr, face_box = detect_and_crop_face(frame)
        if face_bgr is None:
            return JSONResponse({"success": False, "error": "No face detected in image"})

        # Convert BGR → grayscale → stack to 3-channel RGB (matches training pipeline)
        face_gray  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        face_rgb   = np.stack([face_gray, face_gray, face_gray], axis=-1)  # (H, W, 3)
        face_prep  = tf.keras.applications.mobilenet_v2.preprocess_input(face_rgb)
        face_input = face_prep[np.newaxis]  # (1, H, W, 3)

        # CNN inference
        preds    = face_cnn_model.predict(face_input, verbose=0)[0]
        idx      = int(np.argmax(preds))
        idx_to_lbl = cnn_label_info["idx_to_label"]
        dominant = idx_to_lbl[idx].lower()

        emotions = {
            idx_to_lbl[i].lower(): round(float(preds[i]) * 100, 2)
            for i in range(len(preds))
        }

        return JSONResponse({
            "success":  True,
            "dominant": dominant,
            "emoji":    EMOJI_MAP.get(dominant, "🤔"),
            "emotions": emotions,
            "face_box": face_box,
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/analyze/voice")
async def analyze_voice(audio: UploadFile = File(...)):
    if voice_model is None:
        return JSONResponse({"success": False, "error": "Voice model not loaded."})
    try:
        audio_bytes = await audio.read()
        features    = extract_mfcc(audio_bytes)
        preds       = voice_model.predict(features, verbose=0)[0]
        idx         = int(np.argmax(preds))
        label       = VOICE_LABELS[idx]
        all_emotions = {
            VOICE_LABELS[i]: round(float(preds[i]) * 100, 1)
            for i in range(len(preds))
        }
        return JSONResponse({
            "success":    True,
            "dominant":   label,
            "emoji":      EMOJI_MAP.get(label, "🤔"),
            "confidence": round(float(preds[idx]) * 100, 1),
            "emotions":   all_emotions,
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/analyze/combined")
async def analyze_combined(image: str = Form(...), audio: UploadFile = File(...)):
    face_res  = await analyze_face(image=image)
    voice_res = await analyze_voice(audio=audio)
    fd = json.loads(face_res.body)
    vd = json.loads(voice_res.body)
    if fd.get("success") and vd.get("success"):
        all_labels = set(fd["emotions"]) | set(vd["emotions"])
        fused = {}
        for lbl in all_labels:
            f_score = fd["emotions"].get(lbl, 0) / 100
            v_score = vd["emotions"].get(lbl, 0) / 100
            both    = (lbl in fd["emotions"]) + (lbl in vd["emotions"])
            fused[lbl] = round((f_score + v_score) / both * 100, 1)
        dominant = max(fused, key=fused.get)
        return JSONResponse({
            "success":  True,
            "dominant": dominant,
            "emoji":    EMOJI_MAP.get(dominant, "🤔"),
            "emotions": fused,
            "face":     fd,
            "voice":    vd,
        })
    return JSONResponse({"success": False, "face": fd, "voice": vd})


app.mount("/static", StaticFiles(directory="static"), name="static")