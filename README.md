# Emotion AI — Voice + Face Detection Web App

## Project Structure
```
emotion-app/
├── main.py              ← FastAPI backend
├── requirements.txt     ← Python dependencies
├── models/
│   └── voice_emotion_model.h5   ← YOUR existing model goes here
└── static/
    └── index.html       ← Frontend UI
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your voice model
Copy your trained Keras model to:
```
models/voice_emotion_model.h5
```
Then open `main.py` and check these two settings:
```python
VOICE_MODEL_PATH = "models/voice_emotion_model.h5"
VOICE_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```
Update `VOICE_LABELS` to match your model's output classes exactly (in order).

### 3. Check MFCC input shape
The default feature extraction produces shape `(1, 40, 1)` for a CNN.
If your model expects a different shape (e.g. flat `(1, 40)` for Dense layers),
edit the `extract_mfcc()` function in `main.py`:
```python
# For Dense/LSTM models (no channel dim):
return mfcc_mean.reshape(1, n_mfcc)
```

### 4. Run the server
```bash
uvicorn main:app --reload --port 8000
```

Then open: **http://localhost:8000**

---

## How it works

| Endpoint | Input | What it does |
|---|---|---|
| `POST /analyze/face` | base64 image | DeepFace emotion detection |
| `POST /analyze/voice` | audio blob (webm) | Your Keras model via MFCC features |
| `POST /analyze/combined` | image + audio | Fuses both results (averaged probabilities) |

### Emotion fusion
The combined result averages the probability distributions from face and voice.
You can improve this by weighting one modality more than the other:
```python
# In main.py — weight face more heavily:
fused[lbl] = round((f_score * 0.6 + v_score * 0.4) * 100, 1)
```

---

## Tips
- **DeepFace** downloads model weights on first run (~100MB). This is normal.
- The frontend captures a frame every **1.5 seconds** for face analysis.
- For best voice results, record at least **2–3 seconds** of speech.
- Works on Chrome/Edge. Firefox has partial WebM audio support.
