import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ---------------- CONFIG ----------------
SR = 22050
DURATION = 3
SAMPLES = SR * DURATION
MODEL_DIR = "saved_model"

N_MELS = 96
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MAX_FRAMES = 150

app = FastAPI()

# Allow GitHub Pages and localhost to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
encoder = None
mean = None
std = None

@app.on_event("startup")
def load_resources():
    global model, encoder, mean, std
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "emotion_model.keras"))
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "norm.pkl"), "rb") as f:
        norm = pickle.load(f)
        mean = norm["mean"]
        std = norm["std"]
    print(f"[STARTUP] Model loaded. Input shape: {model.input_shape}")
    print(f"[STARTUP] Classes: {list(encoder.classes_)}")
    print(f"[STARTUP] norm mean scalar={np.isscalar(mean)}, std scalar={np.isscalar(std)}")

class AudioData(BaseModel):
    signal: List[float]
    sample_rate: int = 22050  # browser will tell us the actual sample rate

def preprocess(signal: np.ndarray, src_sr: int) -> np.ndarray:
    """Preprocess audio exactly like training code in main.ipynb."""
    signal = np.array(signal, dtype=np.float32)

    # If browser recorded at a different rate, resample to 22050
    if src_sr != SR:
        print(f"[PREPROCESS] Resampling from {src_sr} Hz to {SR} Hz ...")
        signal = librosa.resample(signal, orig_sr=src_sr, target_sr=SR)

    # Trim silence
    signal, _ = librosa.effects.trim(signal, top_db=30)

    # Mean-center (exact match to training)
    signal = signal - np.mean(signal)
    signal = np.clip(signal, -0.1, 0.1)  # match training clean_audio

    # Center crop / pad to exactly SAMPLES
    if len(signal) > SAMPLES:
        start = (len(signal) - SAMPLES) // 2
        signal = signal[start:start + SAMPLES]
    else:
        pad = SAMPLES - len(signal)
        signal = np.pad(signal, (0, pad))

    signal = np.nan_to_num(signal).astype(np.float32)

    # Mel spectrogram â†’ dB (96 bands)
    mel = librosa.feature.melspectrogram(
        y=signal, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel)  # shape: (96, time)

    # MFCCs (40 coefficients)
    mfcc = librosa.feature.mfcc(
        y=signal, sr=SR, n_mfcc=N_MFCC
    )  # shape: (40, time)

    # Concatenate â†’ (136, time)
    features = np.concatenate([mel, mfcc], axis=0)

    # Pad / crop time dimension to MAX_FRAMES=150
    if features.shape[1] > MAX_FRAMES:
        features = features[:, :MAX_FRAMES]
    else:
        pad = MAX_FRAMES - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad)))

    # Transpose â†’ (150, 136) then add channel â†’ (150, 136, 1)
    features = features.T[..., np.newaxis]

    print(f"[PREPROCESS] Final feature shape: {features.shape}")
    return features.astype(np.float32)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("moodwave.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict_emotion(data: AudioData):
    global model, encoder, mean, std

    print(f"[PREDICT] Received signal length: {len(data.signal)}, declared SR: {data.sample_rate}")

    feat = preprocess(np.array(data.signal), data.sample_rate)

    # Normalize using training stats
    feat_norm = (feat - mean) / (std + 1e-6)
    feat_norm = np.expand_dims(feat_norm, axis=0)  # (1, 150, 136, 1)

    print(f"[PREDICT] feat_norm shape: {feat_norm.shape}")

    probs = model.predict(feat_norm, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = encoder.inverse_transform([idx])[0]

    confidences = {
        encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(probs)
    }

    print(f"[PREDICT] Result: {label} ({float(probs[idx])*100:.1f}%)")

    return {
        "emotion": label,
        "confidence": float(probs[idx]),
        "all_scores": confidences
    }



