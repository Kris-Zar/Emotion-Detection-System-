import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

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

# ---------------- YOUTUBE CONFIG ----------------
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

EMOTION_TO_QUERY = {
    "Angry":   "calm stress relief music",
    "Sad":     "sad emotional hindi songs",
    "Happy":   "happy upbeat party songs",
    "Fear":    "relaxing meditation music",
    "Disgust": "lofi chill beats",
    "Neutral": "focus instrumental music"
}

def get_songs(emotion: str):
    if not YOUTUBE_API_KEY:
        print("[YOUTUBE] No API key set, skipping.")
        return []
    try:
        import socket
        socket.setdefaulttimeout(8)  # 8 second timeout on network calls
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        query = EMOTION_TO_QUERY.get(emotion, "lofi music")
        request = youtube.search().list(
            q=query, part="snippet", type="video", maxResults=3
        )
        response = request.execute()
        songs = []
        for item in response.get("items", []):
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            songs.append({"title": title, "url": f"https://www.youtube.com/watch?v={video_id}"})
        print(f"[YOUTUBE] Fetched {len(songs)} songs for {emotion}")
        return songs
    except Exception as e:
        print(f"[YOUTUBE ERROR] {type(e).__name__}: {e}")
        return []

# ---------------- TEXT SENTIMENT ----------------
POSITIVE_WORDS = {
    "happy","joy","joyful","cheerful","delighted","pleased","glad","thrilled","ecstatic",
    "blissful","merry","jolly","elated","overjoyed","euphoric","radiant","beaming",
    "love","adore","cherish","dear","darling","sweetheart","beloved","fond","affection",
    "caring","tender","warm","heart","hug","kiss","miss","romantic","crush",
    "thanks","thank","grateful","thankful","appreciate","blessed","fortunate","lucky",
    "excited","amazing","awesome","incredible","fantastic","wonderful","marvelous",
    "spectacular","magnificent","extraordinary","phenomenal","outstanding","superb",
    "brilliant","excellent","fabulous","glorious","splendid","wow","yay","hooray",
    "beautiful","gorgeous","pretty","lovely","cute","handsome","stunning","elegant",
    "charming","adorable","sweet","kind","gentle","generous","thoughtful","smart",
    "brave","strong","confident","proud","inspiring","success","successful","win",
    "won","winner","victory","achieve","achieved","accomplish","accomplished",
    "congratulations","celebrate","enjoy","fun","laugh","smile","giggle","cheer",
    "dance","sing","play","relax","peaceful","calm","comfort","comfortable","cozy",
    "safe","secure","healthy","fresh","free","freedom","good","great","nice","fine",
    "cool","perfect","best","better","super","top","ideal","right","helpful",
    "friendly","pleasant","satisfied","content","fulfilled","hope","hopeful",
    "optimistic","positive","bright","promise","faith","believe","trust","dream",
    "yes","sure","okay","ok","alright","definitely","absolutely","totally","lol","haha"
}

NEGATIVE_WORDS = {
    "angry","anger","mad","furious","rage","raging","irritated","annoyed","frustrated",
    "outraged","livid","fuming","bitter","hostile","aggressive","violent",
    "hate","hatred","despise","detest","loathe","resent","dislike","disgust",
    "disgusting","disgusted","repulsive","revolting","gross","nasty","vile","toxic",
    "sad","sadness","unhappy","depressed","depression","miserable","gloomy","melancholy",
    "lonely","loneliness","alone","isolated","heartbroken","grief","grieve","mourn",
    "sorrow","sorrowful","cry","crying","tears","weep","weeping","sob",
    "fear","afraid","scared","terrified","frightened","panic","panicking","anxious",
    "anxiety","nervous","worried","worry","dread","horror","horrified","shocked",
    "tense","stressed","stress","overwhelmed","paranoid","nightmare",
    "pain","painful","hurt","hurting","suffering","suffer","agony","ache","broken",
    "sick","illness","disease","dying","death","dead","die","kill","murder","suicide",
    "terrible","horrible","awful","dreadful","pathetic","worthless","useless",
    "hopeless","helpless","powerless","weak","ugly","stupid","dumb","idiot","fool",
    "lazy","careless","selfish","cruel","mean","rude","harsh","fake","liar","cheat",
    "betrayed","betrayal","bad","worse","worst","wrong","mistake","fail","failed",
    "failure","loss","lost","problem","trouble","impossible","unfair","unjust","evil",
    "fight","argue","argument","scream","yell","shout","curse","blame","punish",
    "revenge","threat","abuse","bully","harass","destroy","ruin","quit","reject",
    "rejected","abandon","abandoned","damn","shut","ugh","crap","suck","sucks",
    "trash","garbage","boring","bored","tired","exhausted","disappointed","regret",
    "sorry","unfortunately","sadly","cannot","dont","wont"
}

def get_text_sentiment(text: str) -> str:
    if not text:
        return "neutral"
    words = set(text.lower().split())
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"

def correct_emotion(audio_emotion: str, text_sentiment: str, confidence: float) -> str:
    if confidence > 0.50:
        return audio_emotion
    if text_sentiment == "positive":
        if audio_emotion in ["Angry", "Sad", "Fear", "Disgust", "Neutral"]:
            return "Happy"
    if text_sentiment == "negative":
        if audio_emotion == "Happy":
            return "Angry"
        if audio_emotion == "Neutral":
            return "Sad"
    if text_sentiment == "neutral":
        if audio_emotion in ["Happy", "Angry", "Sad"]:
            return "Neutral"
    return audio_emotion

# ---------------- APP SETUP ----------------
app = FastAPI()

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
    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "emotion_model.keras"))
        with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "norm.pkl"), "rb") as f:
            norm = pickle.load(f)
            mean = norm["mean"]
            std = norm["std"]
        print(f"[STARTUP] Model loaded. Classes: {list(encoder.classes_)}")
    except Exception as e:
        print(f"[STARTUP ERROR] {e}")

class AudioData(BaseModel):
    signal: List[float]
    sample_rate: int = 22050
    text: Optional[str] = ""

def preprocess(signal: np.ndarray, src_sr: int) -> np.ndarray:
    signal = np.array(signal, dtype=np.float32)

    if src_sr != SR:
        signal = librosa.resample(signal, orig_sr=src_sr, target_sr=SR)

    signal, _ = librosa.effects.trim(signal, top_db=30)
    signal = signal - np.mean(signal)

    if len(signal) > SAMPLES:
        start = (len(signal) - SAMPLES) // 2
        signal = signal[start:start + SAMPLES]
    else:
        signal = np.pad(signal, (0, SAMPLES - len(signal)))

    signal = np.nan_to_num(signal).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=signal, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    features = np.concatenate([mel, mfcc], axis=0)

    if features.shape[1] > MAX_FRAMES:
        features = features[:, :MAX_FRAMES]
    else:
        features = np.pad(features, ((0, 0), (0, MAX_FRAMES - features.shape[1])))

    features = features.T[..., np.newaxis]
    print(f"[PREPROCESS] shape: {features.shape}")
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
    # Top-level try/except so we NEVER return a 500
    try:
        global model, encoder, mean, std

        if model is None:
            return JSONResponse(status_code=200, content={"error": "Model not loaded"})

        print(f"[PREDICT] samples={len(data.signal)}, sr={data.sample_rate}, text='{data.text}'")

        feat = preprocess(np.array(data.signal), data.sample_rate)
        feat_norm = (feat - mean) / (std + 1e-6)
        feat_norm = np.expand_dims(feat_norm, axis=0)

        probs = model.predict(feat_norm, verbose=0)[0]
        idx = int(np.argmax(probs))
        audio_emotion = encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])

        text_sentiment = get_text_sentiment(data.text or "")
        final_emotion = correct_emotion(audio_emotion, text_sentiment, confidence)

        confidences = {
            encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        }

        songs = get_songs(final_emotion)

        print(f"[PREDICT] audio={audio_emotion}, final={final_emotion}, conf={confidence:.2f}")

        return {
            "emotion": final_emotion,
            "audio_emotion": audio_emotion,
            "text_sentiment": text_sentiment,
            "confidence": confidence,
            "all_scores": confidences,
            "songs": songs
        }

    except Exception as e:
        print(f"[PREDICT ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Return 200 with error so frontend shows something useful
        return JSONResponse(status_code=200, content={
            "error": str(e),
            "emotion": "Neutral",
            "audio_emotion": "Neutral",
            "text_sentiment": "neutral",
            "confidence": 0.0,
            "all_scores": {},
            "songs": []
        })
