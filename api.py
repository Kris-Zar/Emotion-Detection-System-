import os
import json
import pickle
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

SR = 22050
DURATION = 3
SAMPLES = SR * DURATION
MODEL_DIR = "saved_model"
N_MELS = 96
N_MFCC = 40
N_FFT = 2048
HOP = 512
MAX_FRAMES = 150
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
CACHE_FILE = "yt_cache.json"
api_call_count = 0
MAX_API_CALLS = 20

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except:
        pass

yt_cache = load_cache()

EMOTION_TO_QUERY = {
    "Angry": "calm stress relief music",
    "Sad": "sad emotional hindi songs",
    "Happy": "happy upbeat party songs",
    "Fear": "relaxing meditation music",
    "Disgust": "lofi chill beats",
    "Neutral": "focus instrumental music"
}

EMOTION_TO_ACTION = {
    "Angry": "You seem angry.\n-> Pause before reacting.\n-> Take deep breaths (inhale 4s, exhale 6s).\n-> Walk away from the situation.\n-> Avoid decisions right now.\n-> Release energy physically.",
    "Sad": "You seem sad.\n-> Talk to someone you trust.\n-> Write your thoughts down.\n-> Listen to uplifting music.\n-> Do one small task.\n-> Avoid isolation.",
    "Happy": "You seem happy.\n-> Use this energy productively.\n-> Start something meaningful.\n-> Share positivity.\n-> Capture the moment.\n-> Avoid wasting this state.",
    "Fear": "You seem anxious.\n-> Slow your breathing.\n-> Ask: is this real or imagined?\n-> Break problems into small steps.\n-> Stay grounded in present.\n-> Avoid overthinking.",
    "Disgust": "You seem uncomfortable.\n-> Step away from the trigger.\n-> Reset your environment.\n-> Shift focus.\n-> Avoid dwelling.\n-> Give your brain time.",
    "Neutral": "You are balanced.\n-> Best time for deep work.\n-> Start something important.\n-> Avoid distractions.\n-> Plan clearly.\n-> Maintain this state."
}

def get_songs(emotion):
    global api_call_count
    try:
        query = EMOTION_TO_QUERY.get(emotion, "lofi music")
        if query in yt_cache:
            return yt_cache[query]
        if not YOUTUBE_API_KEY or api_call_count >= MAX_API_CALLS:
            return yt_cache.get(query, [])
        import socket
        socket.setdefaulttimeout(8)
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        response = youtube.search().list(q=query, part="snippet", type="video", maxResults=3).execute()
        api_call_count += 1
        songs = [{"title": item["snippet"]["title"], "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"} for item in response.get("items", [])]
        yt_cache[query] = songs
        save_cache(yt_cache)
        return songs
    except Exception as e:
        print(f"[YOUTUBE ERROR] {e}")
        return []

HAPPY_WORDS = {"happy","joy","joyful","cheerful","delighted","pleased","glad","thrilled","ecstatic","blissful","merry","jolly","elated","overjoyed","euphoric","radiant","beaming","love","adore","cherish","dear","darling","sweetheart","beloved","fond","affection","caring","tender","warm","heart","hug","kiss","miss","romantic","crush","thanks","thank","grateful","thankful","appreciate","blessed","fortunate","lucky","excited","amazing","awesome","incredible","fantastic","wonderful","marvelous","spectacular","magnificent","extraordinary","phenomenal","outstanding","superb","brilliant","excellent","fabulous","glorious","splendid","wow","yay","hooray","beautiful","gorgeous","pretty","lovely","cute","handsome","stunning","elegant","charming","adorable","sweet","kind","gentle","generous","thoughtful","smart","brave","strong","confident","proud","success","win","won","winner","victory","achieve","accomplished","congratulations","celebrate","enjoy","fun","laugh","smile","peaceful","calm","safe","secure","healthy","good","great","nice","perfect","best","better","hope","hopeful","optimistic","yes","okay","alright","lol","haha"}
ANGRY_WORDS = {"angry","anger","mad","furious","rage","raging","irritated","annoyed","frustrated","outraged","livid","fuming","bitter","hostile","aggressive","violent","fight","argue","argument","scream","yell","shout","curse","blame","revenge","threat","damn","hell","stop","enough"}
SAD_WORDS = {"sad","sadness","unhappy","depressed","depression","miserable","gloomy","melancholy","lonely","loneliness","alone","isolated","heartbroken","grief","grieve","mourn","sorrow","cry","crying","tears","weep","sob","hopeless","helpless","weak","boring","bored","tired","exhausted","disappointed","regret","sorry","unfortunately","sadly"}
FEAR_WORDS = {"fear","afraid","scared","terrified","frightened","panic","panicking","anxious","anxiety","nervous","worried","worry","dread","horror","horrified","shocked","tense","stressed","stress","overwhelmed","paranoid","nightmare","creepy","spooky"}
DISGUST_WORDS = {"hate","hatred","despise","detest","loathe","resent","dislike","disgust","disgusting","repulsive","revolting","gross","nasty","vile","toxic","terrible","horrible","awful","dreadful","pathetic","worthless","useless","ugly","stupid","dumb","idiot","fool","lazy","cruel","mean","rude","fake","liar","cheat","betrayed","bad","worse","worst","wrong","fail","failed","failure","impossible","unfair","evil","bully","harass","destroy","ruin","reject","abandon","ugh","crap","suck","sucks","trash","garbage"}

def get_text_sentiment(text):
    if not text:
        return "Neutral"
    words = set(text.lower().split())
    scores = {"Happy": len(words & HAPPY_WORDS), "Angry": len(words & ANGRY_WORDS), "Sad": len(words & SAD_WORDS), "Fear": len(words & FEAR_WORDS), "Disgust": len(words & DISGUST_WORDS)}
    max_score = max(scores.values())
    if max_score == 0:
        return "Neutral"
    return max(scores, key=scores.get)

def correct_emotion(audio_emotion, text_sentiment, confidence):
    if confidence > 0.50:
        return audio_emotion
    ts = text_sentiment.capitalize()
    if ts == "Happy" and audio_emotion in ["Angry","Sad","Fear","Disgust","Neutral"]:
        return "Happy"
    if ts == "Angry" and audio_emotion in ["Happy","Neutral"]:
        return "Angry"
    if ts == "Sad" and audio_emotion in ["Happy","Neutral"]:
        return "Sad"
    if ts == "Fear" and audio_emotion in ["Happy","Neutral"]:
        return "Fear"
    if ts == "Disgust" and audio_emotion in ["Happy","Neutral"]:
        return "Disgust"
    if ts == "Neutral" and audio_emotion in ["Happy","Angry","Sad"]:
        return "Neutral"
    return audio_emotion

def custom_trim(y, top_db=30):
    if len(y) == 0:
        return y
    ref_power = np.max(np.abs(y))
    if ref_power == 0:
        return y
    threshold = ref_power / (10.0 ** (top_db / 20.0))
    non_silent = np.where(np.abs(y) > threshold)[0]
    return y[non_silent[0]:non_silent[-1]+1] if len(non_silent) > 0 else y

def preprocess(signal, src_sr):
    signal = np.array(signal, dtype=np.float32)
    if src_sr != SR:
        signal = librosa.resample(signal, orig_sr=src_sr, target_sr=SR)
    signal = custom_trim(signal, top_db=30)
    signal = signal - np.mean(signal)
    if len(signal) > SAMPLES:
        start = (len(signal) - SAMPLES) // 2
        signal = signal[start:start+SAMPLES]
    else:
        signal = np.pad(signal, (0, SAMPLES - len(signal)))
    signal = np.nan_to_num(signal).astype(np.float32)
    mel = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    feat = np.concatenate([mel, mfcc], axis=0)
    if feat.shape[1] > MAX_FRAMES:
        feat = feat[:, :MAX_FRAMES]
    else:
        feat = np.pad(feat, ((0,0),(0, MAX_FRAMES - feat.shape[1])))
    return feat.T[..., np.newaxis].astype(np.float32)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])

model = None
mean = None
std = None

@app.on_event("startup")
def load_resources():
    global model, mean, std
    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "emotion_model.keras"))
        with open(os.path.join(MODEL_DIR, "norm.pkl"), "rb") as f:
            norm = pickle.load(f)
        mean = norm["mean"]
        std = norm["std"]
        print(f"[STARTUP] Model loaded. Classes: {CLASSES}")
    except Exception as e:
        print(f"[STARTUP ERROR] {e}")

class AudioData(BaseModel):
    signal: List[float]
    sample_rate: int = 22050
    text: Optional[str] = ""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("moodwave.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict_emotion(data: AudioData):
    try:
        if model is None:
            return JSONResponse(status_code=200, content={"error": "Model not loaded yet"})
        feat = preprocess(np.array(data.signal), data.sample_rate)
        feat_norm = (feat - mean) / (std + 1e-6)
        feat_norm = np.expand_dims(feat_norm, axis=0)
        probs = model.predict(feat_norm, verbose=0)[0]
        idx = int(np.argmax(probs))
        audio_emotion = CLASSES[idx]
        confidence = float(probs[idx])
        text_sentiment = get_text_sentiment(data.text or "")
        final_emotion = correct_emotion(audio_emotion, text_sentiment, confidence)
        confidences = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
        songs = get_songs(final_emotion)
        action = EMOTION_TO_ACTION.get(final_emotion, "Take a break.")
        print(f"[PREDICT] audio={audio_emotion} final={final_emotion} conf={confidence:.2f}")
        return {"emotion": final_emotion, "audio_emotion": audio_emotion, "text_sentiment": text_sentiment, "confidence": confidence, "all_scores": confidences, "songs": songs, "action": action}
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=200, content={"error": str(e), "emotion": "Neutral", "audio_emotion": "Neutral", "text_sentiment": "Neutral", "confidence": 0.0, "all_scores": {}, "songs": [], "action": EMOTION_TO_ACTION["Neutral"]})
