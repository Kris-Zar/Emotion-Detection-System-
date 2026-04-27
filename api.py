import os, json, pickle, traceback
import numpy as np
import librosa
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

SR = 22050
DURATION = 3
SAMPLES = SR * DURATION
MODEL_DIR = "saved_model"
N_MELS, N_MFCC, N_FFT, HOP, MAX_FRAMES = 96, 40, 2048, 512, 150
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
EMOTION_TO_QUERY = {"Angry":"calm stress relief music","Sad":"sad emotional hindi songs","Happy":"happy upbeat party songs","Fear":"relaxing meditation music","Disgust":"lofi chill beats","Neutral":"focus instrumental music"}
EMOTION_TO_ACTION = {"Angry":"You seem angry.\n-> Pause before reacting.\n-> Take deep breaths.\n-> Walk away.\n-> Avoid decisions right now.","Sad":"You seem sad.\n-> Talk to someone you trust.\n-> Write your thoughts down.\n-> Listen to uplifting music.","Happy":"You seem happy.\n-> Use this energy productively.\n-> Start something meaningful.\n-> Share positivity.","Fear":"You seem anxious.\n-> Slow your breathing.\n-> Break problems into small steps.\n-> Stay grounded in present.","Disgust":"You seem uncomfortable.\n-> Step away from the trigger.\n-> Reset your environment.\n-> Shift focus.","Neutral":"You are balanced.\n-> Best time for deep work.\n-> Start something important.\n-> Avoid distractions."}

CACHE_FILE = "yt_cache.json"
yt_cache = {}
try:
    if os.path.exists(CACHE_FILE):
        yt_cache = json.load(open(CACHE_FILE))
except: pass

api_call_count = 0

def _fetch_youtube(emotion):
    global api_call_count
    query = EMOTION_TO_QUERY.get(emotion, "lofi music")
    if query in yt_cache:
        return yt_cache[query]
    if not YOUTUBE_API_KEY or api_call_count >= 15:
        return []
    from googleapiclient.discovery import build
    yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
    resp = yt.search().list(q=query, part="snippet", type="video", maxResults=3).execute()
    api_call_count += 1
    songs = [{"title": i["snippet"]["title"], "url": f"https://www.youtube.com/watch?v={i['id']['videoId']}"} for i in resp.get("items",[])]
    yt_cache[query] = songs
    try: json.dump(yt_cache, open(CACHE_FILE,"w"), indent=2)
    except: pass
    return songs

def get_songs(emotion):
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_fetch_youtube, emotion)
            return future.result(timeout=6)
    except Exception as e:
        print(f"[YOUTUBE] {type(e).__name__}: {e}")
        return []

HAPPY_WORDS={"happy","joy","joyful","cheerful","delighted","pleased","glad","thrilled","ecstatic","blissful","merry","elated","overjoyed","euphoric","love","adore","cherish","dear","darling","beloved","fond","warm","heart","hug","kiss","thanks","thank","grateful","thankful","blessed","fortunate","lucky","excited","amazing","awesome","incredible","fantastic","wonderful","marvelous","spectacular","brilliant","excellent","fabulous","wow","yay","hooray","beautiful","gorgeous","lovely","cute","adorable","sweet","kind","gentle","generous","brave","strong","confident","proud","success","win","won","victory","celebrate","enjoy","fun","laugh","smile","peaceful","calm","safe","secure","healthy","good","great","nice","perfect","best","better","hope","hopeful","yes","okay","lol","haha"}
ANGRY_WORDS={"angry","anger","mad","furious","rage","irritated","annoyed","frustrated","outraged","livid","fuming","bitter","hostile","aggressive","violent","fight","argue","scream","yell","shout","curse","blame","revenge","threat","damn","hell","stop"}
SAD_WORDS={"sad","sadness","unhappy","depressed","miserable","gloomy","melancholy","lonely","alone","isolated","heartbroken","grief","mourn","sorrow","cry","crying","tears","weep","sob","hopeless","helpless","weak","bored","tired","exhausted","disappointed","regret","sorry","unfortunately","sadly"}
FEAR_WORDS={"fear","afraid","scared","terrified","frightened","panic","anxious","anxiety","nervous","worried","worry","dread","horror","horrified","shocked","tense","stressed","stress","overwhelmed","paranoid","nightmare","creepy","spooky"}
DISGUST_WORDS={"hate","hatred","despise","loathe","resent","dislike","disgust","disgusting","repulsive","revolting","gross","nasty","vile","toxic","terrible","horrible","awful","pathetic","worthless","useless","ugly","stupid","dumb","idiot","fool","lazy","cruel","mean","rude","fake","liar","cheat","betrayed","bad","worse","worst","wrong","fail","failed","failure","impossible","unfair","evil","bully","destroy","ruin","reject","abandon","ugh","crap","suck","sucks","trash","garbage"}

def get_text_sentiment(text):
    if not text: return "Neutral"
    words = set(text.lower().split())
    scores = {"Happy":len(words&HAPPY_WORDS),"Angry":len(words&ANGRY_WORDS),"Sad":len(words&SAD_WORDS),"Fear":len(words&FEAR_WORDS),"Disgust":len(words&DISGUST_WORDS)}
    m = max(scores.values())
    return "Neutral" if m == 0 else max(scores, key=scores.get)

def correct_emotion(audio_emotion, text_sentiment, confidence):
    if confidence > 0.50: return audio_emotion
    ts = text_sentiment.capitalize()
    mapping = {"Happy":["Angry","Sad","Fear","Disgust","Neutral"],"Angry":["Happy","Neutral"],"Sad":["Happy","Neutral"],"Fear":["Happy","Neutral"],"Disgust":["Happy","Neutral"],"Neutral":["Happy","Angry","Sad"]}
    if ts in mapping and audio_emotion in mapping[ts]: return ts
    return audio_emotion

def custom_trim(y, top_db=30):
    if len(y) == 0: return y
    ref = np.max(np.abs(y))
    if ref == 0: return y
    ns = np.where(np.abs(y) > ref/(10.0**(top_db/20.0)))[0]
    return y[ns[0]:ns[-1]+1] if len(ns) > 0 else y

def preprocess(signal, src_sr):
    signal = np.array(signal, dtype=np.float32)
    if src_sr != SR:
        signal = librosa.resample(signal, orig_sr=src_sr, target_sr=SR)
    signal = custom_trim(signal, top_db=30)
    signal = signal - np.mean(signal)
    if len(signal) > SAMPLES:
        start = (len(signal)-SAMPLES)//2
        signal = signal[start:start+SAMPLES]
    else:
        signal = np.pad(signal, (0, SAMPLES-len(signal)))
    signal = np.nan_to_num(signal).astype(np.float32)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=signal,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS))
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    feat = np.concatenate([mel, mfcc], axis=0)
    feat = feat[:, :MAX_FRAMES] if feat.shape[1]>MAX_FRAMES else np.pad(feat,((0,0),(0,MAX_FRAMES-feat.shape[1])))
    return feat.T[..., np.newaxis].astype(np.float32)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])
model = mean = std = None

@app.on_event("startup")
def load_resources():
    global model, mean, std
    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR,"emotion_model.keras"))
        norm = pickle.load(open(os.path.join(MODEL_DIR,"norm.pkl"),"rb"))
        mean, std = norm["mean"], norm["std"]
        print(f"[STARTUP] OK. Classes: {CLASSES}")
    except Exception as e:
        print(f"[STARTUP ERROR] {e}")

class AudioData(BaseModel):
    signal: List[float]
    sample_rate: int = 22050
    text: Optional[str] = ""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return open("moodwave.html","r",encoding="utf-8").read()

@app.get("/health")
async def health_check():
    return {"status":"ok","model_loaded": model is not None}

@app.post("/predict")
async def predict_emotion(data: AudioData):
    audio_emotion, confidence, confidences, final_emotion, text_sentiment = "Neutral", 0.0, {}, "Neutral", "Neutral"
    try:
        if model is None:
            raise RuntimeError("Model not loaded")
        feat = preprocess(np.array(data.signal), data.sample_rate)
        feat_norm = np.expand_dims((feat - mean)/(std + 1e-6), axis=0)
        probs = model.predict(feat_norm, verbose=0)[0]
        idx = int(np.argmax(probs))
        audio_emotion = CLASSES[idx]
        confidence = float(probs[idx])
        confidences = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
        print(f"[PREDICT] audio={audio_emotion} conf={confidence:.2f}")
    except Exception as e:
        print(f"[PREDICT MODEL ERROR] {e}"); traceback.print_exc()

    try:
        text_sentiment = get_text_sentiment(data.text or "")
        final_emotion = correct_emotion(audio_emotion, text_sentiment, confidence)
    except Exception as e:
        print(f"[PREDICT SENTIMENT ERROR] {e}")
        final_emotion = audio_emotion

    songs = get_songs(final_emotion)
    action = EMOTION_TO_ACTION.get(final_emotion, "Take a break.")

    return {
        "emotion": final_emotion,
        "audio_emotion": audio_emotion,
        "text_sentiment": text_sentiment,
        "confidence": confidence,
        "all_scores": confidences,
        "songs": songs,
        "action": action
    }
