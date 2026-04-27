import os
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import pickle
import whisper
import soundfile as sf
import json
from datetime import datetime
from googleapiclient.discovery import build

# =========================================================
# YOUTUBE API KEY
# =========================================================
os.environ["YOUTUBE_API_KEY"] = "AIzaSyCO79K4HCf-vmTZ4bHw4e8fajHu0dY2rSU"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in environment variables")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


# =========================================================
# CACHE SYSTEM
# =========================================================
CACHE_FILE = "yt_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

yt_cache = load_cache()


# =========================================================
# LOAD MODEL + NORM (Bypassing SKLearn DLL Error!)
# =========================================================
model = tf.keras.models.load_model("saved_model/emotion_model.keras")

# We completely removed encoder.pkl and sklearn to prevent the Windows DLL error!
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

with open("saved_model/norm.pkl", "rb") as f:
    norm = pickle.load(f)

mean, std = norm["mean"], norm["std"]

# =========================================================
# LOAD WHISPER
# =========================================================
whisper_model = whisper.load_model("base")

# =========================================================
# EMOTION CONFIGS
# =========================================================
emotion_to_query = {
    "Angry": "calm stress relief music",
    "Sad": "sad emotional hindi songs",
    "Happy": "happy upbeat party songs",
    "Fear": "relaxing meditation music",
    "Disgust": "lofi chill beats",
    "Neutral": "focus instrumental music"
}

emotion_to_action = {
    "Angry": """You seem angry.
-> Pause before reacting.
-> Take deep breaths (inhale 4s, exhale 6s).
-> Walk away from the situation.
-> Avoid decisions right now.
-> Release energy physically.""",

    "Sad": """You seem sad.
-> Talk to someone you trust.
-> Write your thoughts down.
-> Listen to uplifting music.
-> Do one small task.
-> Avoid isolation.""",

    "Happy": """You seem happy.
-> Use this energy productively.
-> Start something meaningful.
-> Share positivity.
-> Capture the moment.
-> Avoid wasting this state.""",

    "Fear": """You seem anxious.
-> Slow your breathing.
-> Ask: is this real or imagined?
-> Break problems into small steps.
-> Stay grounded in present.
-> Avoid overthinking.""",

    "Disgust": """You seem uncomfortable.
-> Step away from the trigger.
-> Reset your environment.
-> Shift focus.
-> Avoid dwelling.
-> Give your brain time.""",

    "Neutral": """You are balanced.
-> Best time for deep work.
-> Start something important.
-> Avoid distractions.
-> Plan clearly.
-> Maintain this state."""
}

# =========================================================
# AUDIO CONFIG
# =========================================================
SR = 22050
DURATION = 3
SAMPLES = SR * DURATION

N_MELS = 96
N_MFCC = 40
N_FFT = 2048
HOP = 512
MAX_FRAMES = 150

# =========================================================
# RECORD AUDIO
# =========================================================
def record_audio():
    print("\n Recording... Speak now (3 sec)...")

    audio = sd.rec(int(DURATION * SR),
                   samplerate=SR,
                   channels=1,
                   dtype='float32')
    sd.wait()

    audio = audio.flatten()
    path = "temp.wav"
    sf.write(path, audio, SR)

    print("Recording done")
    return audio, path

# =========================================================
# SPEECH TO TEXT (FIXED - BYPASSES FFMPEG / WINERROR 4551)
# =========================================================
def speech_to_text(audio_array):
    try:
        # Whisper requires EXACTLY 16000 Hz audio
        # We resample our 22050 Hz recording down to 16000 Hz using librosa
        audio_16k = librosa.resample(audio_array, orig_sr=SR, target_sr=16000)
        
        # Passing the raw numpy array to whisper bypasses the ffmpeg.exe block!
        result = whisper_model.transcribe(audio_16k)
        text = result["text"].strip()
    except Exception as e:
        print("Whisper error:", e)
        text = ""

    print("\nTEXT:", text)
    return text

# =========================================================
# 6-CLASS TEXT SENTIMENT SYSTEM
# =========================================================
HAPPY_WORDS = {
    "happy", "joy", "joyful", "cheerful", "delighted", "pleased", "glad", "thrilled", "ecstatic", 
    "blissful", "merry", "jolly", "elated", "overjoyed", "euphoric", "radiant", "beaming", "love", 
    "adore", "cherish", "dear", "darling", "sweetheart", "beloved", "fond", "affection", "caring", 
    "tender", "warm", "heart", "hug", "kiss", "miss", "romantic", "crush", "thanks", "thank", 
    "grateful", "thankful", "appreciate", "blessed", "fortunate", "lucky", "privilege", "excited", 
    "amazing", "awesome", "incredible", "fantastic", "wonderful", "marvelous", "spectacular", 
    "magnificent", "extraordinary", "phenomenal", "outstanding", "superb", "brilliant", "excellent", 
    "fabulous", "glorious", "splendid", "wow", "yay", "hooray", "woohoo", "hurray", "beautiful", 
    "gorgeous", "pretty", "lovely", "cute", "handsome", "stunning", "elegant", "charming", "adorable", 
    "sweet", "kind", "gentle", "generous", "thoughtful", "smart", "clever", "wise", "talented", 
    "gifted", "brave", "strong", "confident", "proud", "inspiring", "success", "successful", "win", 
    "won", "winner", "victory", "achieve", "achieved", "accomplish", "accomplished", "congratulations", 
    "congrats", "celebrate", "celebration", "promotion", "graduated", "passed", "qualified", "enjoy", 
    "enjoying", "fun", "laugh", "laughing", "smile", "smiling", "giggle", "cheer", "dance", "dancing", 
    "sing", "singing", "play", "playing", "relax", "relaxing", "peaceful", "calm", "comfort", 
    "comfortable", "cozy", "safe", "secure", "heal", "healthy", "fresh", "free", "freedom", "good", 
    "great", "nice", "fine", "cool", "perfect", "best", "better", "super", "top", "ideal", "right", 
    "correct", "fair", "worthy", "valuable", "useful", "helpful", "friendly", "pleasant", "satisfying", 
    "satisfied", "content", "fulfilled", "complete", "enough", "hope", "hopeful", "optimistic", 
    "positive", "bright", "promise", "promising", "faith", "believe", "trust", "dream", "wish", 
    "desire", "forward", "progress", "improve", "improving", "growing", "bloom", "yep", "yeah", "yes", 
    "sure", "okay", "ok", "alright", "definitely", "absolutely", "certainly", "exactly", "totally", 
    "completely", "truly", "really", "honestly", "finally", "lol", "haha", "hehe"
}

ANGRY_WORDS = {
    "angry", "anger", "mad", "furious", "rage", "raging", "irritated", "irritating", "annoyed", 
    "annoying", "frustrated", "frustrating", "outraged", "livid", "fuming", "bitter", "hostile", 
    "aggressive", "violent", "explosive", "fight", "fighting", "argue", "arguing", "argument", 
    "scream", "screaming", "yell", "yelling", "shout", "shouting", "curse", "cursing", "swear", 
    "blame", "blaming", "punish", "punishment", "revenge", "threat", "threaten", "damn", "hell", 
    "shut", "stop", "enough"
}

SAD_WORDS = {
    "sad", "sadness", "unhappy", "depressed", "depression", "miserable", "gloomy", "melancholy", 
    "lonely", "loneliness", "alone", "isolated", "heartbroken", "heartbreak", "grief", "grieve", 
    "mourn", "mourning", "sorrow", "sorrowful", "cry", "crying", "tears", "weep", "weeping", "sob", 
    "hopeless", "helpless", "powerless", "weak", "boring", "bored", "tired", "exhausted", "fed", 
    "sick", "disappointed", "disappointing", "regret", "sorry", "unfortunately", "sadly"
}

FEAR_WORDS = {
    "fear", "afraid", "scared", "terrified", "frightened", "panic", "panicking", "anxious", "anxiety", 
    "nervous", "worried", "worry", "worrying", "dread", "dreading", "horror", "horrified", "shocking", 
    "shocked", "startled", "tense", "stressed", "stress", "overwhelmed", "paranoid", "nightmare", 
    "creepy", "spooky", "eerie", "coward"
}

DISGUST_WORDS = {
    "hate", "hatred", "despise", "detest", "loathe", "resent", "dislike", "disgust", "disgusting", 
    "disgusted", "repulsive", "revolting", "gross", "nasty", "vile", "toxic", "terrible", "horrible", 
    "awful", "dreadful", "atrocious", "pathetic", "worthless", "useless", "pointless", "meaningless", 
    "ugly", "stupid", "dumb", "idiot", "fool", "foolish", "lazy", "careless", "reckless", "selfish", 
    "greedy", "cruel", "mean", "rude", "harsh", "cold", "fake", "liar", "cheat", "cheater", "betrayed", 
    "betrayal", "bad", "worse", "worst", "wrong", "incorrect", "mistake", "error", "fail", "failed", 
    "failure", "loss", "lost", "lose", "problem", "trouble", "difficult", "hard", "impossible", 
    "unfair", "unjust", "corrupt", "evil", "wicked", "abuse", "abusing", "bully", "bullying", 
    "harass", "destroy", "destroying", "ruin", "ruined", "wreck", "quit", "quitting", "surrender", 
    "reject", "rejected", "rejection", "abandon", "abandoned", "ignore", "no", "nope", "never", 
    "nothing", "nobody", "nowhere", "ugh", "crap", "suck", "sucks", "trash", "garbage"
}

def get_text_sentiment(text):
    if not text:
        return "Neutral"
        
    words = set(text.lower().split())
    
    # Calculate word overlaps for each sentiment
    scores = {
        "Happy": len(words & HAPPY_WORDS),
        "Angry": len(words & ANGRY_WORDS),
        "Sad": len(words & SAD_WORDS),
        "Fear": len(words & FEAR_WORDS),
        "Disgust": len(words & DISGUST_WORDS)
    }
    
    max_score = max(scores.values())
    
    if max_score == 0:
        return "Neutral"
        
    # Find which emotion got the highest score
    for emotion, score in scores.items():
        if score == max_score:
            return emotion

def correct_emotion(audio_emotion, text_sentiment, confidence):
    # Only correct when model confidence is low
    if confidence > 0.50:
        return audio_emotion

    text_sentiment = text_sentiment.capitalize()

    if text_sentiment == "Happy":
        if audio_emotion in ["Angry", "Sad", "Fear", "Disgust", "Neutral"]:
            return "Happy"

    if text_sentiment == "Angry":
        if audio_emotion in ["Happy", "Neutral"]:
            return "Angry"
        if audio_emotion in ["Sad", "Fear", "Disgust"]:
            return audio_emotion  

    if text_sentiment == "Sad":
        if audio_emotion in ["Happy", "Neutral"]:
            return "Sad"
        if audio_emotion in ["Angry", "Fear", "Disgust"]:
            return audio_emotion  

    if text_sentiment == "Fear":
        if audio_emotion in ["Happy", "Neutral"]:
            return "Fear"
        if audio_emotion in ["Angry", "Sad", "Disgust"]:
            return audio_emotion

    if text_sentiment == "Disgust":
        if audio_emotion in ["Happy", "Neutral"]:
            return "Disgust"
        if audio_emotion in ["Angry", "Sad", "Fear"]:
            return audio_emotion

    if text_sentiment == "Neutral":
        if audio_emotion in ["Happy", "Angry", "Sad"]:
            return "Neutral"
        if audio_emotion in ["Fear", "Disgust"]:
            return audio_emotion 

    return audio_emotion

# =========================================================
# FEATURE EXTRACTION (FIXED - NO SKLEARN / NO DLL ERRORS)
# =========================================================
def custom_trim(y, top_db=30):
    """Pure Numpy replacement for librosa.effects.trim to avoid sklearn DLL errors"""
    if len(y) == 0:
        return y
    
    # Calculate threshold based on top_db
    ref_power = np.max(np.abs(y))
    if ref_power == 0:
        return y
        
    threshold = ref_power / (10.0 ** (top_db / 20.0))
    
    # Find indices where amplitude > threshold
    non_silent_indices = np.where(np.abs(y) > threshold)[0]
    
    if len(non_silent_indices) > 0:
        start_idx = non_silent_indices[0]
        end_idx = non_silent_indices[-1]
        return y[start_idx:end_idx+1]
    else:
        return y

def preprocess(signal):
    # USE CUSTOM TRIM INSTEAD OF LIBROSA!
    signal = custom_trim(signal, top_db=30)
    signal = signal - np.mean(signal)

    if len(signal) > SAMPLES:
        start = (len(signal) - SAMPLES) // 2
        signal = signal[start:start + SAMPLES]
    else:
        signal = np.pad(signal, (0, SAMPLES - len(signal)))

    signal = np.nan_to_num(signal).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=signal, sr=SR, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)

    feat = np.concatenate([mel, mfcc], axis=0)

    if feat.shape[1] > MAX_FRAMES:
        feat = feat[:, :MAX_FRAMES]
    else:
        feat = np.pad(feat, ((0, 0), (0, MAX_FRAMES - feat.shape[1])))

    feat = feat.T
    feat = feat[..., np.newaxis]

    return feat.astype(np.float32)

# =========================================================
# YOUTUBE SONG FETCH
# =========================================================
MAX_API_CALLS = 5
api_call_count = 0

def get_songs(emotion, text=""):

    global api_call_count

    try:
        base_query = emotion_to_query.get(emotion, "lofi music")
        query = base_query

        if query in yt_cache:
            return yt_cache[query]

        if api_call_count >= MAX_API_CALLS:
            print("API disabled (quota protection)")
            return yt_cache.get(query, [{
                "title": "Offline fallback",
                "url": "https://youtube.com"
            }])

        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=5
        )

        response = request.execute()
        api_call_count += 1

        songs = []

        for item in response["items"]:
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            url = f"https://www.youtube.com/watch?v={video_id}"

            songs.append({
                "title": title,
                "url": url
            })

        yt_cache[query] = songs
        save_cache(yt_cache)

        return songs

    except Exception as e:
        print("YouTube error:", e)

        return yt_cache.get(query, [{
            "title": "Offline fallback",
            "url": "https://youtube.com"
        }])

# =========================================================
# SAVE HISTORY
# =========================================================
def save_history(data):

    if os.path.exists("history.json"):
        with open("history.json", "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(data)

    with open("history.json", "w") as f:
        json.dump(history, f, indent=4)

# =========================================================
# MAIN PIPELINE
# =========================================================
def predict():

    audio, path = record_audio()

    feat = preprocess(audio)
    feat = np.expand_dims(feat, axis=0)
    feat = (feat - mean) / (std + 1e-6)

    probs = model.predict(feat, verbose=0)[0]
    idx = np.argmax(probs)

    # Use our hardcoded classes to bypass sklearn DLL errors!
    audio_emotion = CLASSES[idx]
    confidence = float(probs[idx])

    # Pass the raw audio array directly instead of the file path!
    text = speech_to_text(audio)
    text_sentiment = get_text_sentiment(text)
    emotion = correct_emotion(audio_emotion, text_sentiment, confidence)

    songs = get_songs(emotion, text)
    action = emotion_to_action.get(emotion, "Take a break.")

    result = {
        "time": str(datetime.now()),
        "emotion": str(emotion),
        "audio_emotion": str(audio_emotion),
        "text_sentiment": text_sentiment,
        "confidence": round(confidence, 3),
        "text": text,
        "songs": songs,
        "action": action
    }

    save_history(result)

    print("\nFINAL RESULT")
    print("---------------------------")
    print(f"Time          : {result['time']}")
    print(f"Audio Emotion : {result['audio_emotion']}")
    print(f"Text Sentiment: {result['text_sentiment']}")
    print(f"Final Emotion : {result['emotion']}")
    if result['emotion'] != result['audio_emotion']:
        print("(Corrected by text sentiment)")
    print(f"Text          : {result['text']}")
    print(f"Confidence    : {result['confidence']}")
    print("\nRecommended Songs:")
    for i, song in enumerate(result["songs"], 1):
        print(f"{i}. {song['title']}")
        print(f"   {song['url']}")

    print("\nWhat you should do:")
    print(result["action"])

# =========================================================
# RUN
# =========================================================
predict()