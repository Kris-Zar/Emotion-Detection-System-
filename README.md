# MoodWave 🌊 — Multimodal Emotion Detection System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Librosa-audio-lightgrey.svg)](https://librosa.org/)
[![Docker](https://img.shields.io/badge/Docker-container-blue.svg?style=flat&logo=Docker&logoColor=white)](https://www.docker.com/)

**MoodWave** is a real-time, web-based multimodal speech and text emotion recognition system. By combining acoustic feature extraction via Convolutional Neural Networks (CNN) with real-time text sentiment correction, MoodWave detects emotional states, recommends coping actions, and dynamically fetches mood-matching music from YouTube.

🔗 **Official Web App / Live Demo:** [MoodWave on Hugging Face Spaces](https://huggingface.co/spaces/Atharv-Parth/Emotion-Detection-System)

---

## 🚀 Key Features

*   🎙️ **Real-Time Web Audio Recording:** Stream, process, and analyze voice inputs directly from the browser (up to 5 seconds of audio).
*   🧠 **Deep Learning Audio Classification:** Extracts 96 Mel Spectrogram bands and 40 MFCC bands (136 total features) using `librosa`, feeding into a custom-trained TensorFlow/Keras CNN classifier.
*   💬 **Multimodal Correction Loop:** Leverages browser-level speech-to-text (Web Speech API) to transcribe voice inputs. The backend applies a keyword-based sentiment analyzer to cross-reference and correct audio predictions when model confidence is low or in case of direct contradictions.
*   🎵 **Contextual YouTube Music Playlists:** Plays curated playlists (e.g., chill lofi, stress relief, upbeat hits) fetched live from the YouTube API (with local JSON caching to protect API quotas).
*   🧘 **Mindfulness Action Recommendations:** Displays dynamic therapeutic recommendations for each mood state (e.g., box breathing for anxiety, step-away alerts for anger, productivity pushes for neutral/balanced states).
*   🌌 **Futuristic Cyberpunk Theme:** Designed with a neon-glowing cyber-style interface using `Orbitron` and `Share Tech Mono` fonts, featuring customized wave visuals and status boards mapping to the current active emotion.

---

## ⚙️ How It Works (Pipeline)

```
[User Speech (Mic)] ──────────────────► [Web Speech API (Text)]
          │                                     │
          ▼ (Web Audio API)                     ▼
    [Audio Buffer]                     [Keyword Sentiment Analysis]
          │                                     │
          ▼ (POST Request)                      │
  [FastAPI Backend]                             │
          │                                     │
          ├─► Resample to 22.05kHz              │
          ├─► Trim silence (30dB)               │
          ├─► Extract Mel Spectrogram + MFCC    │
          └─► CNN Model Prediction ─────────────┼─► [Emotion Correction Logic]
                                                │          │
                                                │          ▼
                                                └────► [Final Emotion]
                                                           │
                                            ┌──────────────┴──────────────┐
                                            ▼                             ▼
                              [Mindfulness Recommendation]      [YouTube Music Player]
```

---

## 🛠️ Technology Stack

*   **Frontend:** HTML5, CSS3 Custom Properties (Vanilla CSS), Web Audio API, HTML5 Canvas Visualizer, Web Speech API (Speech Recognition).
*   **Backend:** Python 3.11, FastAPI, Uvicorn, TensorFlow / Keras, Librosa, NumPy, Pydantic, Google API Client.
*   **Deployment:** Docker, Hugging Face Spaces.

---

## 📁 Repository Structure

```
├── api.py                   # FastAPI backend server & ML preprocessing
├── moodwave.html            # Futuristic interactive frontend UI
├── requirements.txt         # Python dependency specification
├── Dockerfile               # Containerization configuration
├── yt_cache.json            # Local JSON cache for YouTube search queries
└── saved_model/             # Pre-trained models and preprocessing metadata
    ├── emotion_model.keras  # Saved Keras model (preferred format)
    ├── emotion_model.h5     # Legacy model format backup
    ├── encoder.pkl          # Target label categorical encoder
    └── norm.pkl             # Global dataset mean and std for feature normalization
```

---

## ⚡ Local Setup and Execution

### Prerequisites
Make sure you have python 3.11+ installed.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Emotion-Detection-System.git
   cd Emotion-Detection-System
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create an environment variable for your YouTube API key:
   ```bash
   # On Windows (CMD):
   set YOUTUBE_API_KEY=your_youtube_api_key_here
   # On Windows (PowerShell):
   $env:YOUTUBE_API_KEY="your_youtube_api_key_here"
   # On macOS/Linux:
   export YOUTUBE_API_KEY="your_youtube_api_key_here"
   ```
   *Note: If no API key is specified, the system will load songs from the local `yt_cache.json` or fallback to an empty selection safely without crashing.*

5. **Run the FastAPI Server:**
   ```bash
   uvicorn api:app --reload --host 127.0.0.1 --port 7860
   ```

6. **Access the App:**
   Open your browser and navigate to `http://127.0.0.1:7860`.

---

## 🐳 Running with Docker

You can run the application locally inside a container to replicate production environments (such as Hugging Face Spaces).

1. **Build the Docker Image:**
   ```bash
   docker build -t moodwave-app .
   ```

2. **Run the Container:**
   ```bash
   docker run -p 7860:7860 -e YOUTUBE_API_KEY="your_youtube_api_key" moodwave-app
   ```

3. **Access the Container:**
   Open your browser at `http://localhost:7860`.

---

## 🧠 Model & Preprocessing Deep-Dive

### Acoustic Preprocessing Pipeline
1.  **Resampling:** Audio signal is resampled to a consistent `22,050 Hz`.
2.  **Silence Trimming:** Silence is trimmed from the start and end of the audio using a custom energy-based threshold of `30 dB`.
3.  **Normalization:** The signal is zero-centered (`y - mean(y)`).
4.  **Duration Fix:** The audio is padded or cropped to exactly `3.0 seconds` (66,150 samples).
5.  **Feature Extraction:** 
    *   **Mel Spectrogram:** 96 bands, FFT window = 2048, hop length = 512.
    *   **MFCCs:** 40 coefficients.
    *   These features are concatenated along the feature dimension to form a matrix of shape `(136, 150)`.
6.  **Model Reshaping:** The matrix is transposed to shape `(150, 136)`, expanded to a single-channel shape `(150, 136, 1)`, normalized via parameters from `norm.pkl`, and reshaped to `(1, 150, 136, 1)` for TensorFlow execution.

### Multimodal Sentiment Correction Rule
To overcome speech emotion classification ambiguities (e.g., a sad voice reading happy words), MoodWave employs a rule-based override:
*   If the audio classification confidence is <= 35% and a strong sentiment (Happy, Angry, Sad, Fear, Disgust) is expressed in text, the text sentiment takes priority.
*   If direct contradictions are found (e.g., text sentiment is `Happy` but audio classifier predicts `Angry`/`Disgust`, or text is negative/stressed but audio predicts `Happy`), the model adjusts the output class accordingly.

---

## 🎨 Emotion Configurations

| Emotion | UI Color | YouTube Playlist Query | Therapeutic Directive |
|:---:|:---:|---|---|
| **Angry** 🔴 | `#FF3366` | "calm stress relief music" | Pause before reacting. Deep breathing. Walk away. |
| **Sad** 🔵 | `#6699FF` | "sad emotional hindi songs" | Talk to someone. Journal thoughts. Uplifting music. |
| **Happy** 🟡 | `#FFD700` | "happy upbeat party songs" | Harness energy productively. Share positivity. |
| **Fear** 🟣 | `#CC44FF` | "relaxing meditation music" | Slow breathing. Break down problems. Stay grounded. |
| **Disgust** 🟢 | `#44FF88` | "lofi chill beats" | Reset environment. Shift focus away from trigger. |
| **Neutral** 🌐 | `#00C8FF` | "focus instrumental music" | High productivity state. Deep work. Minimize distraction. |

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request if you want to add new features, fix bugs, or optimize model accuracy.

---
## 👥 Authors
 
| Role | Name |
|------|------|
| 🧠 Backend & Model Training | **Atharva Shukla** |[@Atharv Shukla](https://github.com/atharvshukla76)
| 🎨 Frontend Creation and Integration + Repo Maintainer | **Parth Saxena** |[@Parth Saxena](https://github.com/Kris-Zar)
 
### 🧠 Atharva Shukla
Responsible for the core machine learning pipeline — including data preprocessing, CNN model architecture design, training on the FER-2013 dataset, and backend inference logic.
 
### 🎨 Parth Saxena
Responsible for the frontend interface, and connecting the trained model with the detection pipeline for a seamless end-to-end experience.

