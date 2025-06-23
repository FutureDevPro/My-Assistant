import sys
import json
import time
import queue
import tempfile
import webbrowser
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import librosa
import pygame
from gtts import gTTS
from win11toast import toast

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from vosk import Model as VoskModel, KaldiRecognizer

# ==============================
# Initialization
# ==============================

# Load required models
vosk_model_path = Path(__file__).parent / "vosk-model-small-en-us-0.15"
vosk_model = VoskModel(str(vosk_model_path))
stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
stt_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
intent_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Queue for capturing audio stream
audio_queue = queue.Queue()

# ==============================
# Intent Definitions
# ==============================

# Define user intents with example phrases
INTENT_DB = {
    "Shopping": ["buy item", "order product", "purchase something", "shop online"],
    "Weather": ["weather", "forecast", "temperature", "will it rain"],
    "YouTube": ["play video", "open YouTube", "search video"],
    "Translate": ["translate phrase", "say in another language", "translate to"],
    "Location": ["my location", "where am I", "find location"],
    "Restaurants": ["find restaurants", "food nearby", "places to eat"],
    "Maps": ["directions to", "navigate to", "route to"],
    "Time": ["time now", "current time", "what time is it"],
    "News": ["news", "latest headlines", "breaking news"],
    "Quotes": ["quote", "inspirational quote", "motivation"],
    "Introduction": ["my name is", "I am", "this is"],
    "Greetings": ["hello", "hi", "good morning", "good evening"],
    "Wellbeing": ["how are you", "how do you feel", "what's up"],
    "Gratitude": ["thank you", "thanks", "appreciate it"],
    "Farewell": ["bye", "goodbye", "see you later"],
    "Examples": ["I don't know what to say", "Give me some examples"],
    "Search": ["search for", "look up", "find information about","on google"],
    "Stop": ["stop", "exit", "quit", "terminate", "shutdown"],
    "NearbyPlaces": ["nearest railway station","nearest movie theatre","nearest cinema","nearest petrol pump","nearest ATM","nearby places"]

}

# Define web actions for specific intents
ACTION_DISPATCH = {
    "Shopping": lambda q: f"https://www.amazon.in/s?k={q}",
    "Wikipedia": lambda q: f"https://en.wikipedia.org/wiki/{q.replace(' ', '_')}",
    "YouTube": lambda q: f"https://www.youtube.com/results?search_query={q}",
    "Translate": lambda q: f"https://translate.google.com/?sl=auto&tl=en&text={q}",
    "Location": lambda q: "https://www.google.com/maps/search/where+am+I/",
    "Restaurants": lambda q: "https://www.google.com/maps/search/restaurants+near+me/",
    "Weather": lambda q: f"https://www.google.com/search?q=weather+{q}",
    "Maps": lambda q: f"https://www.google.com/maps/search/{q}",
    "Time": lambda q: "https://time.is/",
    "News": lambda q: f"https://news.google.com/search?q={q}",
    "Quotes": lambda q: "https://www.brainyquote.com/quote_of_the_day",
    "Search": lambda q: f"https://www.google.com/search?q={q}",
    "NearbyPlaces": lambda q: f"https://www.google.com/maps/search/{q}",
    "Introduction": lambda q: None,
    "Greetings": lambda q: None,
    "Wellbeing": lambda q: None,
    "Gratitude": lambda q: None,
    "Farewell": lambda q: None,
    "Examples": lambda q: None,
    "Stop": lambda q: "exit"
}

# Responses for common conversational intents
PREDEFINED_RESPONSES = {
    "Greetings": "Hello! How can I assist you today?",
    "Introduction": "Nice to meet you!",
    "Wellbeing": "I'm just code, but I'm doing great! Thanks for asking.",
    "Gratitude": "You're welcome!",
    "Farewell": "Goodbye! Have a great day!"

}

# Pre-compute intent embeddings
intent_vectors = {
    intent: intent_encoder.encode(examples, convert_to_tensor=True)
    for intent, examples in INTENT_DB.items()
}

# ==============================
# Utility Functions
# ==============================

def show_toast(message, title="My Assistant"):
    try:
        toast(title, message, duration="short")
    except Exception as e:
        print(f"[!] Toast Error: {e}")

def speak(text):
    tts = gTTS(text=text)
    filename = tempfile.mktemp(suffix=".mp3")
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def record_audio(duration=4, fs=16000):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    filename = tempfile.mktemp(suffix=".wav")
    sf.write(filename, audio.squeeze(), fs)
    return filename

def transcribe_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    inputs = stt_processor(audio, sampling_rate=16000, return_tensors="pt")
    forced_ids = stt_processor.get_decoder_prompt_ids(language="en", task="transcribe")
    generated_ids = stt_model.generate(inputs.input_features, forced_decoder_ids=forced_ids)
    return stt_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def get_best_intent(transcription):
    input_vec = intent_encoder.encode(transcription, convert_to_tensor=True)
    best_intent, best_score = None, -1
    for intent, vecs in intent_vectors.items():
        score = util.pytorch_cos_sim(input_vec, vecs).max().item()
        if score > best_score:
            best_intent, best_score = intent, score
    return best_intent

def handle_command():
    print("🧠 Wake word detected. Responding...")
    speak("Hey Udit, I’m listening...")

    file = record_audio()
    transcription = transcribe_audio(file)
    print(f"📝 You said:  {transcription}")

    best_intent = get_best_intent(transcription)
    print(f"📌 Detected Intent: {best_intent}")

    response = "Sorry, I didn't understand that."

    if best_intent:
        if best_intent in PREDEFINED_RESPONSES:
            response = PREDEFINED_RESPONSES[best_intent]
            speak(response)
            show_toast(response)
        elif best_intent == "Stop":
            response = "Shutting down. Goodbye!"
            show_toast(response)
            speak(response)
            sys.exit(0)
        elif best_intent == "Examples":
            response = "Here are some things you can try saying."
            show_toast(response)
            speak(response)
            try:
                html_path = Path(__file__).parent / "examples.html"
                webbrowser.open(f"file://{html_path.resolve()}", new=2)
            except Exception as e:
                print(f"[!] Could not open examples HTML page: {e}")
        elif ACTION_DISPATCH.get(best_intent):
            response = f"Opening {best_intent} for you..."
            speak(response)
            query = transcription.replace(" ", "+")
            url = ACTION_DISPATCH[best_intent](query)
            if url:
                try:
                    webbrowser.open(url, new=2)
                except Exception as e:
                    print(f"[!] Failed to open browser: {e}")
            else:
                response = f"No action assigned for intent '{best_intent}'."
        else:
            response = f"Intent '{best_intent}' doesn't trigger any action."
            show_toast(response)
            speak(response)
    else:
        speak(response)

def callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

def listen_for_wake_word(wake_word="hello"):
    rec = KaldiRecognizer(vosk_model, 16000)
    speak("Assistant is activated")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
        print("👂 Say 'Hello' to activate.")
        while True:
            data = audio_queue.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if wake_word in text.lower():
                    print("✅ Wake word detected.")
                    handle_command()
                    print("👂 Listening again...")

# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    time.sleep(5)  # Ensure system is ready after boot
    listen_for_wake_word()
