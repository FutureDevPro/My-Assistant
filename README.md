🎙️ Voice-Activated Virtual Assistant
A lightweight, fully customizable voice assistant built using cutting-edge speech and language models. Designed for hands-free interaction and fast response time, this assistant listens for a wake word, understands spoken commands, and performs various web-based actions or speaks responses back.

🚀 Features
🔊 Voice Wake Word Detection — Say "hello" to activate the assistant.

🧠 Intent Recognition — Uses sentence embeddings to understand your command.

🌐 Performs Real Actions — Opens YouTube, Google Maps, Amazon, News, Weather, etc.

🗣️ Natural Speech Feedback — Speaks responses using Google Text-to-Speech.

🔥 Low Latency — Fast wake-to-response cycle.

⚙️ Easy Customization — Add new intents or actions in just a few lines of code.

🖥️ Auto-Startup Ready — Built to run automatically after system boot.

🛠️ Tech Stack
OpenAI Whisper – Speech-to-text

Vosk – Offline wake word recognition

Sentence Transformers – Semantic intent detection

Google Text-to-Speech (gTTS) – Speech synthesis

pygame, sounddevice, soundfile, librosa – Audio recording/playback

webbrowser, win11toast – Web and desktop notification integrations

📦 Installation
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/voice-assistant.git
cd voice-assistant
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download the Vosk model and place it in the root folder:
Vosk Model (small-en-us)

Run the app:

bash
Copy
Edit
python app.py
🧠 Supported Intents
Examples of commands you can try:

Search: “Search for machine learning on Google”

Weather: “What’s the weather like today?”

YouTube: “Play music on YouTube”

Nearby: “Find nearest petrol pump”

Translate: “Translate hello to Spanish”

Time, News, Quotes, and more!

🛠️ Customizing Intents
To add new intents or actions:

Add examples under INTENT_DB in app.py.

(Optional) Add a URL/action in ACTION_DISPATCH.

Example:

python
Copy
Edit
"BookFlight": ["book flight", "find flights", "plane tickets"],


📄 License
MIT License

