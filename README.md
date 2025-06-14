
Face Access Control System

An advanced real-time Face Access Control system with emotion analysis, easter-egg triggers, voice feedback, on-screen notifications, and session logging.

Features
	•	Real-time Face Detection & Recognition
	•	Uses InsightFace buffalo_l model for accurate detection and embeddings.
	•	Emotion & Gender Analysis
	•	Integrates DeepFace for dominant emotion and gender analysis.
	•	Easter-Egg Logic
	•	Detects a “happy” → “sad for ≥2s” pattern to unlock special events.
	•	Voice Feedback
	•	Offline text-to-speech using pyttsx3 to greet users and announce events.
	•	On-Screen Notifications
	•	Transient OpenCV pop-ups for visual feedback.
	•	Session Logging
	•	Persistent CSV log (logs/access_log.csv) capturing first detection, last emotion, gender, and easter-egg status.
	•	Modular & Clean Design
	•	Utilities separated into utils.py, main logic in register.py and recognizer.py.

Getting Started

Prerequisites
	•	Python 3.10 (tested on macOS Apple Silicon)
	•	venv for isolated environments

Installation
	1.	Clone this repo:

git clone https://github.com/amiralba/face-access-control.git
cd face-access-control


	2.	Create & activate a virtual environment:

python3.10 -m venv venv
source venv/bin/activate


	3.	Install dependencies:

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt



Folder Structure

face-access-control/
├── data/
│   ├── embeddings/      # Stored known face embeddings
│   └── frames/          # Captured frames per registration
├── logs/                # access_log.csv
├── utils.py             # Helper utilities
├── register.py          # Face registration via 5s video
├── recognizer.py        # Real-time recognition loop
├── requirements.txt     # Python dependencies
└── README.md            # This file

Usage

1. Register Faces

python register.py

	•	Enter a name and record a 5-second webcam video.
	•	Embeddings are extracted and saved to data/embeddings/known_faces.pkl.

2. Run Real-Time Recognition

python recognizer.py

	•	Opens webcam feed, detects & recognizes faces.
	•	Shows bounding box, name, similarity, emotion, gender.
	•	Audio greeting & on-screen notifications for easter-egg.
	•	Press q to quit; logs saved to logs/access_log.csv.

Technologies & Models
	•	InsightFace buffalo_l: Face detection, landmarks, recognition embeddings.
	•	ONNX Runtime (Silicon): Accelerated model inference on Apple M-series.
	•	DeepFace: Emotion and gender classification.
	•	OpenCV: Video I/O, drawing, notifications.
	•	NumPy, scikit-learn: Embedding math and similarity.
	•	pyttsx3: Offline text-to-speech engine.

Future Work
	•	Liveness Detection (anti-spoofing, blink/mouth movement)
	•	Multi-Camera Support (RTSP/IP cameras)
	•	FastAPI & Docker: Expose recognition as a service
	•	Web Dashboard: Live feed & analytics (React or Streamlit)
	•	Raspberry Pi / Jetson: Edge deployment

Screenshots

License

This project is licensed under the MIT License. See LICENSE for details.