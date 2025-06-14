import pyttsx3
import csv
import pickle
import os
from datetime import datetime
import cv2
import numpy as np

def check_happy_sad_pattern(history):
    """
    Checks if the emotion history matches the pattern: happy -> sad,
    where the sad emotion lasts at least 2 seconds consecutively.
    Args:
        history (list of tuples): List of (timestamp, emotion) tuples sorted by timestamp.
                                   Timestamp can be float or datetime.
    Returns:
        bool: True if the pattern is found, False otherwise.
    """
    if not history:
        return False

    def duration(start, end):
        if hasattr(start, "total_seconds"):
            return (end - start).total_seconds()
        else:
            return end - start

    idx = len(history) - 1
    if history[idx][1] != "sad":
        return False
    sad_end_ts = history[idx][0]
    while idx >= 0 and history[idx][1] == "sad":
        idx -= 1
    sad_start_ts = history[idx + 1][0]

    if duration(sad_start_ts, sad_end_ts) < 2:
        return False

    for t, emo in history[: idx + 1]:
        if emo == "happy":
            return True

    return False

def load_known_faces(pickle_path='data/embeddings/known_faces.pkl'):
    """
    Loads known face embeddings and names from a pickle file.
    Supports both legacy dict {name: embedding} or structured dict
    {'encodings': [...], 'names': [...]}.
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'encodings' not in data:
        known_face_names = list(data.keys())
        known_face_encodings = list(data.values())
    else:
        known_face_encodings = data.get('encodings', [])
        known_face_names = data.get('names', [])
    return known_face_encodings, known_face_names

def save_access_log(access_log, log_dir='logs', filename='access_log.csv'):
    """
    Saves the access log entries to a CSV file in the specified directory.
    Each row contains: name, first_seen, last_emotion, last_gender, easter_egg_triggered.
    """
    # Always use a single log file
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    if not access_log:
        return log_path

    rows = []
    for name, entry in access_log.items():
        rows.append({
            'name': name,
            'first_seen': entry.get('first_seen').strftime('%Y-%m-%d %H:%M:%S'),
            'last_emotion': entry.get('last_emotion', ''),
            'last_gender': entry.get('last_gender', ''),
            'easter_egg_triggered': 'Yes' if entry.get('easter_egg') else 'No'
        })

    fieldnames = ['name', 'first_seen', 'last_emotion', 'last_gender', 'easter_egg_triggered']
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return log_path


def speak(text):
    """
    Speaks the given text using the pyttsx3 TTS engine.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def show_notification(text, duration=2):
    """
    Displays a transient OpenCV window with the given text for a specified duration (in seconds).
    """
    height, width = 100, 400
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Notification", img)
    cv2.waitKey(int(duration * 1000))
    cv2.destroyWindow("Notification")