import cv2
from insightface.app import FaceAnalysis
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import os
import time

from utils import check_happy_sad_pattern, load_known_faces, save_access_log, speak, show_notification


def initialize_face_app():
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])  # or 'MPSExecutionProvider'
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def start_camera_recognition_loop(app, known_face_encodings, known_face_names):
    cap = cv2.VideoCapture(0)
    emotion_history = {}
    access_log = {}
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print("[INFO] System initialized and ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        results = app.get(frame)
        for face in results:
            bbox = face.bbox.astype(int)
            embedding = face.embedding.astype("float32").reshape(1, -1)

            name = "Unknown"
            best_score = 0.0

            for known_emb, known_name in zip(known_face_encodings, known_face_names):
                known_emb = known_emb.astype("float32").reshape(1, -1)
                if embedding.shape != known_emb.shape:
                    print(f"[WARN] Shape mismatch for {known_name}: {embedding.shape} vs {known_emb.shape}")
                    continue
                similarity = cosine_similarity(embedding, known_emb)[0][0]
                if similarity > 0.5 and similarity > best_score:
                    best_score = similarity
                    name = known_name

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            try:
                x1, y1, x2, y2 = bbox
                face_crop = frame[y1:y2, x1:x2]
                emotion_analysis = DeepFace.analyze(face_crop, actions=['emotion', 'gender'], enforce_detection=False)
                emotion = emotion_analysis[0]["dominant_emotion"]
                gender_probs = emotion_analysis[0]["gender"]
                gender = max(gender_probs, key=gender_probs.get)
            except Exception:
                emotion = "Unknown"
                gender = "Unknown"

            label = f"{name} ({best_score:.2f}) - {emotion} - {gender}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            now = time.time()

            if name not in emotion_history:
                emotion_history[name] = []

            emotion_history[name].append((now, emotion))
            emotion_history[name] = [(t, e) for (t, e) in emotion_history[name] if now - t <= 5]

            if name != "Unknown" and name not in access_log:
                access_log[name] = {
                    "last_emotion": emotion,
                    "last_gender": gender,
                    "easter_egg": False,
                    "timestamps": [datetime.fromtimestamp(now)],
                    "first_seen": datetime.fromtimestamp(now),
                }
            elif name != "Unknown":
                access_log[name]["last_emotion"] = emotion
                access_log[name]["last_gender"] = gender
                access_log[name]["timestamps"].append(datetime.fromtimestamp(now))

            # Check Easter Egg pattern using actual emotion_history
            if name != "Unknown" and check_happy_sad_pattern(emotion_history[name]):
                if not access_log[name]["easter_egg"]:
                    print(f"🎉 Easter Egg detected for {name}! Door opening and sending email...")
                    access_log[name]["easter_egg"] = True

                    # On-screen notification
                    show_notification(f"Easter Egg for {name}!")

                    # Voice announcement for Easter Egg
                    speak(f"Easter egg activated for {name}!")
                emotion_history[name].clear()

        cv2.imshow("Face Recognition - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    save_access_log(access_log)


def main():
    known_face_encodings, known_face_names = load_known_faces()
    app = initialize_face_app()
    start_camera_recognition_loop(app, known_face_encodings, known_face_names)


if __name__ == "__main__":
    main()