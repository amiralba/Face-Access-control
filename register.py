import cv2
import os
import time
import pickle
import numpy as np
from insightface.app import FaceAnalysis

def capture_video_frames(video_path='data/videos/recorded.mp4', frames_dir='data/frames', duration=5, fps=20):
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(video_path, codec, fps, (width, height))

    print(f"🎥 Recording for {duration} seconds...")
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("📸 Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"✅ Saved {frame_count} frames to {frames_dir}")

def extract_embeddings_from_frames(frames_dir='data/frames'):
    embeddings = []
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    for filename in sorted(os.listdir(frames_dir)):
        if filename.endswith('.jpg'):
            frame_path = os.path.join(frames_dir, filename)
            image = cv2.imread(frame_path)
            faces = app.get(image)
            if faces:
                embedding = faces[0].embedding.astype("float32")
                embeddings.append(embedding)
    return embeddings

def save_embedding(name, embedding, db_path='data/embeddings/known_faces.pkl'):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding.flatten()
    elif embedding.ndim != 1:
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            db = pickle.load(f)
    else:
        db = {}
    db[name] = embedding
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)
    print(f"💾 Saved embedding for {name} to {db_path}")

if __name__ == "__main__":
    name = input("Enter the name of the person to register: ").strip().lower().replace(" ", "_")
    video_path = f"data/videos/{name}.mp4"
    frames_dir = f"data/frames/{name}"
    capture_video_frames(video_path=video_path, frames_dir=frames_dir)
    embeddings = extract_embeddings_from_frames(frames_dir=frames_dir)
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0).astype("float32")
        save_embedding(name, avg_embedding)
