import pickle

with open("data/embeddings/known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)

print(f"[DEBUG] Loaded {len(known_faces)} known faces:")
for name, embedding in known_faces.items():
    print(f" - {name}: embedding shape = {embedding.shape}")
