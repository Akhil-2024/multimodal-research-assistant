import json
from sentence_transformers import SentenceTransformer

# Load chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Free local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

# Convert text chunks into vectors
embeddings = model.encode(texts).tolist()

# Save chunks + embeddings
for i in range(len(chunks)):
    chunks[i]["embedding"] = embeddings[i]

with open("embeddings.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4)

print(f"Created embeddings for {len(chunks)} chunks")
print("Saved to embeddings.json")