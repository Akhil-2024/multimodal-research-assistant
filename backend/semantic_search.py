import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("embeddings.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("Ask question about PDF: ")

query_embedding = model.encode(query)

scores = []

for chunk in chunks:
    chunk_embedding = np.array(chunk["embedding"])

    similarity = np.dot(query_embedding, chunk_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
    )

    scores.append(
        {
            "chunk_id": chunk["chunk_id"],
            "score": similarity,
            "text": chunk["text"]
        }
    )

scores = sorted(scores, key=lambda x: x["score"], reverse=True)

top_k = 3

print("\n===== TOP RELEVANT CHUNKS =====\n")

for result in scores[:top_k]:
    print("Chunk ID:", result["chunk_id"])
    print("Score:", result["score"])
    print(result["text"][:700])
    print("-" * 80)
