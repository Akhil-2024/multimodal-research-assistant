import json
import faiss
import numpy as np

# Load embeddings
with open("embeddings.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Convert embeddings to numpy array
embedding_matrix = np.array(
    [chunk["embedding"] for chunk in chunks],
    dtype="float32"
)

# Normalize vectors for cosine similarity
faiss.normalize_L2(embedding_matrix)

# Get embedding dimension
dimension = embedding_matrix.shape[1]

# Create FAISS index
index = faiss.IndexFlatIP(dimension)

# Add embeddings to index
index.add(embedding_matrix)

# Save FAISS index
faiss.write_index(index, "pdf_index.faiss")

# Save metadata separately
metadata = []

for chunk in chunks:
    metadata.append({
        "chunk_id": chunk["chunk_id"],
        "page": chunk["page"],
        "text": chunk["text"]
    })

with open("metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

print("FAISS index created successfully.")
print(f"Total vectors stored: {index.ntotal}")
print("Saved files:")
print("- pdf_index.faiss")
print("- metadata.json")