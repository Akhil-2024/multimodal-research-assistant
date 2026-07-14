from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

load_dotenv("../.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index
index = faiss.read_index("pdf_index.faiss")

# Load metadata
with open("metadata.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks_faiss(query, top_k=3):
    query_embedding = embedding_model.encode([query]).astype("float32")

    # Normalize query for cosine similarity
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        chunk = metadata[idx]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "score": float(score),
            "text": chunk["text"]
        })

    return results

print("===== FAISS RAG Chatbot =====")
print("Type 'exit' to quit\n")

while True:
    question = input("Ask PDF question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    top_chunks = retrieve_chunks_faiss(question)

    if top_chunks[0]["score"] < 0.30:
        print("\nAssistant: I don't know from the PDF.")
        print()
        continue

    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    prompt = f"""
You are a research assistant.
Answer the question using ONLY the given PDF context.
If the answer is not present in the context, say: "I don't know from the PDF."

PDF Context:
{context}

Question:
{question}

Answer briefly:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180
        )

        print("\nAssistant:", response.choices[0].message.content)
        print("\nSources:")

        for chunk in top_chunks:
            print(
                f"- Page {chunk['page']} | Chunk {chunk['chunk_id']} | FAISS Score: {chunk['score']:.3f}"
            )

        print()

    except Exception as e:
        print("\nError:", e)
        print()