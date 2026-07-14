import streamlit as st
import time
import pickle
import hashlib
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import os

load_dotenv("../.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_HISTORY_FILE = "chat_history.json"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Multi-Modal Research Assistant", layout="wide")

st.title("📚 Multi-PDF Intelligent Research Assistant")
st.write("Upload multiple PDFs, summarize them, extract key points, and ask questions using FAISS-based RAG.")

st.sidebar.title("⚙️ Settings")

low_cost_mode = st.sidebar.checkbox("Low Cost Mode", value=True)

if low_cost_mode:
    top_k_value = 3
    answer_tokens = 120
    summary_chunks = 5
    summary_tokens = 300
    keypoint_tokens = 180
    memory_window = 4
    st.sidebar.success("Low Cost Mode ON")
else:
    top_k_value = 5
    answer_tokens = 220
    summary_chunks = 8
    summary_tokens = 500
    keypoint_tokens = 300
    memory_window = 8
    st.sidebar.warning("Low Cost Mode OFF")

st.sidebar.write(f"Top-K Chunks: {top_k_value}")
st.sidebar.write(f"Answer Tokens: {answer_tokens}")
st.sidebar.write(f"Summary Chunks: {summary_chunks}")
st.sidebar.write(f"Summary Tokens: {summary_tokens}")
st.sidebar.write(f"Key Point Tokens: {keypoint_tokens}")
st.sidebar.write(f"Memory Window: {memory_window}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(chat_history, file, indent=4, ensure_ascii=False)


def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def export_chat_history(chat_history):
    text_data = ""

    for msg in chat_history:
        if msg["role"] == "user":
            text_data += f"You: {msg['content']}\n\n"
        else:
            text_data += f"Assistant: {msg['content']}\n\n"

    return text_data


def get_files_hash(uploaded_files):
    hash_obj = hashlib.md5()

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        hash_obj.update(uploaded_file.name.encode("utf-8"))
        hash_obj.update(file_bytes)

    return hash_obj.hexdigest()


def extract_text_from_pdfs(uploaded_files):
    chunks = []
    chunk_size = 500
    overlap = 150

    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        file_name = uploaded_file.name

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text:
                continue

            start = 0

            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                chunks.append({
                    "chunk_id": len(chunks),
                    "source_file": file_name,
                    "page": page_num,
                    "text": chunk_text
                })

                start = end - overlap

    return chunks


def build_faiss_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve_chunks_faiss(query, index, chunks, top_k):
    query_embedding = embedding_model.encode([query]).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "score": float(score),
            "text": chunk["text"]
        })

    return results


def generate_research_summary(chunks, summary_chunks, summary_tokens):
    summary_context = "\n\n".join(
        [
            f"Source: {chunk['source_file']}, Page: {chunk['page']}\n{chunk['text']}"
            for chunk in chunks[:summary_chunks]
        ]
    )

    summary_prompt = f"""
You are an AI research assistant.

Using ONLY the given PDF context, generate a structured research summary.

PDF Context:
{summary_context}

Give the output in this format:

1. Short Summary
2. Main Topic
3. Key Concepts
4. Methodology / Approach
5. Important Technical Points
6. Limitations
7. Future Scope
8. Useful Notes for Students

Keep it clear and concise.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=summary_tokens
    )

    return response.choices[0].message.content


def generate_key_points(chunks, keypoint_tokens):
    keypoint_context = "\n\n".join(
        [
            f"Source: {chunk['source_file']}, Page: {chunk['page']}\n{chunk['text']}"
            for chunk in chunks[:5]
        ]
    )

    keypoint_prompt = f"""
You are an AI research assistant.

Using ONLY the given PDF context, extract the most important key points.

PDF Context:
{keypoint_context}

Give 5 to 7 concise key points.
Mention source file/page if useful.
Do not add information outside the PDF.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": keypoint_prompt}],
        max_tokens=keypoint_tokens
    )

    return response.choices[0].message.content


if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()


if st.sidebar.button("Clear Chat Memory"):
    st.session_state.chat_history = []
    save_chat_history([])
    st.sidebar.success("Chat memory cleared.")

if st.session_state.chat_history:
    chat_text = export_chat_history(st.session_state.chat_history)

    st.sidebar.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name="research_chat_history.txt",
        mime="text/plain"
    )


uploaded_pdfs = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_pdfs:
    st.success(f"{len(uploaded_pdfs)} PDF(s) uploaded successfully.")

    uploaded_file_names = [file.name for file in uploaded_pdfs]

    if (
        "uploaded_file_names" not in st.session_state
        or st.session_state.uploaded_file_names != uploaded_file_names
    ):
        with st.spinner("Loading or processing PDFs..."):
            files_hash = get_files_hash(uploaded_pdfs)

            chunks_cache_path = os.path.join(CACHE_DIR, f"{files_hash}_chunks.pkl")
            faiss_cache_path = os.path.join(CACHE_DIR, f"{files_hash}_index.faiss")

            if os.path.exists(chunks_cache_path) and os.path.exists(faiss_cache_path):
                with open(chunks_cache_path, "rb") as file:
                    chunks = pickle.load(file)

                faiss_index = faiss.read_index(faiss_cache_path)

                st.success("Loaded cached PDF knowledge base.")

            else:
                chunks = extract_text_from_pdfs(uploaded_pdfs)

                if len(chunks) == 0:
                    st.error("No readable text found in the uploaded PDFs.")
                    st.stop()

                faiss_index = build_faiss_index(chunks)

                with open(chunks_cache_path, "wb") as file:
                    pickle.dump(chunks, file)

                faiss.write_index(faiss_index, faiss_cache_path)

                st.success("Processed PDFs and saved cache.")

            st.session_state.chunks = chunks
            st.session_state.faiss_index = faiss_index
            st.session_state.uploaded_file_names = uploaded_file_names
            st.session_state.chat_history = []
            save_chat_history([])

        st.success(f"Knowledge base created. Total chunks: {len(st.session_state.chunks)}")

    st.subheader("📄 Research Paper Summarization")

    if st.button("Generate Research Summary"):
        with st.spinner("Generating research summary..."):
            summary = generate_research_summary(
                st.session_state.chunks,
                summary_chunks,
                summary_tokens
            )

        st.subheader("Generated Research Summary")
        st.write(summary)

    st.subheader("📝 Extract Key Points")

    if st.button("Generate Key Points"):
        with st.spinner("Extracting key points..."):
            key_points = generate_key_points(
                st.session_state.chunks,
                keypoint_tokens
            )

        st.subheader("Important Key Points")
        st.write(key_points)

    st.subheader("💬 Conversational PDF Chat")

    if st.session_state.chat_history:
        st.write("### Chat History")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")

    question = st.text_input("Ask a question or follow-up about the uploaded PDFs:")

    if st.button("Ask"):
        start_time = time.time()

        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            top_chunks = retrieve_chunks_faiss(
                question,
                st.session_state.faiss_index,
                st.session_state.chunks,
                top_k_value
            )

            top_score = top_chunks[0]["score"]
            avg_score = sum(chunk["score"] for chunk in top_chunks) / len(top_chunks)

            if top_score < 0.30:
                end_time = time.time()
                latency = end_time - start_time

                answer = "I don't know from the uploaded PDFs."

                st.error(answer)

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

                save_chat_history(st.session_state.chat_history)

                st.subheader("📊 Evaluation Metrics")
                st.write(f"Top Retrieval Score: {top_score:.3f}")
                st.write(f"Average Retrieval Score: {avg_score:.3f}")
                st.write(f"Response Latency: {latency:.2f} seconds")
                st.error("Answer Status: Low Confidence")

            else:
                context = "\n\n".join(
                    [
                        f"Source: {chunk['source_file']}, Page: {chunk['page']}\n{chunk['text']}"
                        for chunk in top_chunks
                    ]
                )

                recent_memory = st.session_state.chat_history[-memory_window:]

                messages = [
                    {
                        "role": "system",
                        "content": "You are a concise research assistant. Answer only using the uploaded PDF context. If answer is not present, say you don't know from the PDFs."
                    }
                ]

                for msg in recent_memory:
                    messages.append(msg)

                user_prompt = f"""
PDF Context:
{context}

Current Question:
{question}

Instructions:
- Answer only using the PDF context.
- Keep the answer brief and clear.
- At the end, add a section called "Sources Used".
- In "Sources Used", mention the source file name and page number.
- If the answer is not present in the PDF context, say:
  "I don't know from the uploaded PDFs."

Answer Format:

Answer:
<your answer>

Sources Used:
- <source file>, Page <page number>
"""

                messages.append({
                    "role": "user",
                    "content": user_prompt
                })

                with st.spinner("Generating answer..."):
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=messages,
                        max_tokens=answer_tokens
                    )

                answer = response.choices[0].message.content

                end_time = time.time()
                latency = end_time - start_time

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

                save_chat_history(st.session_state.chat_history)

                st.subheader("Response")
                st.write(answer.replace("Answer:", "").strip())

                best_chunk = top_chunks[0]

                st.subheader("🏆 Best Matching Source")
                st.info(
                    f"{best_chunk['source_file']} | Page {best_chunk['page']} | "
                    f"Chunk {best_chunk['chunk_id']} | FAISS Score: {best_chunk['score']:.3f}"
                )

                with st.expander("View Best Matching Text"):
                    st.write(best_chunk["text"])

                st.subheader("All Retrieved Sources")
                for chunk in top_chunks:
                    st.write(
                        f"{chunk['source_file']} | Page {chunk['page']} | "
                        f"Chunk {chunk['chunk_id']} | FAISS Score: {chunk['score']:.3f}"
                    )

                with st.expander("🔍 Show Retrieved Context"):
                    for chunk in top_chunks:
                        st.markdown(
                            f"**{chunk['source_file']} | Page {chunk['page']} | Chunk {chunk['chunk_id']}**"
                        )
                        st.write(chunk["text"])
                        st.write("---")

                st.subheader("📊 Evaluation Metrics")
                st.write(f"Top Retrieval Score: {top_score:.3f}")
                st.write(f"Average Retrieval Score: {avg_score:.3f}")
                st.write(f"Response Latency: {latency:.2f} seconds")

                if top_score >= 0.50:
                    st.success("Answer Status: Strongly Grounded")
                elif top_score >= 0.30:
                    st.warning("Answer Status: Moderately Grounded")
                else:
                    st.error("Answer Status: Low Confidence")

else:
    st.info("Please upload one or more PDFs to begin.")