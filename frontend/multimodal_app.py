"""Streamlit client for the FastAPI multimodal service (legacy frontend/app.py is retained)."""
import os
import uuid
from urllib.parse import quote

import requests
import streamlit as st

st.set_page_config(page_title="Multimodal Search & RAG", layout="wide")
st.title("🔎 Multimodal Semantic Search, Recommendation & RAG")
st.caption("Upload PDFs and images, find related content across modalities, and chat with cited PDF sources.")

API_URL = st.sidebar.text_input(
    "FastAPI server URL", os.getenv("API_URL", "http://127.0.0.1:8000")
).rstrip("/")
top_k = st.sidebar.slider("Top-K results", 1, 15, 5)
st.sidebar.info("PDF text: MiniLM. Images and cross-modal similarity: CLIP. Storage: FAISS + SQLite.")
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "conversation" not in st.session_state:
    st.session_state.conversation = []


def request_json(method, route, **kwargs):
    try:
        response = requests.request(method, API_URL + route, timeout=180, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" | {exc.response.text[:350]}"
        st.error(f"API request failed: {exc}{detail}")
        return None


def render_results(items):
    if not items:
        st.info("No matching indexed content yet.")
        return
    for idx, hit in enumerate(items):
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(
                    f"**{hit.get('source_file', 'Unknown')}** — {hit.get('kind')} "
                    + (f"— page {hit['page']}" if hit.get("page") else "")
                )
                st.caption(
                    f"ID: {hit['id']} | cosine score: {hit.get('score', 0):.3f} "
                    f"| space: {hit.get('embedding_space')}"
                )
                if hit.get("text"):
                    st.write(hit["text"][:650])
                if st.button("Recommend similar", key=f"rec_{idx}_{hit['id']}"):
                    result = request_json("GET", "/recommend/" + quote(hit["id"], safe=""),
                                          params={"top_k": top_k})
                    if result is not None:
                        st.session_state.recommendations = result["results"]
            with right:
                if hit.get("image_url"):
                    try:
                        image = requests.get(API_URL + hit["image_url"], timeout=20)
                        image.raise_for_status()
                        st.image(image.content, use_container_width=True)
                    except requests.exceptions.RequestException:
                        st.caption("Image preview unavailable.")


try:
    status = requests.get(API_URL + "/health", timeout=3).json()
    st.sidebar.success(f"Connected | PDF chunks: {status['text_chunks']} | Images: {status['images']}")
    if not status["llm_configured"]:
        st.sidebar.warning("No OPENAI_API_KEY: chat will return retrieved passages.")
except (requests.exceptions.RequestException, ValueError, KeyError):
    st.sidebar.warning("FastAPI server not available. Start it before using the UI.")

tab_ingest, tab_search, tab_image, tab_chat = st.tabs(
    ["📥 Ingest files", "🔍 Text search", "🖼️ Search by image", "💬 PDF RAG chat"]
)

with tab_ingest:
    st.subheader("Build a multimodal knowledge base")
    pdfs = st.file_uploader("PDF documents", type=["pdf"], accept_multiple_files=True, key="pdfs")
    if st.button("Index selected PDFs"):
        for file in pdfs or []:
            result = request_json(
                "POST", "/documents/pdf",
                files={"file": (file.name, file.getvalue(), "application/pdf")},
            )
            if result is not None:
                st.success(f"{file.name}: {result['chunks_added']} new text chunks")
    images = st.file_uploader(
        "Image files", type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True, key="images"
    )
    caption = st.text_input("Optional image caption / metadata")
    if st.button("Index selected images"):
        for file in images or []:
            result = request_json(
                "POST", "/documents/image",
                files={"file": (file.name, file.getvalue(), file.type or "image/png")},
                data={"caption": caption},
            )
            if result is not None:
                st.success(f"{file.name}: {'indexed' if result['added'] else 'already indexed'}")

with tab_search:
    st.subheader("Text → PDF passages and/or images")
    with st.form("text_search"):
        query = st.text_input("Search query", placeholder="e.g. wind turbine control diagram")
        modality = st.selectbox("Search target", ["all", "text", "image"])
        submitted = st.form_submit_button("Search")
    if submitted:
        result = request_json("POST", "/search/text",
                              json={"query": query, "top_k": top_k, "modality": modality})
        if result is not None:
            st.session_state.text_results = result["results"]
            st.session_state.recommendations = []
    render_results(st.session_state.get("text_results", []))
    if st.session_state.get("recommendations"):
        st.subheader("Similar recommendations")
        render_results(st.session_state.recommendations)

with tab_image:
    st.subheader("Image → similar images or PDF text")
    image_query = st.file_uploader(
        "Upload a query image", type=["png", "jpg", "jpeg", "webp"], key="image_query"
    )
    image_target = st.selectbox("Search target", ["all", "image", "text"], key="image_target")
    if st.button("Search using image") and image_query is not None:
        result = request_json(
            "POST", "/search/image",
            files={"file": (image_query.name, image_query.getvalue(), image_query.type)},
            data={"top_k": top_k, "modality": image_target}
        )
        if result is not None:
            st.session_state.image_results = result["results"]
    render_results(st.session_state.get("image_results", []))

with tab_chat:
    st.subheader("Conversation grounded in indexed PDFs")
    st.caption(f"Session: {st.session_state.session_id}")
    if st.button("Clear this conversation"):
        request_json("DELETE", "/sessions/" + st.session_state.session_id)
        st.session_state.conversation = []
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ask a question about your indexed PDF documents")
    if question:
        st.session_state.conversation.append({"role": "user", "content": question})
        result = request_json("POST", "/chat", json={
            "question": question, "session_id": st.session_state.session_id, "top_k": top_k
        })
        if result is not None:
            answer = result["answer"]
            source_lines = [
                f"{h['source_file']} p.{h['page']} (score {h['score']:.2f})"
                for h in result.get("sources", [])
            ]
            if source_lines:
                answer += "\n\n**Retrieved sources:** " + "; ".join(source_lines)
            st.session_state.conversation.append({"role": "assistant", "content": answer})
            st.rerun()
