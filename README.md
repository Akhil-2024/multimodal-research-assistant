# Multimodal Semantic Search, Recommendation & RAG Platform

**A multimodal research assistant for semantic PDF search, image/text retrieval, similarity-based recommendations, and conversational RAG.**

**Stack:** Python 3.11 · PyTorch · CLIP · Sentence Transformers · FAISS · FastAPI · Streamlit · SQLite · OpenAI API (optional) · Docker (optional)

[![API tests](https://github.com/Akhil-2024/multimodal-research-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/Akhil-2024/multimodal-research-assistant/actions/workflows/tests.yml)

> **Project status:** The essential features are implemented and committed on `main`. Automated API/integration-style tests using mock embedding models have passed on GitHub Actions. A full real-model retrieval benchmark and production deployment have **not** been completed. **Deployment is optional and intentionally not required** to understand, test, or demonstrate the project.

## Table of contents

1. [Problem statement and objective](#1-problem-statement-and-objective)
2. [What I have implemented](#2-what-i-have-implemented)
3. [End-to-end architecture](#3-end-to-end-architecture)
4. [Development procedure: how the project was built](#4-development-procedure-how-the-project-was-built)
5. [Repository structure and files](#5-repository-structure-and-files)
6. [Technology and algorithm explanations](#6-technology-and-algorithm-explanations)
7. [Step-by-step setup on Windows CMD (no Docker)](#7-step-by-step-setup-on-windows-cmd-no-docker)
8. [Demonstrate every feature](#8-demonstrate-every-feature)
9. [FastAPI endpoint reference and CMD examples](#9-fastapi-endpoint-reference-and-cmd-examples)
10. [Testing and verification](#10-testing-and-verification)
11. [Optional Docker setup](#11-optional-docker-setup)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations, security and remaining work](#13-limitations-security-and-remaining-work)
14. [Resume and interview explanation](#14-resume-and-interview-explanation)

## 1. Problem statement and objective

**Problem:** Traditional keyword search struggles to find material expressed in different words; plain PDF chat cannot retrieve images using text or retrieve related PDF passages using an image. Users also need context-aware follow-up questions and a way to discover related content.

**Goal:** Build a single prototype where a user can:

- Upload one or more **text-based PDF documents** and **images**.
- Search PDF passages by meaning, not only exact keywords.
- Use **text to find images**, **an image to find other images**, or **an image to find related PDF text**.
- Receive recommendations for semantically similar indexed items.
- Ask PDF-grounded questions, get filenames/page references, and continue a conversation.
- Access the functionality through a **FastAPI REST service** and a **Streamlit UI**.

**Concrete example:** Upload a research PDF about wind turbines and several relevant images. Search for "wind turbine control" to retrieve related passages/images; upload a turbine image to retrieve visually or semantically related indexed content; then ask the chat assistant to explain a concept from the PDF.

## 2. What I have implemented

| Feature | Status | Where |
|---|---|---|
| PDF extraction, overlapping chunks and page metadata | Implemented | `backend/platform.py` |
| SentenceTransformer dense embeddings | Implemented | `backend/platform.py` |
| FAISS-based text semantic search | Implemented | `backend/platform.py` |
| CLIP image and text embeddings (PyTorch-backed) | Implemented | `backend/platform.py` |
| Text → image retrieval | Implemented | `backend/platform.py` |
| Image → image retrieval | Implemented | `backend/platform.py` |
| Image → PDF-text retrieval | Implemented | `backend/platform.py` |
| Mixed text search across image/text results | Implemented (rank fusion) | `backend/platform.py` |
| Similar-item recommendation | Implemented | `backend/platform.py` |
| Retrieval-augmented PDF question answering | Implemented; LLM requires an API key | `backend/platform.py` |
| Session-based conversation memory | Implemented (SQLite) | `backend/platform.py` |
| FastAPI inference/upload/search/chat endpoints | Implemented | `backend/api.py` |
| Streamlit frontend for the new API | Implemented | `frontend/multimodal_app.py` |
| Automated tests without downloading model weights | Passed in GitHub Actions | `tests/test_platform.py` |
| Docker configuration | Included, **optional** | `Dockerfile`, `docker-compose.yml` |
| End-to-end real-model accuracy/latency benchmark | **Not yet measured** | Future work |
| Public/cloud deployment | **Not done; not required** | Future work |

**Important distinction:** The original `frontend/app.py` and older files under `backend/` were the **first PDF-only prototype**. The newer multimodal implementation is the combination of `backend/platform.py`, `backend/api.py`, and `frontend/multimodal_app.py`.

## 3. End-to-end architecture

~~~text
                      USER: PDF / IMAGE / QUESTION
                                  |
                                  v
                    Streamlit UI / Swagger / HTTP client
                                  |
                                  v
                         FASTAPI REST SERVICE
                           backend/api.py
                                  |
                                  v
                     MULTIMODAL SERVICE LAYER
                         backend/platform.py
              +-------------------+---------------------+
              |                   |                     |
              v                   v                     v
         PDF INGESTION      IMAGE INGESTION        SESSION MEMORY
         pypdf extract      Pillow decode          SQLite database
         page-wise chunks   RGB preprocessing      user/assistant turns
              |                   |
          +---+-----+             |
          |         |             |
          v         v             v
        MiniLM   CLIP text      CLIP image
      embeddings embeddings     embeddings
          |         |             |
          v         +------+------+
     text.faiss             |
     + metadata             v
                        clip.faiss
                        + metadata
          |                  |
          +--------+---------+
                   |
                   v
            NEAREST-NEIGHBOR SEARCH
          cosine similarity + optional
             reciprocal rank fusion
                   |
          +--------+---------+
          |                  |
          v                  v
    RAG RETRIEVAL     SIMILARITY RECOMMENDATION
    PDF passages      related indexed CLIP items
          |
          v
    Retrieved context + previous session messages
          |
          v
    Optional OpenAI LLM -> answer + PDF sources
    Without API key -> retrieved passages only
~~~

### How the two vector spaces work

- **Text index (MiniLM):** PDF chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`; a text query is encoded with the **same model**.
- **Multimodal index (CLIP):** Image embeddings and short representations of PDF text chunks are encoded with `clip-ViT-B-32`; image queries and text queries can search this shared CLIP space.
- **For a text query with target `all`:** The service searches MiniLM text hits and CLIP image hits and merges their **ranks** using Reciprocal Rank Fusion (RRF). It does **not** directly compare raw MiniLM and CLIP similarity scores.
- **RAG chat** is grounded in PDF text retrieved from the MiniLM index. It is not an image-question-answering LLM.

## 4. Development procedure: how the project was built

This describes the implementation stages and can also serve as a roadmap for rebuilding the project from scratch.

### Stage 1 — Build a PDF-based research assistant (original prototype)

1. **Read PDFs:** Implement PDF parsing using `pypdf` (`backend/pdf_reader.py`).
2. **Split text:** Create manageable overlapping pieces (`backend/chunk_pdf.py`).
3. **Encode passages:** Convert chunks into dense semantic vectors with Sentence Transformers (`backend/create_embeddings.py`).
4. **Search meaning:** First implement cosine similarity directly (`backend/semantic_search.py`), then build FAISS indexing (`backend/faiss_index.py`).
5. **Answer questions:** Retrieve top PDF passages and pass them as context to an LLM (`backend/rag_chatbot.py`).
6. **Add a research UI:** Build a multi-PDF Streamlit assistant with summarization, key-point extraction, chat history and retrieval displays (`frontend/app.py`).

**Output of Stage 1:** A PDF-only semantic-search/RAG prototype. Original generated index/cache files may still be present in the repository, but the new application uses a separate runtime directory.

### Stage 2 — Extend the system to multimodal retrieval

1. Keep the **MiniLM PDF text index** for high-quality text-to-text retrieval.
2. Add a pretrained **CLIP encoder**, backed by PyTorch through Sentence Transformers.
3. Encode uploaded images and PDF text chunks into CLIP's shared image/text embedding space.
4. Persist normalized vectors in **FAISS IndexFlatIP** and metadata in JSON files.
5. Support **text → images**, **image → images**, and **image → PDF passages**.
6. Add **Reciprocal Rank Fusion** for text queries that request both image and PDF results.
7. Generate stable IDs from content hashes to avoid indexing the same uploaded bytes repeatedly.

**Output of Stage 2:** Search and retrieval across multiple input modalities.

### Stage 3 — Implement recommendation and conversational RAG

1. Locate an existing item's vector in the CLIP index.
2. Retrieve the nearest neighboring items; exclude the queried item from its own recommendations.
3. For PDF chat, retrieve relevant text chunks and attach filename/page metadata.
4. Send retrieved passages to an LLM **only when an API key is configured**.
5. Save chat turns in **SQLite**, isolated by `session_id`; include recent turns when interpreting follow-up questions.
6. Provide a **retrieval-only fallback** when no LLM key is available.

**Output of Stage 3:** Similar-content recommendations and conversational PDF-grounded question answering.

### Stage 4 — Expose reusable APIs and user interface

1. Create FastAPI routes for PDF/image ingestion, multimodal search, recommendation, chat and session history.
2. Add input validation (e.g. allowed modalities, top-k ranges, maximum upload size).
3. Build a separate Streamlit client that talks to FastAPI over HTTP.
4. Add basic persistence under `runtime_data/`.
5. Add offline-style API tests and a GitHub Actions test workflow.

**Output of Stage 4:** A modular local application, testable through its UI or interactive FastAPI documentation.

### Stage 5 — Verification and next steps

- **Done:** Tests with mocked encoders cover ingestion/deduplication, both cross-modal retrieval directions, recommendation, image retrieval, chat fallback, session isolation and invalid inputs.
- **Still needed for a stronger portfolio:** Run a real CLIP/MiniLM sample dataset, manually inspect search results, measure Recall@K/MRR/latency, and collect updated screenshots.
- **Intentionally skipped:** Production/cloud deployment. It is **not** necessary to demonstrate the core résumé features.

## 5. Repository structure and files

~~~text
multimodal-research-assistant/
|
|-- backend/
|   |-- __init__.py             # Python package
|   |-- platform.py             # NEW: embeddings, ingestion, FAISS, RAG, memory
|   |-- api.py                  # NEW: FastAPI routes and request validation
|   |-- pdf_reader.py           # Original PDF reading script
|   |-- chunk_pdf.py            # Original chunking script
|   |-- create_embeddings.py    # Original embedding generation script
|   |-- semantic_search.py      # Original cosine semantic-search script
|   |-- faiss_index.py          # Original FAISS index-building script
|   |-- rag_chatbot.py          # Original CLI PDF RAG prototype
|   |-- chatbot.py              # Original CLI LLM chat + JSON memory
|   |-- *.json, *.faiss         # Legacy generated artifacts (where tracked)
|
|-- frontend/
|   |-- multimodal_app.py       # NEW: Streamlit client for FastAPI
|   |-- app.py                  # Original multi-PDF Streamlit app
|   |-- cache/                  # Legacy cache files
|
|-- tests/
|   |-- test_platform.py       # Offline-style API integration tests
|
|-- data/
|   |-- pdfs/
|       |-- sample.pdf          # Sample PDF supplied with original prototype
|
|-- docs/                     # Original prototype screenshots/diagram
|-- .github/workflows/
|   |-- tests.yml              # Automated GitHub Actions tests
|-- requirements.txt           # Python dependencies
|-- .env.example               # Example environment variables
|-- .gitignore                 # Ignore secrets and generated local files
|-- Dockerfile                 # Optional container image
|-- docker-compose.yml         # Optional API + UI containers
|-- README.md                  # Complete setup, concepts, demo and status
|
|-- runtime_data/              # GENERATED on first use; not committed
    |-- text.faiss             # Dense PDF text vectors
    |-- text_metadata.json     # PDF text chunk source/page metadata
    |-- clip.faiss             # Image + PDF text CLIP vectors
    |-- clip_metadata.json     # CLIP metadata / image locations
    |-- images/                # Indexed image previews
    |-- sessions.sqlite3       # Session-wise conversation history
~~~

### Key functions to explain in interviews

| Function / component | Responsibility |
|---|---|
| `chunk_text()` | Produce overlapping, bounded text chunks |
| `ingest_pdf()` | Parse pages, embed chunks using two models, save both indexes |
| `ingest_image()` | Decode image, compute CLIP embedding, persist image/index |
| `search_text()` | Search PDF chunks and/or images from a text query |
| `search_image()` | Use a query image to search indexed image/text vectors |
| `reciprocal_rank_fusion()` | Merge two ranked results without mixing incomparable raw scores |
| `recommend()` | Find nearest CLIP neighbors to an indexed item |
| `chat()` | Retrieve context, use recent session history, generate/fallback answer |
| `history()` / `clear_history()` | Read/clear session messages in SQLite |
| FastAPI `/docs` | Auto-generated interactive API interface |

## 6. Technology and algorithm explanations

| Technology | Why used |
|---|---|
| **Python** | Backend and ML orchestration |
| **PyTorch** | Neural model execution via Sentence Transformers / CLIP |
| **Sentence Transformers (MiniLM)** | Turns PDF text and user text into semantic vectors |
| **CLIP** | Places images and text into a shared representation for cross-modal retrieval |
| **FAISS** | Efficient vector nearest-neighbor retrieval and recommendations |
| **NumPy** | Float32 vector conversion and array operations |
| **pypdf** | Extracts selectable text from uploaded PDFs |
| **Pillow** | Validates and normalizes uploaded images |
| **FastAPI + Pydantic** | Typed HTTP endpoints and request validation |
| **Streamlit** | Simple UI for uploading, querying and displaying results |
| **SQLite** | Persists session-scoped conversation messages |
| **OpenAI API (optional)** | Generates an answer from retrieved PDF passages |
| **Docker (optional)** | Reproducible container packaging; **not required** to run locally |

### Essential algorithms

**Dense embeddings:** A neural model transforms a text passage or image into a numeric vector intended to capture semantic meaning.

**Cosine similarity:** For vectors `a` and `b`:

~~~text
cosine(a, b) = (a · b) / (||a|| * ||b||)
~~~

The service normalizes embeddings to unit length and uses **FAISS IndexFlatIP**. For normalized vectors, inner product is cosine similarity. IndexFlatIP is **exact exhaustive search**, not an approximate nearest-neighbor index.

**Reciprocal Rank Fusion (RRF):**

~~~text
RRF(item) = sum_over_ranked_lists [ 1 / (60 + rank(item)) ]
~~~

This allows combining **rank positions** from MiniLM text search and CLIP image search without pretending their raw similarity scores are directly comparable.

**Retrieval-Augmented Generation (RAG):**

~~~text
Question -> embed with MiniLM -> retrieve top-k PDF chunks
         -> attach source/page + recent chat turns
         -> optional LLM answer constrained to supplied passages
         -> return answer and retrieved source metadata
~~~

A similarity score is **not** a calibrated factual confidence percentage. LLM citations are requested in the prompt; check the returned source metadata yourself when accuracy matters.

## 7. Step-by-step setup on Windows CMD (no Docker)

**Recommended path for this project:** run locally and skip the lengthy Docker build.

### Prerequisites

- Windows 10/11 and **Command Prompt (CMD)**.
- **Python 3.11**, Git and internet access for Python dependencies / first model download.
- Enough free storage and memory for **PyTorch**, MiniLM, CLIP and FAISS.
- OpenAI API key **only if you want generated RAG answers**. Semantic search, recommendations, and retrieval-only chat do not need it.

> **CMD vs PowerShell:** CMD uses `copy`, `cd /d` and `set`. `Copy-Item` and `$HOME` are PowerShell syntax, **not CMD syntax**.

### Step 1 — Clone the repository

Open CMD in the folder where you want the project:

~~~bat
cd /d "%USERPROFILE%\Desktop"
git clone https://github.com/Akhil-2024/multimodal-research-assistant.git
cd multimodal-research-assistant
git switch main
~~~

If you **already cloned** the project, do not clone again. From its existing project directory:

~~~bat
git fetch origin
git switch main
git pull origin main
~~~

**Check:** `dir` should list `backend`, `frontend`, `requirements.txt` and `README.md`.

### Step 2 — Check Python

~~~bat
py -3.11 --version
~~~

Expected: a Python 3.11.x version. Install Python 3.11 and retry if the command fails.

### Step 3 — Create and activate a virtual environment

Run from the **repository root**:

~~~bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
~~~

Your prompt should now begin with `(.venv)`. This isolates the project dependencies from other Python projects.

### Step 4 — Install required Python packages

~~~bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

**Note:** PyTorch is a large package. The first installation may take significant time on a slow connection; you only need to install it once per environment.

### Step 5 — Configure the LLM (optional)

- **No key:** Skip this step. `/chat` returns relevant PDF excerpts in **retrieval-only** mode.
- **With a key:** In the **same CMD window that will run FastAPI**, set the environment variable:

~~~bat
set "OPENAI_API_KEY=YOUR_OPENAI_API_KEY"
~~~

Do not commit a real key or post it in screenshots. For the non-Docker startup described here, simply copying `.env.example` is **not enough to load the key**; the API reads the process environment variable.

Optionally select a different supported model:

~~~bat
set "OPENAI_MODEL=gpt-4.1-mini"
~~~

### Step 6 — Start the FastAPI backend (CMD window #1)

~~~bat
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
~~~

Keep the terminal open. Visit:

- **Health:** http://127.0.0.1:8000/health
- **Swagger UI:** http://127.0.0.1:8000/docs

A fresh empty installation should return a JSON object similar to:

~~~json
{
  "status": "ok",
  "text_chunks": 0,
  "images": 0,
  "clip_vectors": 0,
  "llm_configured": false
}
~~~

Counts increase when you upload files; `llm_configured` is `true` only when the API process sees an OpenAI key.

### Step 7 — Start the Streamlit frontend (CMD window #2)

Open a **second CMD window**, navigate to the **same cloned project folder**, and run:

~~~bat
.venv\Scripts\activate.bat
python -m streamlit run frontend\multimodal_app.py
~~~

Open: **http://localhost:8501**

The Streamlit client calls the API at **http://127.0.0.1:8000**. Start the backend first, otherwise the UI cannot serve queries.

### Step 8 — Download models on first indexing request

Uploading a PDF/image or running a search that needs a model triggers the first download of pretrained **MiniLM** and **CLIP** weights. Subsequent runs can reuse cached weights. A GPU is not required for this prototype.

**Stop:** Press **Ctrl+C** in the Streamlit and API terminal windows when finished.

## 8. Demonstrate every feature

Use the new Streamlit client at **http://localhost:8501**, or its API endpoints.

| Demo step | Action | What to verify |
|---|---|---|
| 1. Ingest PDF | Open **Ingest files** → select a text-based PDF → **Index selected PDFs** | New PDF chunk count; no empty-text error |
| 2. Ingest images | Upload 2–3 PNG/JPG images → **Index selected images** | Images count increases |
| 3. Text → PDF | **Text search** → choose `text` → ask a topic from your PDF | Relevant snippets with filename and page |
| 4. Text → image | **Text search** → choose `image` | Relevant images with preview and similarity score |
| 5. Mixed search | Choose `all` | Both image and PDF hits, when present |
| 6. Image → image | **Search by image** → choose `image` | Similar indexed images |
| 7. Image → PDF | **Search by image** → choose `text` | PDF passages retrieved through CLIP |
| 8. Recommend | Click **Recommend similar** under a result | Nearest indexed CLIP items excluding the original |
| 9. RAG chat | **PDF RAG chat** → ask about uploaded PDF | Retrieved passages; generated answer only with a configured LLM |
| 10. Memory | Ask a follow-up using the same session | Recent chat turns carried into the next prompt; check session history |
| 11. Persistence | Restart the service with the same `runtime_data/` | FAISS indexes and SQLite memory remain available |

**Suggested sample dataset:** one technical PDF about autonomous robots and at least three related images (robot, car, wind turbine). Indexing one image only is not sufficient for a meaningful image-to-image recommendation demonstration.

**Demo caveat:** The current UI's recommendation results appear under the text-search tab. The original research UI screenshots under `docs/` may show the older PDF-only prototype, not the new UI.

## 9. FastAPI endpoint reference and CMD examples

### Endpoints

| Method | Endpoint | Input / output |
|---|---|---|
| `GET` | `/health` | Service health and index counts |
| `POST` | `/documents/pdf` | Multipart `file` → PDF ID, chunks added |
| `POST` | `/documents/image` | Multipart `file`, optional `caption` → image ID |
| `POST` | `/search/text` | JSON `query`, `top_k`, `modality` → ranked results |
| `POST` | `/search/image` | Multipart query image, `top_k`, `modality` → results |
| `GET` | `/recommend/{item_id}` | Similar indexed CLIP items |
| `POST` | `/chat` | JSON `question`, optional `session_id`, `top_k` → answer/sources |
| `GET` | `/sessions/{session_id}` | Session's saved messages |
| `DELETE` | `/sessions/{session_id}` | Clear a session's saved messages |
| `GET` | `/assets/{item_id}` | Image PNG preview |

Search modalities: `all`, `text`, `image`. The default upload limit is **20 MB per file** (configure with `MAX_UPLOAD_MB` in the process environment).

**Easiest option:** Open **http://127.0.0.1:8000/docs**. Expand any endpoint → **Try it out** → fill the fields → **Execute**.

### Optional Windows CMD cURL examples

**A. Health check**

~~~bat
curl http://127.0.0.1:8000/health
~~~

**B. Upload the included sample PDF (from repository root)**

~~~bat
curl -X POST http://127.0.0.1:8000/documents/pdf -F "file=@data\pdfs\sample.pdf"
~~~

**C. Upload an image** (replace the example path with a real image on your PC)

~~~bat
curl -X POST http://127.0.0.1:8000/documents/image -F "file=@C:\path\to\your\image.jpg" -F "caption=Research image"
~~~

**D. Text semantic search**

~~~bat
curl -X POST http://127.0.0.1:8000/search/text -H "Content-Type: application/json" -d "{\"query\":\"wind turbine\",\"top_k\":5,\"modality\":\"all\"}"
~~~

**E. Image → PDF text**

~~~bat
curl -X POST http://127.0.0.1:8000/search/image -F "file=@C:\path\to\your\image.jpg" -F "modality=text" -F "top_k=5"
~~~

**F. Similar item recommendation** (replace with an actual `id` from a search result)

~~~bat
curl "http://127.0.0.1:8000/recommend/img:REPLACE_WITH_ACTUAL_ID?top_k=5"
~~~

**G. PDF-grounded chat**

~~~bat
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"Explain the method described in my uploaded PDF\",\"session_id\":\"demo_session_1\",\"top_k\":4}"
~~~

**H. View the same session's chat history**

~~~bat
curl http://127.0.0.1:8000/sessions/demo_session_1
~~~

## 10. Testing and verification

### Automated tests

Run from the project root after dependency installation:

~~~bat
python -m pytest -q tests\test_platform.py
~~~

Tests use small **fake encoders** and a **fake PDF reader**, so they do not download real CLIP/MiniLM weights or call a paid LLM. The test suite exercises:

- PDF upload and duplicate detection.
- FAISS text search.
- Text → image and image → text API paths.
- Similarity recommendations excluding the queried item.
- Image preview response.
- Session-isolated memory, session deletion and retrieval-only fallback.
- Invalid requests and text-chunk boundaries.

GitHub Actions automatically runs these tests on push/pull request: [CI workflow](https://github.com/Akhil-2024/multimodal-research-assistant/actions/workflows/tests.yml).

**Verification note:** Passing mocked tests establishes API/control-flow behavior; it **does not** prove real CLIP semantic relevance, model-download success on every machine, RAG factual accuracy, or production performance.

### Real-model acceptance checklist (not yet fully verified)

- [ ] Index an actual PDF and 3+ real images locally.
- [ ] Verify a text query returns semantically appropriate PDF chunks.
- [ ] Verify a descriptive text query returns related images.
- [ ] Verify an image query retrieves similar images.
- [ ] Verify an image query retrieves related PDF chunks.
- [ ] Check recommendations exclude the selected item.
- [ ] Check a follow-up RAG question and confirm source pages.
- [ ] Test restart persistence and inspect failure cases.
- [ ] Measure retrieval Recall@K/MRR and response latency on a labeled dataset.

**Do not add accuracy, speedup, or scale figures to the résumé until these are actually measured.**

## 11. Optional Docker setup

Docker files are included for reproducibility, but **you can skip this entire section**. On some internet connections, downloading CPU PyTorch in a Docker build can be slow.

For **Windows CMD**, from the repository root:

~~~bat
copy .env.example .env
docker compose up --build
~~~

- API: http://localhost:8000/docs
- UI: http://localhost:8501
- Health: http://localhost:8000/health

Docker Compose uses the local `.env` file for variables including `OPENAI_API_KEY`. **Never commit `.env`**. Stop with **Ctrl+C**; optionally run `docker compose down` to stop containers.

This project is currently designed for a **single local API process**, not a multi-worker distributed FAISS deployment.

## 12. Troubleshooting

| Problem | Why | Solution |
|---|---|---|
| `'Copy-Item' is not recognized` in CMD | PowerShell command used in CMD | Use `copy .env.example .env` |
| `cd "$HOME\Desktop"` fails in CMD | PowerShell variable syntax | Use `cd /d "%USERPROFILE%\Desktop"` |
| `py -3.11` not found | Python 3.11 missing | Install Python 3.11; verify with `py -3.11 --version` |
| `ModuleNotFoundError` | Wrong interpreter/environment | Activate `.venv` and install `requirements.txt` |
| UI says API unavailable | Backend not running | Start `python -m uvicorn backend.api:app` first |
| First query/indexing is slow | Model checkpoints download on first use | Ensure internet access and adequate free disk/RAM |
| Chat returns passages, not a generated answer | `OPENAI_API_KEY` absent in backend environment | Set the key in backend CMD before starting Uvicorn |
| "No extractable text in PDF" | Scanned/image-only PDF | Use a selectable-text PDF; OCR is a future feature |
| No search hits | No files ingested or modality filter excludes hits | Ingest PDF/images, then check `/health` |
| Port 8000 already in use | Another process occupies port | Stop the other service or use another Uvicorn port and update UI `API_URL` |
| Docker `Temporary failure in name resolution` | Docker networking/DNS issue | Prefer local Python setup; inspect Docker Desktop DNS/proxy settings only if Docker is required |

## 13. Limitations, security and remaining work

### Current limitations (be explicit)

- **Prototype, not deployed production service.** FastAPI gives reusable inference routes, but "scalable" in this project means a modular architecture; horizontal scaling, distributed indexing and production load testing are **not** implemented.
- **No built-in authentication, tenant isolation, rate limits or HTTPS termination.** Use for trusted local demos, not a public service or private/confidential uploads.
- **PDFs need extractable text.** OCR for scanned documents is not included.
- **CLIP has a short text context.** Only the beginning of each PDF passage is used for its CLIP representation; this may reduce image-to-PDF retrieval quality.
- **FAISS IndexFlatIP** is an exact flat index. Large data collections may require ANN indexes and optimized storage.
- **Image content is used for retrieval**, not for generating an image-grounded multimodal LLM answer. RAG chat draws its context from PDF text.
- **LLM answers are not guaranteed correct.** Retrieved source/page references should be manually checked.
- **Local persistence:** SQLite stores messages; FAISS index files/metadata/images are stored under `runtime_data/`. No robust multi-process/transactional index update system is implemented.
- Some **legacy prototype cache files** were committed previously. They are distinct from the new runtime storage. Do not put credentials or sensitive data into any tracked files.

### Priority next work (not necessary to claim core feature implementation)

1. **Highest value:** Real-model smoke test and small manually labeled retrieval evaluation, including screenshots.
2. **Quality:** Better PDF chunking, retrieval reranking, evaluation of image-text matches and answer citations.
3. **Coverage:** OCR for scanned PDFs; captioning for uploaded images; hybrid sparse+dense retrieval.
4. **Robustness/security:** Authentication, access separation, rate limits, safer storage and multi-worker concurrency.
5. **Optional deployment:** Only after local behavior, correctness, and security have been validated.

## 14. Resume and interview explanation

### Resume-aligned summary

**Multimodal Semantic Search, Recommendation & RAG Platform | PyTorch, CLIP, FAISS, FastAPI**

- Developed a research-assistant prototype combining CLIP cross-modal embeddings, Sentence Transformers, and FAISS-based semantic retrieval over uploaded PDFs and images.
- Implemented text-to-image, image-to-image, and image-to-text retrieval, plus nearest-neighbor content recommendations using normalized embeddings.
- Exposed upload/search/recommendation/RAG services through FastAPI, with a Streamlit client, PDF source metadata, and SQLite-backed conversational session memory.

**Be precise in interviews:** The core implementation exists, automated tests pass with mocked embeddings, and Docker packaging is included but deployment was deliberately skipped. Do not claim validated large-scale throughput, deployed infrastructure, trained/fine-tuned CLIP, or measured retrieval accuracy without evidence.

### 60-second interview walkthrough

> "I started with a multi-PDF semantic research assistant. I extracted and chunked PDF text, generated SentenceTransformer embeddings, and indexed them in FAISS. Then I extended it with a CLIP embedding index that stores both image features and PDF text representations, enabling cross-modal retrieval. For mixed text search, I use separate MiniLM and CLIP rankings combined with reciprocal rank fusion because their scores are not directly comparable. A recommendation endpoint finds nearest CLIP neighbors. The RAG endpoint retrieves relevant PDF chunks and optionally passes them to an LLM with page metadata and recent SQLite chat history. The system is exposed via FastAPI with a Streamlit client. I added automated API tests; the next step is an empirical real-model retrieval benchmark." 

### Five common interview questions

**Q1. Why both MiniLM and CLIP?**  
MiniLM specializes in text semantics for PDF question retrieval; CLIP supports image/text similarity in a shared embedding space.

**Q2. Why FAISS and what index did you use?**  
FAISS provides vector similarity search. I used `IndexFlatIP` with L2-normalized vectors, so inner product corresponds to cosine similarity.

**Q3. How is this RAG rather than simple LLM chat?**  
The system first retrieves source PDF passages and then provides them to the LLM as supporting context, returning the retrieved citations alongside the answer.

**Q4. How are recommendations generated?**  
For an indexed item, retrieve its CLIP embedding, search for nearest neighbors, and exclude the item itself.

**Q5. What did you do to make follow-up conversations possible?**  
I persist user/assistant turns in a SQLite table keyed by session ID and include a recent window when generating later responses.

---

**Repository:** https://github.com/Akhil-2024/multimodal-research-assistant  
**Primary application:** `backend/platform.py` + `backend/api.py` + `frontend/multimodal_app.py`  
**Fastest starting point:** [Section 7 — Windows CMD, no Docker](#7-step-by-step-setup-on-windows-cmd-no-docker)  
**Author:** Akhilesh Kumar Patel · M.Tech Control & Automation, IIT Delhi
