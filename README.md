# Multimodal Semantic Search, Recommendation & RAG Platform

A portfolio-ready extension of the original **Multi-PDF Research Assistant**. It integrates **PyTorch-backed CLIP**, **Sentence Transformers**, **FAISS**, **FastAPI**, **Streamlit**, **SQLite session memory**, and **OpenAI-compatible LLM generation**.

The original PDF scripts and the original Streamlit UI in frontend/app.py remain in the repository. The **new** API-based application is in backend/platform.py, backend/api.py and frontend/multimodal_app.py.

## What it does

| Capability | Implementation |
| --- | --- |
| PDF ingestion | Extract selectable PDF text with pypdf; create overlapping page-aware chunks |
| Semantic PDF search | all-MiniLM-L6-v2 embeddings, cosine similarity via FAISS IndexFlatIP |
| Text to image search | CLIP text query embedded in the same vector space as uploaded images |
| Image to image search | CLIP image embedding and FAISS cosine nearest neighbors |
| Image to PDF text search | CLIP image query retrieves CLIP-embedded text chunks |
| Text to mixed results | Search MiniLM text index and CLIP image index; merge **ranks** with reciprocal rank fusion |
| Content recommendations | Reconstruct an indexed CLIP vector and find nearby items, excluding itself |
| Grounded PDF RAG | Retrieve source passages and provide them to an LLM with source/page attribution |
| Chat memory | SQLite messages isolated by session ID, retrieved for follow-up conversations |
| Inference API | FastAPI with upload, search, recommend, RAG, session and health routes |
| UI and Docker | Separate Streamlit API client, Dockerfile and Docker Compose |

### Architecture

~~~text
                           Streamlit client / curl / Swagger
                                       |
                                       v
                              FastAPI (backend/api.py)
                                       |
                           MultimodalService (platform.py)
                +----------------------+---------------------+
                |                      |                     |
          PDF extraction          Image ingestion       Session history
          pypdf + chunks          Pillow                 SQLite
                |                      |
                v                      v
       SentenceTransformer       CLIP image encoder
          MiniLM (384-d)             (512-d)
                |                      |
          text.faiss <----+      clip.faiss <--- CLIP text embeddings
                |         |           |          from PDF chunks
                +---------+-----------+
                     retrieval / recommendation
                                 |
                          context + sources
                                 |
                           optional OpenAI LLM
                                 |
                       answer + page citations
~~~

**Important:** MiniLM vectors and CLIP vectors are **not** directly compared. Each index uses normalized embeddings and inner-product cosine search. Mixed text search uses reciprocal rank fusion to combine the rankings, not scores from different models.

## Quick start: Docker (recommended)

Prerequisites: Docker Desktop with Linux containers enabled, internet access for downloading model weights, and enough RAM/disk for CPU PyTorch + CLIP.

~~~bash
git clone https://github.com/Akhil-2024/multimodal-research-assistant.git
cd multimodal-research-assistant
git checkout feature/multimodal-rag-fastapi
cp .env.example .env
docker compose up --build
~~~

On **Windows PowerShell**, replace the copy line with:

~~~powershell
Copy-Item .env.example .env
~~~

Open:
- **Streamlit UI:** http://localhost:8501
- **FastAPI Swagger docs:** http://localhost:8000/docs
- **API health:** http://localhost:8000/health

The first embedding request downloads the two public model checkpoints via Sentence Transformers. It runs on CPU by default; a GPU is optional.

To enable generative answers, edit the local **.env** file and set **OPENAI_API_KEY** to your own key, then restart Compose. Without it, chat works in **retrieval-only mode**, returning cited passages instead of pretending an LLM ran.

## Quick start: without Docker

Use Python 3.11:

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m uvicorn backend.api:app --reload
~~~

In a **second** terminal run:

~~~bash
python -m streamlit run frontend/multimodal_app.py
~~~

On Windows, activate the environment with **.venv\Scripts\activate** in Command Prompt or **.venv\Scripts\Activate.ps1** in PowerShell (subject to your execution policy). Run both commands from the repository root. You can alternatively use the browser-based Swagger API at http://localhost:8000/docs.

## Demo: build, search, recommend, chat

1. In the UI's **Ingest files** tab, upload one or more PDFs and 2-3 images (e.g. robot / car / turbine photos).
2. Open **Text search** and ask for "wind turbine" with target **all**. You'll get PDF snippets and relevant images if indexed.
3. Under any result, click **Recommend similar** to find neighboring CLIP items.
4. In **Search by image**, upload a query photo and choose **image** for visually similar uploads, or **text** to retrieve conceptually related PDF passages.
5. In **PDF RAG chat**, ask a question about uploaded PDFs, then a follow-up. Results include filenames, page numbers, similarity scores and persisted session history.

You can also use Swagger at **/docs** to upload files, then run all HTTP endpoints manually.

### API reference

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /health | Index counts and LLM configuration status |
| POST | /documents/pdf | Multipart PDF upload and page-aware indexing |
| POST | /documents/image | Multipart image upload with optional caption |
| POST | /search/text | JSON: query, top_k, modality = all/text/image |
| POST | /search/image | Multipart query image, top_k, modality |
| GET | /recommend/{item_id} | Retrieve related items using CLIP |
| POST | /chat | JSON: question, session_id (optional), top_k |
| GET | /sessions/{session_id} | Get session-scoped conversation history |
| DELETE | /sessions/{session_id} | Erase that session's conversation |
| GET | /assets/{item_id} | Stream indexed image as PNG |

The API exposes source name, PDF page, similarity score, embedding space, and a stable item ID. For image hits, **image_url** points to the preview. File uploads are capped at **20 MB** by default (configurable with MAX_UPLOAD_MB).

## Technical decisions and limitations

- **Vector store:** FAISS IndexFlatIP performs exhaustive exact nearest-neighbor search. It is suitable for local portfolios and modest collections; large-scale search needs approximate indexes or a distributed vector database.
- **Model inference:** CPU models are loaded lazily. The initial query may be slower. This is a single-instance prototype, not a throughput benchmark or horizontally scalable production deployment.
- **PDF parsing:** pypdf only handles selectable text; scanned pages require OCR (not implemented). CLIP text is truncated for its short input context.
- **RAG:** A retrieved passage is *evidence*, not proof of factual correctness. The LLM is instructed to abstain if unsupported and to cite source filenames and pages. Retrieval scores are **not** calibrated answer-confidence probabilities.
- **Durability:** Generated indices and images live in runtime_data/ (mounted as a Docker volume). SQLite stores per-session messages. Avoid multiple workers writing the same FAISS files until storage transactions are redesigned.
- **Security:** This is a local demo with no authentication, user quotas or network rate limiting. Do **not** expose it to the public internet or upload confidential material without adding authentication, access controls and TLS.
- **Old tracked files:** Earlier prototype files under backend/ and frontend/cache/ include generated caches and chat state. New .gitignore rules prevent *new untracked* runtime artifacts; existing tracked files must be separately removed from Git history if they contain private data.

## Tests

~~~bash
python -m pytest -q tests/test_platform.py
~~~

Tests use mocked embedding models, a fake PDF reader and no LLM calls, to check PDF deduplication, both cross-modal directions, recommendations, image previews, validation and session isolation. These tests do not measure real retrieval accuracy. A GitHub Actions workflow runs the test suite.

## Next engineering improvements

1. Add a labeled multimodal retrieval benchmark (Recall@K / MRR / latency p95) and inspect failure cases.
2. Add authentication, per-user indexes, input sanitization and rate limits.
3. Use ANN indexing for large datasets, transactional metadata storage, background ingestion queues and index locking.
4. Evaluate hybrid BM25 + dense retrieval, reranking, image captions and OCR for scanned PDFs.
5. Add CI Docker smoke tests, load tests and GPU deployment when needed.

**Original PDF-only Streamlit demo:** frontend/app.py remains available via the old command **streamlit run frontend/app.py**; the new API-connected UI is **frontend/multimodal_app.py**.
