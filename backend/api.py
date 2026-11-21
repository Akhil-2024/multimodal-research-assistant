"""FastAPI inference layer. Interactive docs available at /docs."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.platform import MultimodalService

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
app = FastAPI(
    title="Multimodal Semantic Search, Recommendation & RAG",
    version="1.0.0",
    description="CLIP image/text retrieval, SentenceTransformer PDF search, FAISS recommendations, and grounded RAG.",
)


@lru_cache(maxsize=1)
def get_service() -> MultimodalService:
    return MultimodalService()


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=30)
    modality: Literal["all", "text", "image"] = "all"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=80)
    top_k: int = Field(default=4, ge=1, le=15)


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    return data


def bad_input(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
def health(service: MultimodalService = Depends(get_service)):
    return {"status": "ok", **service.stats()}


@app.post("/documents/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    service: MultimodalService = Depends(get_service),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Expected a .pdf file")
    data = await read_upload(file)
    try:
        return service.ingest_pdf(file.filename or "document.pdf", data)
    except ValueError as exc:
        raise bad_input(exc) from exc


@app.post("/documents/image")
async def upload_image(
    file: UploadFile = File(...),
    caption: str = Form(default="", max_length=1000),
    service: MultimodalService = Depends(get_service),
):
    data = await read_upload(file)
    try:
        return service.ingest_image(file.filename or "image", data, caption)
    except ValueError as exc:
        raise bad_input(exc) from exc


@app.post("/search/text")
def search_text(
    body: TextSearchRequest,
    service: MultimodalService = Depends(get_service),
):
    try:
        results = service.search_text(body.query, body.top_k, body.modality)
        return {"query": body.query, "results": results, "count": len(results)}
    except ValueError as exc:
        raise bad_input(exc) from exc


@app.post("/search/image")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=5, ge=1, le=30),
    modality: Literal["all", "text", "image"] = Form(default="all"),
    service: MultimodalService = Depends(get_service),
):
    data = await read_upload(file)
    try:
        results = service.search_image(data, top_k, modality)
        return {"results": results, "count": len(results)}
    except ValueError as exc:
        raise bad_input(exc) from exc


@app.get("/recommend/{item_id:path}")
def recommend(
    item_id: str,
    top_k: int = Query(default=5, ge=1, le=30),
    service: MultimodalService = Depends(get_service),
):
    try:
        results = service.recommend(item_id, top_k)
        return {"item_id": item_id, "results": results, "count": len(results)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/chat")
def chat(
    body: ChatRequest,
    service: MultimodalService = Depends(get_service),
):
    try:
        return service.chat(body.question, body.session_id, body.top_k)
    except ValueError as exc:
        raise bad_input(exc) from exc


@app.get("/sessions/{session_id}")
def get_session(session_id: str, service: MultimodalService = Depends(get_service)):
    try:
        return {"session_id": session_id, "messages": service.history(session_id, limit=100)}
    except ValueError as exc:
        raise bad_input(exc) from exc


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, service: MultimodalService = Depends(get_service)):
    try:
        service.clear_history(session_id)
        return {"session_id": session_id, "cleared": True}
    except ValueError as exc:
        raise bad_input(exc) from exc


@app.get("/assets/{item_id:path}")
def image_asset(item_id: str, service: MultimodalService = Depends(get_service)):
    try:
        return FileResponse(service.image_path(item_id), media_type="image/png")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc
