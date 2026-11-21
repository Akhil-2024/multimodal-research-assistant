"""Offline integration tests: encoder weights and LLM calls are not downloaded."""
import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.platform as platform
from backend.api import app, get_service


class FakeEncoder:
    def encode(self, items, **kwargs):
        vectors = []
        for item in items:
            if isinstance(item, Image.Image):
                vectors.append([1.0, 0.2, 0.1, 0.3])
            else:
                content = str(item).lower()
                vectors.append([1.0 if "cat" in content else 0.3,
                                0.7 if "wind" in content else 0.2, 0.1, 0.3])
        return np.asarray(vectors, dtype=np.float32)


class FakePage:
    def extract_text(self):
        return "A cat uses visual features to recognize related images in this document."


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "SentenceTransformer", lambda name: FakeEncoder())
    monkeypatch.setattr(
        platform, "PdfReader", lambda source: SimpleNamespace(pages=[FakePage()])
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = platform.MultimodalService(tmp_path)
    app.dependency_overrides[get_service] = lambda: service
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_pdf_ingest_search_and_dedup(client):
    first = client.post(
        "/documents/pdf", files={"file": ("paper.pdf", b"not-a-real-pdf", "application/pdf")}
    )
    assert first.status_code == 200
    assert first.json()["chunks_added"] == 1
    second = client.post(
        "/documents/pdf", files={"file": ("paper.pdf", b"not-a-real-pdf", "application/pdf")}
    )
    assert second.json()["chunks_added"] == 0
    result = client.post(
        "/search/text", json={"query": "cat", "modality": "text", "top_k": 3}
    )
    assert result.status_code == 200
    assert result.json()["results"][0]["page"] == 1


def test_crossmodal_search_recommendation_and_image_asset(client):
    client.post(
        "/documents/pdf", files={"file": ("paper.pdf", b"test-pdf", "application/pdf")}
    )
    uploaded = client.post(
        "/documents/image", files={"file": ("cat.png", image_bytes(), "image/png")},
        data={"caption": "A red square"}
    )
    assert uploaded.status_code == 200
    item_id = uploaded.json()["item_id"]
    found = client.post("/search/text", json={"query": "cat", "modality": "image"})
    assert any(hit["id"] == item_id for hit in found.json()["results"])
    reverse = client.post(
        "/search/image", files={"file": ("query.png", image_bytes(), "image/png")},
        data={"modality": "text"}
    )
    assert reverse.status_code == 200
    assert reverse.json()["results"][0]["kind"] == "text"
    recs = client.get(f"/recommend/{item_id}")
    assert recs.status_code == 200
    assert all(hit["id"] != item_id for hit in recs.json()["results"])
    asset = client.get(f"/assets/{item_id}")
    assert asset.status_code == 200 and asset.headers["content-type"] == "image/png"


def test_chat_memory_is_session_scoped_and_falls_back_without_key(client):
    client.post(
        "/documents/pdf", files={"file": ("paper.pdf", b"test-pdf", "application/pdf")}
    )
    response = client.post("/chat", json={"question": "How does the cat recognize images?"})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "retrieval_only"
    history = client.get(f"/sessions/{data['session_id']}").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert client.get("/sessions/another-session").json()["messages"] == []
    assert client.delete(f"/sessions/{data['session_id']}").status_code == 200
    assert client.get(f"/sessions/{data['session_id']}").json()["messages"] == []


def test_invalid_inputs(client):
    assert client.post("/search/text", json={"query": ""}).status_code == 422
    assert client.post("/documents/pdf", files={"file": ("wrong.txt", b"abc")}).status_code == 400
    assert client.post(
        "/documents/image", files={"file": ("broken.png", b"not-an-image")}
    ).status_code == 400
    assert client.post(
        "/chat", json={"question": "hello", "session_id": "../invalid"}
    ).status_code == 400
    assert client.get("/recommend/unknown").status_code == 404


def test_chunking_boundaries_and_rank_fusion():
    assert platform.chunk_text("abc", size=5, overlap=2) == ["abc"]
    assert platform.chunk_text("abcdefghij", size=6, overlap=2) == ["abcdef", "efghij"]
    a = [{"id": "a"}, {"id": "b"}]
    b = [{"id": "b"}, {"id": "c"}]
    assert platform.reciprocal_rank_fusion([a, b], 3)[0]["id"] == "b"
