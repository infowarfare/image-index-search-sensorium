import pytest
import base64
import io
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def mock_app_state(client, tmp_path):
    """Mockt Pipeline und Document Store im app.state."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    fake_image = img_dir / "apple.jpg"
    fake_image.write_bytes(b"fakeimagebytes")

    store = MagicMock()
    store.count_documents.return_value = 5

    search_pipeline = MagicMock()
    search_pipeline.run_async = AsyncMock(return_value={
        "retriever": {
            "documents": [
                MagicMock(score=0.85, meta={"file_path": str(fake_image)})
            ]
        }
    })

    indexing_pipeline = MagicMock()
    indexing_pipeline.run_async = AsyncMock(return_value={})

    client.app.state.doc_store = store
    client.app.state.search_pipeline = search_pipeline
    client.app.state.indexing_pipeline = indexing_pipeline

    with patch("main.IMAGE_DIR", img_dir):
        yield


def create_test_image() -> bytes:
    """Erstellt ein minimales Test-Bild."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestStats:
    def test_returns_200(self, client):
        response = client.get("/stats")
        assert response.status_code == 200

    def test_returns_indexed_images_key(self, client):
        assert "indexed_images" in client.get("/stats").json()

    def test_returns_integer(self, client):
        assert isinstance(client.get("/stats").json()["indexed_images"], int)


class TestSearch:
    def test_returns_200(self, client):
        response = client.post("/search", json={"query": "Apfel", "top_k": 1})
        assert response.status_code == 200

    def test_returns_list(self, client):
        response = client.post("/search", json={"query": "Apfel", "top_k": 1})
        assert isinstance(response.json(), list)

    def test_result_has_required_fields(self, client):
        response = client.post("/search", json={"query": "Apfel", "top_k": 1})
        result = response.json()[0]
        assert "filename" in result
        assert "score" in result
        assert "image_base64" in result

    def test_score_is_float(self, client):
        response = client.post("/search", json={"query": "Apfel", "top_k": 1})
        assert isinstance(response.json()[0]["score"], float)

    def test_missing_query_fails(self, client):
        response = client.post("/search", json={})
        assert response.status_code == 422

    def test_image_base64_is_valid(self, client):
        response = client.post("/search", json={"query": "Apfel", "top_k": 1})
        image_b64 = response.json()[0]["image_base64"]
        decoded = base64.b64decode(image_b64)
        assert len(decoded) > 0


class TestIndex:
    def test_returns_202(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/index",
            files={"files": ("test.jpg", image_bytes, "image/jpeg")}
        )
        assert response.status_code == 202

    def test_returns_message(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/index",
            files={"files": ("test.jpg", image_bytes, "image/jpeg")}
        )
        assert "message" in response.json()

    def test_multiple_images(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/index",
            files=[
                ("files", ("img1.jpg", image_bytes, "image/jpeg")),
                ("files", ("img2.jpg", image_bytes, "image/jpeg")),
                ("files", ("img3.jpg", image_bytes, "image/jpeg")),
            ]
        )
        assert response.status_code == 202

    def test_message_contains_correct_count(self, client):
        image_bytes = create_test_image()
        response = client.post(
            "/index",
            files=[
                ("files", ("img1.jpg", image_bytes, "image/jpeg")),
                ("files", ("img2.jpg", image_bytes, "image/jpeg")),
            ]
        )
        assert "2" in response.json()["message"]

    def test_no_files_fails(self, client):
        response = client.post("/index")
        assert response.status_code == 422

    def test_pipeline_was_called(self, client, tmp_path):
        image_bytes = create_test_image()
        with patch("main.IMAGE_DIR", tmp_path / "images"):
            response = client.post(
                "/index",
                files={"files": ("test.jpg", image_bytes, "image/jpeg")}
            )
        # 202 = Accepted, Background Task wurde gestartet
        assert response.status_code == 202
        # Datei wurde auf Disk gespeichert
        assert (tmp_path / "images" / "test.jpg").exists()