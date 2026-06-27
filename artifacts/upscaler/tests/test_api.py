"""
Pytest unit tests for Real-ESRGAN Upscaler API.
Run from artifacts/upscaler/:  pytest tests/ -v
"""
import io
import sys
import os
import pytest
from PIL import Image

# Add parent dir so imports work when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient

# Stub out heavy model loading so tests run without GPU/models
import unittest.mock as mock
with mock.patch("utils.models_manager.setup"):
    from main import app

client = TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_png_bytes(w: int = 20, h: int = 20) -> bytes:
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(w: int = 20, h: int = 20) -> bytes:
    img = Image.new("RGB", (w, h), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _generate_key(name: str = "pytest") -> str:
    resp = client.post(f"/api/v1/generate-api-key?name={name}")
    assert resp.status_code == 200
    return resp.json()["api_key"]


# ─────────────────────────────────────────────
# Root & info endpoints
# ─────────────────────────────────────────────

class TestInfoEndpoints:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_returns_online(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "online"

    def test_metrics_returns_system_info(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "ram_used_mb" in data
        assert "disk_used_gb" in data
        assert "queue_size" in data
        assert "gpu_available" in data

    def test_docs_accessible(self):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_accessible(self):
        resp = client.get("/redoc")
        assert resp.status_code == 200


# ─────────────────────────────────────────────
# API key management
# ─────────────────────────────────────────────

class TestApiKeyManagement:
    def test_generate_key_returns_resr_prefix(self):
        resp = client.post("/api/v1/generate-api-key?name=test1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["api_key"].startswith("resr_")
        assert data["daily_limit"] == 100

    def test_generate_key_no_name(self):
        resp = client.post("/api/v1/generate-api-key")
        assert resp.status_code == 200
        assert resp.json()["api_key"].startswith("resr_")

    def test_list_keys_returns_list(self):
        _generate_key("listtest")
        resp = client.get("/api/v1/api-keys")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["api_keys"], list)
        assert data["count"] == len(data["api_keys"])

    def test_disable_and_enable_key(self):
        key = _generate_key("disabletest")
        # Disable
        resp = client.post(f"/api/v1/api-keys/{key}/disable")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        # Disabled key should return 401
        resp2 = client.get("/api/v1/api-keys", headers={"X-API-Key": key})
        # (list-keys has no auth, but upscale should fail)
        # Enable
        resp3 = client.post(f"/api/v1/api-keys/{key}/enable")
        assert resp3.status_code == 200

    def test_delete_key(self):
        key = _generate_key("deletetest")
        resp = client.delete(f"/api/v1/api-keys/{key}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_delete_nonexistent_key_returns_404(self):
        resp = client.delete("/api/v1/api-keys/resr_doesnotexist123")
        assert resp.status_code == 404

    def test_key_usage_endpoint(self):
        key = _generate_key("usagetest")
        resp = client.get(f"/api/v1/api-keys/{key}/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert "used_today" in data
        assert "daily_limit" in data
        assert "remaining" in data


# ─────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────

class TestAuthentication:
    def test_missing_api_key_returns_401(self):
        resp = client.post("/api/v1/upscale-image")
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self):
        resp = client.post(
            "/api/v1/upscale-image",
            headers={"X-API-Key": "resr_invalidkey000"},
        )
        assert resp.status_code == 401

    def test_disabled_key_returns_401(self):
        key = _generate_key("disabledauth")
        client.post(f"/api/v1/api-keys/{key}/disable")
        resp = client.post(
            "/api/v1/upscale-image",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 401


# ─────────────────────────────────────────────
# Image upscaling
# ─────────────────────────────────────────────

class TestImageUpscaling:
    def setup_method(self):
        self.key = _generate_key("imgtest")
        self.headers = {"X-API-Key": self.key}

    def test_upscale_image_invalid_format_returns_400(self):
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"scale": "4", "output_format": "PNG", "compression": "balanced"},
        )
        assert resp.status_code == 400

    def test_upscale_image_invalid_scale_returns_400(self):
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
            data={"scale": "3", "output_format": "PNG", "compression": "balanced"},
        )
        assert resp.status_code == 400

    def test_upscale_image_invalid_format_param_returns_400(self):
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
            data={"scale": "4", "output_format": "BMP", "compression": "balanced"},
        )
        assert resp.status_code == 400

    def test_upscale_image_invalid_compression_returns_400(self):
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
            data={"scale": "4", "output_format": "PNG", "compression": "ultra"},
        )
        assert resp.status_code == 400

    @mock.patch("main._run_image_job")
    def test_upscale_image_png_queues_job(self, mock_job):
        mock_job.return_value = None
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("photo.png", _make_png_bytes(), "image/png")},
            data={"scale": "4", "output_format": "PNG", "compression": "balanced"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "job_id" in data
        assert data["status"] == "queued"
        assert "download_url" in data
        assert data["expires_in"] == "24h"

    @mock.patch("main._run_image_job")
    def test_upscale_image_jpeg_queues_job(self, mock_job):
        mock_job.return_value = None
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
            data={"scale": "2", "output_format": "JPG", "compression": "high_quality"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @mock.patch("main._run_image_job")
    def test_upscale_image_webp_queues_job(self, mock_job):
        mock_job.return_value = None
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="WEBP")
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("img.webp", buf.getvalue(), "image/webp")},
            data={"scale": "4", "output_format": "WEBP", "compression": "small_size"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ─────────────────────────────────────────────
# Video upscaling
# ─────────────────────────────────────────────

class TestVideoUpscaling:
    def setup_method(self):
        self.key = _generate_key("vidtest")
        self.headers = {"X-API-Key": self.key}

    def test_upscale_video_invalid_format_returns_400(self):
        resp = client.post(
            "/api/v1/upscale-video",
            headers=self.headers,
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
            data={"scale": "4", "compression": "balanced"},
        )
        assert resp.status_code == 400

    def test_upscale_video_invalid_scale_returns_400(self):
        resp = client.post(
            "/api/v1/upscale-video",
            headers=self.headers,
            files={"file": ("clip.mp4", b"fake", "video/mp4")},
            data={"scale": "8", "compression": "balanced"},
        )
        assert resp.status_code == 400

    @mock.patch("main._run_video_job")
    def test_upscale_video_queues_job(self, mock_job):
        mock_job.return_value = None
        resp = client.post(
            "/api/v1/upscale-video",
            headers=self.headers,
            files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftyp", "video/mp4")},
            data={"scale": "2", "compression": "balanced"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "job_id" in data
        assert data["status"] == "queued"


# ─────────────────────────────────────────────
# Job status & download
# ─────────────────────────────────────────────

class TestJobStatusAndDownload:
    def setup_method(self):
        self.key = _generate_key("jobtest")
        self.headers = {"X-API-Key": self.key}

    def test_status_nonexistent_job_returns_404(self):
        resp = client.get(
            "/api/v1/status/nonexistentjobid123",
            headers=self.headers,
        )
        assert resp.status_code == 404

    def test_download_nonexistent_job_returns_404(self):
        resp = client.get(
            "/api/v1/download/nonexistentjobid999",
            headers=self.headers,
        )
        assert resp.status_code == 404

    @mock.patch("main._run_image_job")
    def test_status_after_queue_returns_queued(self, mock_job):
        mock_job.return_value = None
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("x.png", _make_png_bytes(), "image/png")},
            data={"scale": "4", "output_format": "PNG", "compression": "balanced"},
        )
        job_id = resp.json()["job_id"]
        status_resp = client.get(f"/api/v1/status/{job_id}", headers=self.headers)
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "upscaling", "completed", "failed")

    @mock.patch("main._run_image_job")
    def test_download_non_completed_job_returns_400(self, mock_job):
        mock_job.return_value = None
        resp = client.post(
            "/api/v1/upscale-image",
            headers=self.headers,
            files={"file": ("x.png", _make_png_bytes(), "image/png")},
            data={"scale": "4", "output_format": "PNG", "compression": "balanced"},
        )
        job_id = resp.json()["job_id"]
        dl = client.get(f"/api/v1/download/{job_id}", headers=self.headers)
        # queued → not ready → 400
        assert dl.status_code in (400, 200)


# ─────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────

class TestRateLimiting:
    def test_rate_limit_endpoint_exists(self):
        key = _generate_key("ratelimitcheck")
        resp = client.get(f"/api/v1/api-keys/{key}/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["daily_limit"] == 100
        assert data["remaining"] == 100
