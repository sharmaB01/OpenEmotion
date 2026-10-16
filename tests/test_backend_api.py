import importlib
import os
import sys
import time
import base64
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from openemotion.api.adapter import batch_result
from openemotion.api.app import create_app
from openemotion.api.download import VideoDownloader
from openemotion.api.jobs import JobStore
from openemotion.api.schema import EMOTIONS, BatchResult, StudentVideoAnalysisResult
from openemotion.api.settings import APISettings
from openemotion.config import CLASS_NAMES
from openemotion.realtime.realtime_processor import RealtimeProcessor
from openemotion.service.ai_service import AIService
from openemotion.video.video_processor import VideoProcessor


class Detector:
    def __init__(self):
        self.faces = [{"face_id": 1, "bbox": [0, 0, 32, 32]}]

    def detect_faces(self, frame):
        return self.faces


class FixedModel(torch.nn.Module):
    """Deterministic logits; real decoding, preprocessing and softmax still run."""

    index = CLASS_NAMES.index("surprise")

    def forward(self, tensor):
        output = torch.zeros((tensor.shape[0], 7), device=tensor.device)
        output[:, self.index] = 3
        return output


@pytest.fixture
def settings(tmp_path):
    return APISettings(
        _env_file=None,
        api_key="integration-test-key",
        cache_path=tmp_path / "jobs.sqlite3",
    )


@pytest.fixture
def service():
    service = AIService.__new__(AIService)
    service.device = torch.device("cpu")
    service.model = FixedModel()
    service.detector = Detector()
    service.realtime_processor = RealtimeProcessor(
        service.model, service.detector, service.device
    )
    service.video_processor = VideoProcessor(
        service.model, service.detector, service.device
    )
    return service


@pytest.fixture
def jpeg():
    output = BytesIO()
    Image.new("RGB", (32, 32), "gray").save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
def video_bytes(tmp_path):
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 64))
    assert writer.isOpened(), "OpenCV needs an MP4 encoder for this integration test"
    for _ in range(8):
        writer.write(np.full((64, 64, 3), 128, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def api(settings, service, video_bytes):
    downloads = []

    def serve(request):
        downloads.append(request)
        return httpx.Response(
            200, content=video_bytes, headers={"content-type": "video/mp4"}
        )

    downloader = VideoDownloader(settings, httpx.MockTransport(serve))
    app = create_app(settings, service, downloader)
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client, app=app, downloads=downloads, downloader=downloader
        )


AUTH = {"Authorization": "Bearer integration-test-key"}
STAMP = "2026-09-29T19:00:00+07:00"
VIDEO = {
    "recording_id": "recording-1",
    "video_url": "https://api.cloudinary.com/v1_1/demo/video/download?format=mp4&signature=test",
    "duration": 2.0,
}


def send_frame(client, jpeg, **fields):
    return client.post(
        "/v1/analyze/frame",
        headers=AUTH,
        data={"frame_id": "f-1", "timestamp": STAMP, **fields},
        files={"frame": ("frame", jpeg, "image/jpeg")},
    )


def send_video(client, key="job-1", **fields):
    return client.post(
        "/v1/analyze/video",
        headers={**AUTH, "Idempotency-Key": key},
        json={**VIDEO, **fields},
    )


def send_student_video(client, **fields):
    return client.post(
        "/v1/analyze/student-video",
        headers=AUTH,
        json={**VIDEO, "student_id": "student-1", **fields},
    )


def send_frames(client, jpeg, **fields):
    return client.post(
        "/v1/analyze/frames",
        headers=AUTH,
        json={
            "session_id": "meeting-1",
            "frames": [
                {
                    "frame_id": "f-1",
                    "timestamp": STAMP,
                    "image": base64.b64encode(jpeg).decode("ascii"),
                    "content_type": "image/jpeg",
                    "student_id": "student-1",
                    **fields,
                }
            ],
        },
    )


@pytest.mark.parametrize(
    "model_label,backend_label",
    [("disgust", "disgusted"), ("fear", "fearful"), ("surprise", "surprised")],
)
def test_realtime_model_to_backend(api, jpeg, service, model_label, backend_label):
    service.model.index = CLASS_NAMES.index(model_label)
    response = send_frame(api.client, jpeg)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["frame_id"] == "f-1"
    assert datetime.fromisoformat(result["timestamp"]) == datetime.fromisoformat(STAMP)
    assert result["emotion"] == backend_label
    assert result["face_id"] == "1"
    assert set(result["probabilities"]) == set(EMOTIONS)
    assert result["confidence"] == max(result["probabilities"].values())
    assert result["mock"] is False
    assert "face" not in result and "status" not in result


def test_no_face_is_successful_detection_failure(api, jpeg, service):
    service.detector.faces = []
    response = send_frame(api.client, jpeg)
    assert response.status_code == 200
    result = response.json()
    assert result["emotion"] == "fail_detection"
    assert result["face_detected"] is False
    assert result["failure_reason"] == "no_face"
    assert result["confidence"] == 0 and result["face_id"] is None
    assert result["probabilities"] == dict.fromkeys(EMOTIONS, 0)


def test_batch_frames_preserve_student_mapping(api, jpeg):
    response = send_frames(api.client, jpeg)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["session_id"] == "meeting-1"
    assert result["results"][0]["student_id"] == "student-1"
    frame = result["results"][0]["result"]
    assert frame["frame_id"] == "f-1"
    assert frame["emotion"] == "surprised"
    assert set(frame["probabilities"]) == set(EMOTIONS)


def test_batch_frames_validate_payload(api, jpeg):
    assert send_frames(api.client, jpeg, image="not-base64").status_code == 422
    assert send_frames(api.client, jpeg, timestamp="2026-09-29T00:00:00").status_code == 422


def test_invalid_frames_and_timestamps(api, jpeg):
    assert send_frame(api.client, b"invalid").status_code == 422
    assert (
        send_frame(api.client, jpeg, timestamp="2026-09-29T00:00:00").status_code == 422
    )
    assert send_frame(api.client, jpeg, frame_id="x" * 65).status_code == 422
    assert send_frame(api.client, jpeg, timestamp="broken").status_code == 422
    assert send_frame(api.client, b"a" * (512 * 1024 + 1)).status_code == 413


def test_auth_and_body_limits(api, jpeg):
    response = api.client.post(
        "/v1/analyze/frame",
        files={"frame": ("frame", jpeg, "image/jpeg")},
        data={"frame_id": "f1", "timestamp": STAMP},
    )
    assert response.status_code == 401
    assert (
        api.client.post(
            "/v1/analyze/video", json=VIDEO, headers={"Idempotency-Key": "job"}
        ).status_code
        == 401
    )
    assert (
        api.client.post("/v1/analyze/video", json=VIDEO, headers=AUTH).status_code
        == 422
    )
    assert (
        api.client.post("/v1/analyze/frame", content=b"a" * (600 * 1024)).status_code
        == 413
    )


def test_inference_error_is_not_no_face(api, jpeg, service, monkeypatch):
    def broken(*args):
        raise RuntimeError("internal credentials must not appear")

    monkeypatch.setattr(service.model, "forward", broken)
    response = send_frame(api.client, jpeg)
    assert response.status_code == 500
    assert "credentials" not in response.text


def test_video_counts_timeline_and_no_identity_guess(api, service):
    # Two faces per sampled frame => four samples, not two timeline votes.
    service.detector.faces.append({"face_id": 2, "bbox": [32, 32, 32, 32]})
    response = send_video(api.client)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["sample_counts"] == {"surprised": 4}
    assert [point["timestamp"] for point in result["timeline"]] == [0.0, 0.0, 1.0, 1.0]
    assert all(
        point["confidence"] > 0 and point["student_id"] is None
        for point in result["timeline"]
    )
    assert isinstance(result["summary"], str)
    assert result["mock"] is False


def test_empty_video_analysis(api, service):
    service.detector.faces = []
    response = send_video(api.client)
    assert response.status_code == 200
    assert response.json()["sample_counts"] == {}
    assert response.json()["timeline"] == []


def test_student_video_groups_cloudinary_timestamps_by_emotion(api):
    response = send_student_video(api.client)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["recording_id"] == "recording-1"
    assert result["student_id"] == "student-1"
    assert result["duration"] == 2.0
    assert result["sample_counts"] == {"surprised": 2}
    assert set(result["emotion_timestamps"]) == set(EMOTIONS)
    assert [point["timestamp"] for point in result["timeline"]] == [0.0, 1.0]
    assert all(
        point["student_id"] == "student-1" and point["emotion"] == "surprised"
        for point in result["timeline"]
    )
    assert [point["timestamp"] for point in result["emotion_timestamps"]["surprised"]] == [
        0.0,
        1.0,
    ]
    assert result["emotion_timestamps"]["happy"] == []


def test_student_video_can_use_actual_video_duration(api):
    response = send_student_video(api.client, duration=None)
    assert response.status_code == 200, response.text
    assert response.json()["duration"] == 2.0


def test_student_video_response_normalizes_empty_emotion_groups():
    result = StudentVideoAnalysisResult.model_validate(
        {
            "recording_id": "recording-1",
            "duration": 0,
            "sample_counts": {},
            "timeline": [],
            "emotion_timestamps": {},
        }
    )
    assert result.model_dump()["emotion_timestamps"] == {
        emotion: [] for emotion in EMOTIONS
    }


def test_student_video_rejects_inconsistent_response_payload(
    api, service, monkeypatch
):
    invalid = {
        "recording_id": "recording-1",
        "student_id": "student-1",
        "duration": 2.0,
        "sample_counts": {"surprised": 1},
        "timeline": [
            {
                "timestamp": 0.0,
                "emotion": "surprised",
                "confidence": 0.9,
                "face_id": "1",
                "bbox": [0, 0, 32, 32],
                "uncertain": False,
                "student_id": "student-1",
            }
        ],
        # The old count-only validator accepted this mismatched timestamp.
        "emotion_timestamps": {
            "surprised": [
                {
                    "timestamp": 1.0,
                    "confidence": 0.9,
                    "face_id": "1",
                    "bbox": [0, 0, 32, 32],
                    "uncertain": False,
                }
            ]
        },
    }
    monkeypatch.setattr(service, "analyze_student_video", lambda *args, **kwargs: invalid)
    response = send_student_video(api.client)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid video or analysis result"}

    invalid["timeline"][0]["timestamp"] = 3.0
    invalid["emotion_timestamps"]["surprised"][0]["timestamp"] = 3.0
    assert send_student_video(api.client).status_code == 422

    invalid["timeline"][0]["timestamp"] = 0.0
    invalid["emotion_timestamps"]["surprised"][0]["timestamp"] = 0.0
    invalid["timeline"][0]["student_id"] = "student-2"
    assert send_student_video(api.client).status_code == 422


def test_student_video_request_validation_and_auth(api):
    assert api.client.post("/v1/analyze/student-video", json=VIDEO).status_code == 401
    assert send_student_video(api.client, duration=0).status_code == 422
    assert send_student_video(api.client, duration=7201).status_code == 422
    assert send_student_video(api.client, sample_fps=0).status_code == 422
    assert send_student_video(api.client, sample_fps=5.1).status_code == 422
    assert send_student_video(api.client, unexpected=True).status_code == 422


def test_video_duration_is_rejected_before_inference(api, service, monkeypatch):
    analyzed = False

    def unexpected_inference(*args, **kwargs):
        nonlocal analyzed
        analyzed = True
        return []

    monkeypatch.setattr(service.video_processor, "_analyze_frame", unexpected_inference)
    assert send_video(api.client, key="short-video", duration=0.5).status_code == 422
    assert not analyzed
    assert send_student_video(api.client, duration=1.0).status_code == 422
    assert not analyzed


def test_success_cached_across_app_restart_and_url_rotation(api, settings, service):
    first = send_video(api.client)
    assert first.status_code == 200
    app = create_app(settings, service, api.downloader)
    with TestClient(app) as client:
        second = send_video(client, video_url=VIDEO["video_url"] + "-rotated")
        assert second.json() == first.json()
        assert len(api.downloads) == 1
        assert send_video(client, recording_id="different-recording").status_code == 409
        assert send_video(client, duration=3).status_code == 409


def test_failed_video_retry_cleans_temp_file(api, service, monkeypatch):
    original = service.analyze_video
    paths = []

    def broken(path, *args, **kwargs):
        paths.append(Path(path))
        raise RuntimeError("private-download-url-and-token")

    monkeypatch.setattr(service, "analyze_video", broken)
    response = send_video(api.client)
    assert response.status_code == 500 and "token" not in response.text
    assert paths and not paths[0].exists()
    monkeypatch.setattr(service, "analyze_video", original)
    assert send_video(api.client).status_code == 200
    assert len(api.downloads) == 2


def test_video_bounds_and_timeout(api, service, monkeypatch):
    assert send_video(api.client, duration=0.5).status_code == 422
    assert send_video(api.client, duration=7201).status_code == 422
    assert send_video(api.client, duration=-1).status_code == 422

    def timeout(*args, **kwargs):
        raise TimeoutError("private path")

    monkeypatch.setattr(service, "analyze_video", timeout)
    assert send_video(api.client, key="timeout").status_code == 504


def test_timeline_cap_preserves_counts():
    result = batch_result(
        [
            {"timestamp": i / 10, "emotion": "fear", "confidence": 0.9}
            for i in range(20001)
        ],
        2100,
    )
    assert result.sample_counts == {"fearful": 20001}
    assert len(result.timeline) == 20000
    assert result.timeline[0].timestamp == 0 and result.timeline[-1].timestamp == 2000


def test_job_leases_and_retry(settings, monkeypatch):
    store = JobStore(settings.cache_path)
    owner, _ = store.claim("job", "recording", 2.0, 100)
    with pytest.raises(HTTPException, match="already processing"):
        store.claim("job", "recording", 2.0, 100)
    clock = time.time()
    monkeypatch.setattr("openemotion.api.jobs.time.time", lambda: clock + 101)
    new_owner, _ = store.claim("job", "recording", 2.0, 100)
    with pytest.raises(HTTPException, match="lease expired"):
        store.finish("job", owner, BatchResult(sample_counts={}))
    store.release("job", owner)
    store.finish("job", new_owner, BatchResult(sample_counts={}))
    assert store.claim("job", "recording", 2.0, 100)[1].sample_counts == {}


@pytest.fixture
def backend_client():
    default = (
        Path(__file__).resolve().parents[3]
        / "face-emotion-backend"
        / "face-emotion-backend"
    )
    path = Path(os.environ.get("BACKEND_ROOT", str(default)))
    if not (path / "app/integrations/ai.py").exists():
        pytest.skip("Set BACKEND_ROOT to run the actual backend client contract tests")
    sys.path.insert(0, str(path))
    try:
        yield importlib.import_module("app.integrations.ai").HttpAIClient
    finally:
        sys.path.remove(str(path))


@pytest.mark.asyncio
async def test_actual_backend_http_client(api, jpeg, backend_client):
    config = SimpleNamespace(
        ai_base_url="http://testserver",
        ai_api_key="integration-test-key",
        ai_timeout_seconds=120,
        frame_timeout_seconds=10,
        max_ai_response_bytes=8 * 1024 * 1024,
    )
    client = backend_client(config, httpx.ASGITransport(app=api.app))
    timestamp = datetime.now(timezone.utc)
    frame = await client.analyze_frame(jpeg, "image/jpeg", "be-frame", timestamp)
    assert frame.frame_id == "be-frame" and frame.timestamp == timestamp
    assert frame.emotion == "surprised" and frame.face_detected
    video = await client.analyze_video(
        "recording-1", VIDEO["video_url"], 2.0, "backend-job"
    )
    assert video.sample_counts == {"surprised": 2}
    assert video.timeline[0].confidence > 0


@pytest.mark.asyncio
async def test_actual_backend_no_face(api, jpeg, backend_client, service):
    service.detector.faces = []
    config = SimpleNamespace(
        ai_base_url="http://testserver",
        ai_api_key="integration-test-key",
        ai_timeout_seconds=120,
        frame_timeout_seconds=10,
        max_ai_response_bytes=8 * 1024 * 1024,
    )
    client = backend_client(config, httpx.ASGITransport(app=api.app))
    result = await client.analyze_frame(
        jpeg, "image/jpeg", "be-empty", datetime.now(timezone.utc)
    )
    assert result.failure_reason == "no_face" and not result.face_detected
    video = await client.analyze_video(
        "recording-1", VIDEO["video_url"], 2.0, "backend-empty-job"
    )
    assert video.sample_counts == {} and video.timeline == []


def test_startup_requires_trained_weights(settings):
    settings.model_path = settings.cache_path.parent / "missing-model.pth"
    with pytest.raises(FileNotFoundError), TestClient(create_app(settings)):
        pytest.fail("API must not start without trained model weights")


def test_video_capture_released_on_timeout(service, video_bytes, tmp_path, monkeypatch):
    path = tmp_path / "video.mp4"
    path.write_bytes(video_bytes)
    capture = cv2.VideoCapture(str(path))
    monkeypatch.setattr(
        "openemotion.video.video_processor.cv2.VideoCapture", lambda _: capture
    )
    with pytest.raises(TimeoutError):
        service.video_processor.process_video(str(path), deadline=time.monotonic() - 1)
    assert not capture.isOpened()
