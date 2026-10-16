import base64

import cv2
import numpy as np
import pytest
import torch

from openemotion.analysis.emotion_analyzer import EmotionAnalyzer
from openemotion.config import CLASS_NAMES
from openemotion.detection.face_detector import FaceDetector, create_face_detector
from openemotion.model.emotion_model import create_model
from openemotion.model.inference import predict_emotion
from openemotion.realtime.realtime_processor import RealtimeProcessor
from openemotion.service.ai_service import AIService
from openemotion.video.video_processor import VideoProcessor


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dummy_model(device):
    model = create_model(len(CLASS_NAMES))
    model.eval()
    return model


@pytest.fixture
def dummy_face_bgr():
    # Synthetic face-like 100x100 BGR image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    return img


@pytest.fixture
def dummy_frame_base64(dummy_face_bgr):
    _, buffer = cv2.imencode(".jpg", dummy_face_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# 1. Model Tests
def test_create_model():
    model = create_model(7)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


def test_predict_emotion(dummy_model, dummy_face_bgr, device):
    emotion, confidence, probs = predict_emotion(dummy_model, dummy_face_bgr, device)
    assert emotion in CLASS_NAMES
    assert 0.0 <= confidence <= 1.0
    assert len(probs) == 7
    assert pytest.approx(sum(probs.values()), 0.01) == 1.0


# 2. Face Detector Tests
def test_face_detector_initialization():
    detector = create_face_detector()
    assert isinstance(detector, FaceDetector)


def test_face_detector_empty_frame():
    detector = create_face_detector()
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detector.detect_faces(empty) == []


# 3. Realtime Processor & Edge Cases
def test_realtime_processor_no_face(dummy_model, device):
    detector = create_face_detector()
    processor = RealtimeProcessor(dummy_model, detector, device)

    # Black image has no face
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", black_img)
    b64_str = base64.b64encode(buffer).decode("utf-8")

    batch = [
        {"frame_id": 1, "timestamp": 0.0, "image": b64_str}
    ]
    res = processor.process_realtime_batch("session_test", batch)

    assert res["session_id"] == "session_test"
    assert len(res["results"]) == 1
    assert res["results"][0]["status"] == "no_face_detected"
    assert res["results"][0]["error_code"] == "NO_FACE_DETECTED"


def test_realtime_processor_corrupt_frame(dummy_model, device):
    detector = create_face_detector()
    processor = RealtimeProcessor(dummy_model, detector, device)

    batch = [
        {"frame_id": 1, "timestamp": 0.0, "image": "invalid_base64_string!!!"}
    ]
    res = processor.process_realtime_batch("session_test", batch)
    assert res["results"][0]["status"] == "failed"
    assert res["results"][0]["error_code"] == "FRAME_DECODE_ERROR"


# 4. Video Processor Edge Cases
def test_video_processor_invalid_file(dummy_model, device):
    detector = create_face_detector()
    processor = VideoProcessor(dummy_model, detector, device)

    res = processor.process_video("/non/existent/video.mp4")
    assert res["status"] == "failed"
    assert res["error_code"] == "INVALID_VIDEO"


def test_video_processor_unsupported_format(dummy_model, device, tmp_path):
    detector = create_face_detector()
    processor = VideoProcessor(dummy_model, detector, device)

    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("not a video")

    res = processor.process_video(str(invalid_file))
    assert res["status"] == "failed"
    assert res["error_code"] == "UNSUPPORTED_FORMAT"


# 5. Emotion Analyzer Tests
def test_emotion_analyzer():
    raw_results = [
        {"timestamp": 0.0, "face_id": 1, "emotion": "happy", "confidence": 0.9},
        {"timestamp": 1.0, "face_id": 1, "emotion": "neutral", "confidence": 0.8},
        {"timestamp": 2.0, "face_id": 1, "emotion": "happy", "confidence": 0.85},
        {"timestamp": 2.0, "status": "no_face_detected"},
    ]

    report = EmotionAnalyzer.analyze(raw_results, duration=10.0)
    assert report["duration"] == 10.0
    assert report["summary"]["dominant_emotion"] == "happy"
    assert "happy" in report["emotion_distribution"]
    assert report["emotion_distribution"]["happy"] > 0
    assert len(report["timeline"]) == 3


def test_emotion_analyzer_empty():
    report = EmotionAnalyzer.analyze([], duration=0.0)
    assert report["summary"]["dominant_emotion"] == "neutral"
    assert report["participants"] == 0
    assert report["timeline"] == []


# 6. AI Service Fail-Fast & Initialization
def test_ai_service_missing_model():
    with pytest.raises(FileNotFoundError):
        AIService(model_path="/non/existent/model.pth")


def test_ai_service_with_real_model():
    from openemotion.config import DEFAULT_MODEL_PATH

    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("Trained model weights are not present")
    service = AIService(model_path=str(DEFAULT_MODEL_PATH), device="cpu")
    res = service.process_realtime_batch("test_session", [])
    assert res["session_id"] == "test_session"
    assert res["results"] == []

