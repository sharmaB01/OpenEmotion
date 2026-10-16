from openemotion.config import CLASS_NAMES, IMG_SIZE, NUM_CLASSES
from openemotion.detection.face_detector import FaceDetector, create_face_detector
from openemotion.model.emotion_model import create_model, load_model
from openemotion.model.inference import predict_emotion
from openemotion.realtime.realtime_processor import RealtimeProcessor
from openemotion.service.ai_service import AIService
from openemotion.video.video_processor import VideoProcessor

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "IMG_SIZE",
    "NUM_CLASSES",
    "create_model",
    "load_model",
    "predict_emotion",
    "FaceDetector",
    "create_face_detector",
    "RealtimeProcessor",
    "VideoProcessor",
    "AIService",
]
