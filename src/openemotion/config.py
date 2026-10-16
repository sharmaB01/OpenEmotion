"""Central configuration for the OpenEmotion system."""

from pathlib import Path

# --- Emotion Classes ---
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

NUM_CLASSES = len(CLASS_NAMES)

# --- Image ---
IMG_SIZE = 224

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # OpenEmotion/
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "emotion_model_best.pth"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Face Detection ---
FACE_MIN_SIZE = (60, 60)
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5

# --- Emotion Thresholds ---
LOW_EMOTION_CONFIDENCE = 0.4

# --- Video Processing ---
VIDEO_SAMPLE_FPS = 1  # 1 frame per second
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

# --- ImageNet Normalization ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
