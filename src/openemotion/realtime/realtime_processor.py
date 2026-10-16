import base64
from threading import RLock
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from openemotion.detection.face_detector import FaceDetector
from openemotion.config import LOW_EMOTION_CONFIDENCE
from openemotion.model.inference import predict_emotion


class RealtimeProcessor:
    """Xử lý cảm xúc theo thời gian thực (nhận batch frame từ Backend)."""

    def __init__(
        self,
        model: torch.nn.Module,
        detector: FaceDetector,
        device: torch.device,
        inference_lock=None,
        low_confidence_threshold: float = LOW_EMOTION_CONFIDENCE,
    ):
        self.model = model
        self.detector = detector
        self.device = device
        self.inference_lock = inference_lock or RLock()
        self.low_confidence_threshold = low_confidence_threshold

    def decode_base64_image(self, base64_str: str) -> np.ndarray | None:
        """Decode base64 string to OpenCV BGR image."""
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]
            image_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def process_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        with self.inference_lock:
            return self._process_frame(frame_data)

    def _process_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý 1 frame trong batch.
        Input frame_data: {"frame_id": 101, "timestamp": 12.4, "image": "<base64>"}
        """
        frame_id = frame_data.get("frame_id")
        timestamp = frame_data.get("timestamp", 0.0)
        image_input = frame_data.get("image")

        if image_input is None:
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "status": "failed",
                "error_code": "FRAME_DECODE_ERROR",
                "message": "Hình ảnh rỗng hoặc không hợp lệ.",
            }

        if isinstance(image_input, str):
            frame_bgr = self.decode_base64_image(image_input)
        elif isinstance(image_input, np.ndarray):
            frame_bgr = image_input
        else:
            frame_bgr = None

        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "status": "failed",
                "error_code": "FRAME_DECODE_ERROR",
                "message": "Không thể giải mã hình ảnh.",
            }

        # Phát hiện mặt
        faces = self.detector.detect_faces(frame_bgr)
        if not faces:
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "status": "no_face_detected",
                "error_code": "NO_FACE_DETECTED",
            }

        # Chọn mặt lớn nhất nếu có nhiều mặt
        largest_face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        x, y, w, h = largest_face["bbox"]
        face_crop = frame_bgr[y:y + h, x:x + w]

        if face_crop.size == 0:
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "status": "no_face_detected",
                "error_code": "NO_FACE_DETECTED",
            }

        emotion, confidence, probs = predict_emotion(self.model, face_crop, self.device)
        uncertain = confidence < self.low_confidence_threshold

        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "status": "ok",
            "face": {
                "face_id": largest_face["face_id"],
                "bbox": [x, y, w, h],
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
                "uncertain": uncertain,
            },
        }

    def process_realtime_batch(self, session_id: str, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Nhận batch frame từ Backend và xử lý.
        """
        results = []
        for f in frames:
            res = self.process_frame(f)
            results.append(res)

        return {
            "session_id": session_id,
            "results": results,
        }
