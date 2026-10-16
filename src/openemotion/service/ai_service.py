from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch

from openemotion.analysis.emotion_analyzer import EmotionAnalyzer
from openemotion.api.adapter import batch_result, frame_result, student_video_result
from openemotion.config import CLASS_NAMES, LOW_EMOTION_CONFIDENCE, VIDEO_SAMPLE_FPS
from openemotion.detection.face_detector import create_face_detector
from openemotion.model.emotion_model import load_model
from openemotion.model.inference import predict_emotion
from openemotion.realtime.realtime_processor import RealtimeProcessor
from openemotion.video.video_processor import VideoProcessor


class AIService:
    """
    Cổng giao tiếp chính (Service Interface) của AI Module cho Backend.
    Quản lý mô hình, Face Detector và phân phối tác vụ Realtime / Video Processing.
    """

    def __init__(
        self,
        model_path: str,
        device: Union[str, torch.device] = "cpu",
        low_confidence_threshold: float = LOW_EMOTION_CONFIDENCE,
    ):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}. Service initialization failed.")

        try:
            self.model = load_model(str(path), self.device, len(CLASS_NAMES))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}") from e

        try:
            self.detector = create_face_detector()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Face Detector: {str(e)}") from e

        # Serialize individual detector/model operations while allowing realtime
        # requests to run between sampled video frames.
        inference_lock = RLock()
        self.realtime_processor = RealtimeProcessor(
            self.model,
            self.detector,
            self.device,
            inference_lock,
            low_confidence_threshold,
        )
        self.video_processor = VideoProcessor(
            self.model,
            self.detector,
            self.device,
            inference_lock,
            low_confidence_threshold,
        )

    def analyze_frame(self, frame: bytes, frame_id: str, timestamp: datetime):
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        raw = self.realtime_processor.process_frame({
            "frame_id": frame_id, "timestamp": timestamp, "image": image,
        })
        return frame_result(raw, frame_id, timestamp)

    def analyze_frame_batch(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for frame in frames:
            result = self.analyze_frame(
                frame["image"],
                frame["frame_id"],
                frame["timestamp"],
            )
            results.append(
                {
                    "student_id": frame.get("student_id"),
                    "result": result,
                }
            )
        return results

    def analyze_video(self, video_path: str, duration: float, deadline: float | None = None):
        raw = self.video_processor.process_video(
            video_path,
            deadline=deadline,
            # Container metadata can differ slightly from the backend duration.
            max_duration=duration + 0.5,
        )
        if raw.get("status") != "completed":
            raise ValueError("Video could not be decoded or analyzed")
        return batch_result(raw["raw_results"], duration)

    def analyze_student_video(
        self,
        video_path: str,
        recording_id: str,
        student_id: str | None = None,
        duration: float | None = None,
        sample_fps: float = VIDEO_SAMPLE_FPS,
        deadline: float | None = None,
        max_duration: float | None = None,
    ):
        duration_limit = max_duration
        if duration is not None:
            request_limit = duration + 0.5
            duration_limit = (
                min(duration_limit, request_limit)
                if duration_limit is not None
                else request_limit
            )
        raw = self.video_processor.process_video(
            video_path,
            fps_sample=sample_fps,
            deadline=deadline,
            max_duration=duration_limit,
        )
        if raw.get("status") != "completed":
            raise ValueError("Video could not be decoded or analyzed")
        actual_duration = raw.get("duration", 0.0)
        effective_duration = duration if duration is not None else actual_duration
        if actual_duration > effective_duration + 0.5:
            raise ValueError("Video duration exceeds request duration")
        return student_video_result(
            raw["raw_results"],
            recording_id=recording_id,
            duration=effective_duration,
            student_id=student_id,
        )

    def process_realtime_batch(self, session_id: str, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Interface Realtime: Nhận batch frame từ Backend và trả về kết quả cảm xúc."""
        return self.realtime_processor.process_realtime_batch(session_id, frames)

    def process_video_meeting(self, meeting_id: str, video_path: str) -> Dict[str, Any]:
        """Interface Video: Phân tích video ghi lại buổi học và trả kết quả tổng hợp."""
        video_res = self.video_processor.process_video(video_path)

        if video_res.get("status") == "failed":
            return {
                "status": "failed",
                "meeting_id": meeting_id,
                "error_code": video_res.get("error_code", "UNKNOWN_ERROR"),
                "message": video_res.get("message", "Xử lý video thất bại."),
            }

        analysis = EmotionAnalyzer.analyze(
            video_res["raw_results"],
            duration=video_res.get("duration", 0.0),
        )

        return {
            "status": "completed",
            "meeting_id": meeting_id,
            "duration": analysis["duration"],
            "participants": analysis["participants"],
            "summary": analysis["summary"],
            "emotion_distribution": analysis["emotion_distribution"],
            "timeline": analysis["timeline"],
        }

    def analyze_single_image(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Tiện ích test/debug nội bộ: Phân tích 1 ảnh đơn (đã chốt ở mục 10.4).
        """
        if isinstance(image, str):
            frame_bgr = self.realtime_processor.decode_base64_image(image)
        elif isinstance(image, np.ndarray):
            frame_bgr = image
        else:
            frame_bgr = None

        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "status": "failed",
                "error_code": "FRAME_DECODE_ERROR",
                "message": "Không thể giải mã hình ảnh.",
            }

        faces = self.detector.detect_faces(frame_bgr)
        if not faces:
            return {
                "status": "failed",
                "error_code": "NO_FACE_DETECTED",
                "message": "Không phát hiện khuôn mặt nào trong frame.",
            }

        results = []
        for face_info in faces:
            x, y, w, h = face_info["bbox"]
            face_crop = frame_bgr[y:y + h, x:x + w]
            if face_crop.size == 0:
                continue
            emotion, confidence, probs = predict_emotion(self.model, face_crop, self.device)
            results.append({
                "face_id": face_info["face_id"],
                "bbox": [x, y, w, h],
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
            })

        return {
            "status": "completed",
            "faces": results,
        }
