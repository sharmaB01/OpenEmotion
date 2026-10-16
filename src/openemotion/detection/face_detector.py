from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class FaceDetector:
    """Face Detector using OpenCV Haar Cascades."""

    def __init__(self, min_size: Tuple[int, int] = (60, 60), scale_factor: float = 1.1, min_neighbors: int = 5):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Could not load OpenCV Haar cascade classifier.")
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Phát hiện các khuôn mặt trong frame.

        Returns:
            List[Dict]: Danh sách khuôn mặt dạng `[{"face_id": 1, "bbox": [x, y, w, h]}]`
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        detected = []
        for idx, (x, y, w, h) in enumerate(faces, start=1):
            detected.append({
                "face_id": idx,
                "bbox": [int(x), int(y), int(w), int(h)],
            })
        return detected

    def get_largest_face(self, frame_bgr: np.ndarray) -> Dict[str, Any] | None:
        """
        Trả về bbox của khuôn mặt lớn nhất (nhiều mặt -> chọn bbox lớn nhất).
        """
        faces = self.detect_faces(frame_bgr)
        if not faces:
            return None
        # Max by area (w * h)
        return max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])


def create_face_detector() -> FaceDetector:
    return FaceDetector()
