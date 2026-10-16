from typing import Dict, Any, Tuple
import cv2
import numpy as np
import torch
from openemotion.config import CLASS_NAMES
from openemotion.model.emotion_model import INFERENCE_TRANSFORM


def predict_emotion(
    model: torch.nn.Module,
    face_bgr: np.ndarray,
    device: torch.device,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Nhận diện cảm xúc từ một ảnh khuôn mặt đã cropped (BGR format).

    Returns:
        emotion (str): Cảm xúc có xác suất cao nhất.
        confidence (float): Độ tin cậy (0.0 -> 1.0).
        probabilities (dict): Xác suất của cả 7 loại cảm xúc.
    """
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Invalid face image")

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFERENCE_TRANSFORM(face_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs_tensor = torch.softmax(output, dim=1)[0]
        confidence, prediction = torch.max(probs_tensor, dim=0)

    probabilities = {
        name: float(probs_tensor[idx].item())
        for idx, name in enumerate(CLASS_NAMES)
    }

    return CLASS_NAMES[prediction.item()], float(confidence.item()), probabilities
