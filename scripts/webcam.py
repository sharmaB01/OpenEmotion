import argparse
import sys
from pathlib import Path

import cv2
import torch

# Allow running this file directly from the repository.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openemotion.config import CLASS_NAMES
from openemotion.detection.face_detector import create_face_detector
from openemotion.model.emotion_model import load_model
from openemotion.model.inference import predict_emotion


def main():
    parser = argparse.ArgumentParser(
        description="OpenEmotion real-time OpenCV webcam demo"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained .pth model",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = load_model(
        args.model,
        device,
        len(CLASS_NAMES),
    )

    detector = create_face_detector()

    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("OpenEmotion camera started.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        faces = detector.detect_faces(frame)

        for face_info in faces:
            x, y, w, h = face_info["bbox"]
            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                continue

            emotion, confidence, probs = predict_emotion(
                model,
                face,
                device,
            )

            label = f"{emotion} {confidence * 100:.1f}%"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("OpenEmotion", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
