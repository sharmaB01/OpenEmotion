"""Convert model labels/results without changing the trained class order."""

from collections import Counter
from datetime import datetime

from openemotion.api.schema import (
    BatchResult,
    EMOTIONS,
    FrameResult,
    StudentVideoAnalysisResult,
    TimelinePoint,
    VideoEmotionTimestamp,
    VideoTimelinePoint,
)

LABELS = {"disgust": "disgusted", "fear": "fearful", "surprise": "surprised"}


def emotion_name(name: str) -> str:
    return LABELS.get(name, name)


def frame_result(raw: dict, frame_id: str, timestamp: datetime) -> FrameResult:
    if raw.get("status") == "no_face_detected":
        return FrameResult.failed(frame_id, timestamp, reason="no_face")
    if raw.get("status") != "ok":
        raise ValueError("Frame could not be decoded or analyzed")
    face = raw["face"]
    return FrameResult(
        frame_id=frame_id,
        timestamp=timestamp,
        face_detected=True,
        # Per-frame detection identifier; never a student identity or tracking ID.
        face_id=str(face["face_id"]),
        emotion=emotion_name(face["emotion"]),
        confidence=face["confidence"],
        probabilities={emotion_name(k): v for k, v in face["probabilities"].items()},
    )


def batch_result(raw_results: list[dict], duration: float) -> BatchResult:
    counts = Counter()
    points = []
    for raw in raw_results:
        if raw.get("status") == "no_face_detected":
            continue
        point = TimelinePoint(
            timestamp=raw["timestamp"],
            emotion=emotion_name(raw["emotion"]),
            confidence=raw["confidence"],
            student_id=None,
        )
        if point.timestamp > duration:
            raise ValueError("Video sample exceeds backend recording duration")
        counts[point.emotion] += 1
        points.append(point)
    points.sort(key=lambda point: point.timestamp)
    # Keep all counts while returning at most 20,000 representative samples.
    if len(points) > 20000:
        points = [points[i * (len(points) - 1) // 19999] for i in range(20000)]
    total = sum(counts.values())
    return BatchResult(
        sample_counts=dict(counts),
        timeline=points,
        summary=f"{total} classified samples."
        if total
        else "No faces detected in video.",
    )


def student_video_result(
    raw_results: list[dict],
    recording_id: str,
    duration: float,
    student_id: str | None = None,
) -> StudentVideoAnalysisResult:
    counts = Counter()
    points = []
    for raw in raw_results:
        if raw.get("status") == "no_face_detected":
            continue
        emotion = emotion_name(raw["emotion"])
        point = VideoTimelinePoint(
            timestamp=raw["timestamp"],
            emotion=emotion,
            confidence=raw["confidence"],
            face_id=str(raw["face_id"]) if "face_id" in raw else None,
            bbox=raw.get("bbox"),
            uncertain=raw.get("uncertain", False),
            student_id=student_id,
        )
        if point.timestamp > duration:
            raise ValueError("Video sample exceeds recording duration")
        counts[emotion] += 1
        points.append(point)
    points.sort(key=lambda point: point.timestamp)
    if len(points) > 20000:
        points = [points[i * (len(points) - 1) // 19999] for i in range(20000)]
    grouped = {emotion: [] for emotion in EMOTIONS}
    for point in points:
        grouped[point.emotion].append(
            VideoEmotionTimestamp(
                timestamp=point.timestamp,
                confidence=point.confidence,
                face_id=point.face_id,
                bbox=point.bbox,
                uncertain=point.uncertain,
            )
        )
    total = sum(counts.values())
    return StudentVideoAnalysisResult(
        recording_id=recording_id,
        student_id=student_id,
        duration=duration,
        sample_counts=dict(counts),
        timeline=points,
        emotion_timestamps=grouped,
        summary=f"{total} classified samples."
        if total
        else "No faces detected in video.",
    )
