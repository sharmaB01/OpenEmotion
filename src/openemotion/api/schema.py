"""Wire schema mirrored from backend app/integrations/ai_schema.py.

Keep compatible with that source; cross-repository contract tests check the client.
"""

from collections import Counter
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Emotion = Literal[
    "happy", "neutral", "sad", "angry", "surprised", "fearful", "disgusted"
]
EMOTIONS = ("happy", "neutral", "sad", "angry", "surprised", "fearful", "disgusted")


class EmotionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emotion: Emotion
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    mock: bool = False


class FrameResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    face_detected: bool
    emotion: Emotion | Literal["fail_detection"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    probabilities: dict[
        Emotion, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    ]
    face_id: str | None = Field(default=None, max_length=128)
    failure_reason: Literal["no_face", "ai_error"] | None = None
    mock: bool = False
    # Contract v1 section 6.1: identifies the checkpoint that produced the result.
    model_version: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def valid_detection(self):
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must include timezone")
        if set(self.probabilities) != set(EMOTIONS):
            raise ValueError("probabilities must contain all seven emotion classes")
        if self.face_detected:
            if (
                self.emotion == "fail_detection"
                or not self.face_id
                or self.failure_reason
            ):
                raise ValueError(
                    "Detected face requires emotion and face_id, without failure_reason"
                )
            if abs(sum(self.probabilities.values()) - 1) > 0.001:
                raise ValueError("Detected face probabilities must sum to one")
            if abs(self.probabilities[self.emotion] - self.confidence) > 0.001:
                raise ValueError(
                    "confidence must match the selected emotion probability"
                )
            if self.confidence < max(self.probabilities.values()):
                raise ValueError("emotion must be a most probable class")
        else:
            if self.emotion != "fail_detection" or self.confidence != 0 or self.face_id:
                raise ValueError(
                    "Undetected face must have fail_detection, zero confidence, null face_id"
                )
            if any(self.probabilities.values()):
                raise ValueError("Undetected face probabilities must be zero")
            self.failure_reason = self.failure_reason or "no_face"
        return self

    @classmethod
    def failed(cls, frame_id, timestamp, reason="ai_error"):
        return cls(
            frame_id=frame_id,
            timestamp=timestamp,
            face_detected=False,
            emotion="fail_detection",
            confidence=0,
            probabilities=dict.fromkeys(EMOTIONS, 0.0),
            failure_reason=reason,
        )


class TimelinePoint(EmotionResult):
    timestamp: float = Field(
        ge=0, allow_inf_nan=False, description="Seconds from recording start"
    )
    student_id: str | None = Field(default=None, max_length=36)


class BatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_counts: dict[
        Emotion, Annotated[int, Field(strict=True, ge=0, le=1000000000)]
    ] = Field(description="Counts, not percentages; required for correct aggregation")
    timeline: list[TimelinePoint] = Field(default_factory=list, max_length=20000)
    summary: str = Field(default="", max_length=4000)
    mock: bool = False
    model_version: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def counts_valid(self):
        timeline_counts = Counter(point.emotion for point in self.timeline)
        if any(
            count > self.sample_counts.get(emotion, 0)
            for emotion, count in timeline_counts.items()
        ):
            raise ValueError("Timeline must be a subset of the counted samples")
        return self


class VideoEmotionTimestamp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: float = Field(
        ge=0, allow_inf_nan=False, description="Seconds from recording start"
    )
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    face_id: str | None = Field(default=None, max_length=128)
    bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    uncertain: bool = False


class VideoTimelinePoint(VideoEmotionTimestamp):
    emotion: Emotion
    student_id: str | None = Field(default=None, max_length=36)


class StudentVideoAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recording_id: str = Field(min_length=1, max_length=36)
    student_id: str | None = Field(default=None, max_length=36)
    duration: float = Field(ge=0, allow_inf_nan=False)
    sample_counts: dict[
        Emotion, Annotated[int, Field(strict=True, ge=0, le=1000000000)]
    ] = Field(description="Counts per emotion across all analyzed face samples")
    timeline: list[VideoTimelinePoint] = Field(default_factory=list, max_length=20000)
    emotion_timestamps: dict[Emotion, list[VideoEmotionTimestamp]] = Field(
        default_factory=dict,
        description="Timeline grouped by emotion for quick FE/BE lookup",
    )
    summary: str = Field(default="", max_length=4000)
    mock: bool = False

    @model_validator(mode="after")
    def grouped_timestamps_match_timeline(self):
        # Keep a stable JSON template for consumers: all seven emotion buckets
        # are always present, including empty buckets.
        self.emotion_timestamps = {
            emotion: self.emotion_timestamps.get(emotion, []) for emotion in EMOTIONS
        }
        timeline_counts = Counter(point.emotion for point in self.timeline)
        if any(point.timestamp > self.duration for point in self.timeline):
            raise ValueError("Timeline timestamp cannot exceed video duration")
        if any(point.student_id != self.student_id for point in self.timeline):
            raise ValueError("Timeline student_id must match the result student_id")
        expected_groups = {emotion: [] for emotion in EMOTIONS}
        for point in self.timeline:
            expected_groups[point.emotion].append(
                VideoEmotionTimestamp(
                    timestamp=point.timestamp,
                    confidence=point.confidence,
                    face_id=point.face_id,
                    bbox=point.bbox,
                    uncertain=point.uncertain,
                )
            )
        if self.emotion_timestamps != expected_groups:
            raise ValueError("emotion_timestamps must group the returned timeline")
        if any(
            count > self.sample_counts.get(emotion, 0)
            for emotion, count in timeline_counts.items()
        ):
            raise ValueError("Timeline must be a subset of the counted samples")
        return self
