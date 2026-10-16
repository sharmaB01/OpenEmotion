import base64
import binascii
import logging
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from openemotion.api.download import VideoDownloader
from openemotion.api.jobs import JobStore
from openemotion.api.schema import (
    BatchResult,
    FrameResult,
    StudentVideoAnalysisResult,
)
from openemotion.api.settings import APISettings
from openemotion.config import VIDEO_SAMPLE_FPS

logger = logging.getLogger(__name__)


class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recording_id: str = Field(min_length=1, max_length=36)
    video_url: str = Field(min_length=1, max_length=16384, repr=False)
    duration: float = Field(ge=0, allow_inf_nan=False)


class StudentVideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recording_id: str = Field(min_length=1, max_length=36)
    video_url: str = Field(min_length=1, max_length=16384, repr=False)
    student_id: str | None = Field(default=None, max_length=36)
    duration: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sample_fps: float = Field(
        default=VIDEO_SAMPLE_FPS, gt=0, le=5, allow_inf_nan=False
    )


class FrameBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    image: str = Field(min_length=1, max_length=1024 * 1024, repr=False)
    content_type: str = Field(default="image/jpeg", pattern=r"^image/(jpeg|png)$")
    student_id: str | None = Field(default=None, max_length=36)


class FrameBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=128)
    frames: list[FrameBatchItem] = Field(min_length=1, max_length=64)


class FrameBatchEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    student_id: str | None = None
    result: FrameResult


class FrameBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    results: list[FrameBatchEntry]


class BodyLimitMiddleware:
    """Limit multipart/JSON before the framework buffers/parses an upload."""

    def __init__(self, app, max_frame_bytes, max_batch_bytes):
        self.app = app
        self.max_frame_bytes = max_frame_bytes
        self.max_batch_bytes = max_batch_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            max_bytes = (
                self.max_batch_bytes
                if scope.get("path") == "/v1/analyze/frames"
                else self.max_frame_bytes
            )
            if len(body) > max_bytes:
                response = JSONResponse(
                    {"detail": "Request body exceeds limit"}, status_code=413
                )
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def validate_image(data: bytes, content_type: str):
    expected = {"image/jpeg": "JPEG", "image/png": "PNG"}.get(content_type)
    if expected is None:
        raise HTTPException(415, "Frame must be JPEG or PNG")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != expected or image.width * image.height > 1920 * 1080:
                raise ValueError("Invalid image format or pixel budget")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(
            422, "Invalid image or frame pixel budget exceeded"
        ) from None


def decode_frame_payload(frame: FrameBatchItem, max_bytes: int) -> bytes:
    payload = frame.image.split(",", 1)[1] if "," in frame.image else frame.image
    try:
        data = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(422, "Invalid base64 frame") from None
    if len(data) > max_bytes:
        raise HTTPException(413, "Frame exceeds byte limit")
    validate_image(data, frame.content_type)
    return data


def create_app(settings=None, service=None, downloader=None):
    settings = settings or APISettings()
    # httpx INFO messages include signed URLs; do not emit those request logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    @asynccontextmanager
    async def lifespan(app):
        if service is None:
            from openemotion.service.ai_service import AIService

            app.state.ai = await run_in_threadpool(
                AIService,
                str(settings.model_path),
                settings.device,
                settings.low_confidence_threshold,
            )
        else:
            app.state.ai = service
        app.state.jobs = JobStore(settings.cache_path)
        app.state.downloader = downloader or VideoDownloader(settings)
        yield

    app = FastAPI(title="OpenEmotion Backend API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        BodyLimitMiddleware,
        max_frame_bytes=settings.max_frame_bytes + 64 * 1024,
        max_batch_bytes=(settings.max_frame_bytes + 16 * 1024) * 64,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never reflect a signed video URL or input image in validation responses.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": e["loc"], "type": e["type"], "msg": e["msg"]}
                    for e in exc.errors()
                ]
            },
        )

    model_version = (settings.model_version or settings.model_path.stem)[:64]

    def authorize(authorization: Annotated[str | None, Header()] = None):
        expected = "Bearer " + settings.api_key.get_secret_value()
        if not authorization or not secrets.compare_digest(
            authorization.encode(), expected.encode()
        ):
            raise HTTPException(
                401, "Invalid AI API key", headers={"WWW-Authenticate": "Bearer"}
            )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post(
        "/v1/analyze/frame",
        response_model=FrameResult,
        dependencies=[Depends(authorize)],
    )
    def analyze_frame(
        frame: Annotated[UploadFile, File()],
        frame_id: Annotated[str, Form(min_length=1, max_length=64)],
        timestamp: Annotated[datetime, Form()],
    ):
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise HTTPException(422, "timestamp must include timezone")
        data = frame.file.read(settings.max_frame_bytes + 1)
        if len(data) > settings.max_frame_bytes:
            raise HTTPException(413, "Frame exceeds byte limit")
        validate_image(data, frame.content_type)
        try:
            result = FrameResult.model_validate(
                app.state.ai.analyze_frame(data, frame_id, timestamp)
            )
            if result.frame_id != frame_id or result.timestamp != timestamp:
                raise ValueError("Mismatched frame metadata")
            return result.model_copy(update={"model_version": model_version})
        except Exception as exc:  # noqa: BLE001 - redact native/model errors at the HTTP boundary
            logger.warning("Frame inference failed (%s)", type(exc).__name__)
            raise HTTPException(500, "Frame inference failed") from None

    @app.post(
        "/v1/analyze/frames",
        response_model=FrameBatchResponse,
        dependencies=[Depends(authorize)],
    )
    def analyze_frames(data: FrameBatchRequest):
        frames = []
        for item in data.frames:
            if item.timestamp.tzinfo is None or item.timestamp.utcoffset() is None:
                raise HTTPException(422, "timestamp must include timezone")
            frames.append(
                {
                    "frame_id": item.frame_id,
                    "timestamp": item.timestamp,
                    "image": decode_frame_payload(item, settings.max_frame_bytes),
                    "student_id": item.student_id,
                }
            )
        try:
            results = app.state.ai.analyze_frame_batch(frames)
            return {"session_id": data.session_id, "results": results}
        except Exception as exc:  # noqa: BLE001 - redact model/native details
            logger.warning("Batch frame inference failed (%s)", type(exc).__name__)
            raise HTTPException(500, "Batch frame inference failed") from None

    @app.post(
        "/v1/analyze/video",
        response_model=BatchResult,
        dependencies=[Depends(authorize)],
    )
    def analyze_video(
        data: VideoRequest,
        idempotency_key: Annotated[str, Header(min_length=1, max_length=128)],
    ):
        if data.duration > settings.max_video_seconds:
            raise HTTPException(422, "Recording exceeds duration limit")
        jobs = app.state.jobs
        owner, cached = jobs.claim(
            idempotency_key,
            data.recording_id,
            data.duration,
            settings.processing_timeout_seconds + 60,
        )
        if cached is not None:
            return cached
        deadline = time.monotonic() + settings.processing_timeout_seconds
        try:
            with TemporaryDirectory(prefix="openemotion-") as directory:
                path = app.state.downloader.download(
                    data.video_url, Path(directory), deadline
                )
                result = BatchResult.model_validate(
                    app.state.ai.analyze_video(
                        str(path), data.duration, deadline=deadline
                    )
                )
                if time.monotonic() > deadline:
                    raise TimeoutError("Video analysis deadline exceeded")
                if any(point.timestamp > data.duration for point in result.timeline):
                    raise ValueError("Video sample exceeds recording duration")
            result = result.model_copy(update={"model_version": model_version})
            jobs.finish(idempotency_key, owner, result)
            return result
        except Exception as exc:
            jobs.release(idempotency_key, owner)
            logger.warning("Video inference failed (%s)", type(exc).__name__)
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
                raise HTTPException(
                    504, "Video analysis timed out; retry request"
                ) from None
            if isinstance(exc, httpx.HTTPError):
                raise HTTPException(
                    502, "Video download failed; retry request"
                ) from None
            if isinstance(exc, ValueError):
                raise HTTPException(422, "Invalid video or analysis result") from None
            raise HTTPException(500, "Video analysis failed; retry request") from None

    @app.post(
        "/v1/analyze/student-video",
        response_model=StudentVideoAnalysisResult,
        dependencies=[Depends(authorize)],
    )
    def analyze_student_video(data: StudentVideoRequest):
        if data.duration is not None and data.duration > settings.max_video_seconds:
            raise HTTPException(422, "Recording exceeds duration limit")
        deadline = time.monotonic() + settings.processing_timeout_seconds
        try:
            with TemporaryDirectory(prefix="openemotion-") as directory:
                path = app.state.downloader.download(
                    data.video_url, Path(directory), deadline
                )
                result = StudentVideoAnalysisResult.model_validate(
                    app.state.ai.analyze_student_video(
                        str(path),
                        recording_id=data.recording_id,
                        student_id=data.student_id,
                        duration=data.duration,
                        sample_fps=data.sample_fps,
                        deadline=deadline,
                        max_duration=settings.max_video_seconds,
                    )
                )
                if result.duration > settings.max_video_seconds:
                    raise ValueError("Video duration exceeds configured limit")
                if time.monotonic() > deadline:
                    raise TimeoutError("Video analysis deadline exceeded")
            return result
        except Exception as exc:
            logger.warning("Student video inference failed (%s)", type(exc).__name__)
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
                raise HTTPException(
                    504, "Video analysis timed out; retry request"
                ) from None
            if isinstance(exc, httpx.HTTPError):
                raise HTTPException(
                    502, "Video download failed; retry request"
                ) from None
            if isinstance(exc, ValueError):
                raise HTTPException(422, "Invalid video or analysis result") from None
            raise HTTPException(500, "Video analysis failed; retry request") from None

    return app
