# Kiến trúc AI hiện tại

OpenEmotion là service AI tách rời backend lớp học. Backend chịu trách nhiệm xác
thực người dùng, meeting, student, recording và lưu kết quả. Service AI chỉ nhận
frame/video hợp lệ, phát hiện khuôn mặt, crop, phân loại cảm xúc và trả schema đã
thống nhất.

## Luồng realtime

```text
FE camera
   |
   v
Backend meeting/signaling
   |
   v
POST /v1/analyze/frame hoặc /v1/analyze/frames
   |
   v
OpenEmotion API
   |
   v
Validate auth, bytes, image format, timestamp
   |
   v
RealtimeProcessor
   |
   v
OpenCV Haar face detector -> largest face crop
   |
   v
EfficientNet-B0 emotion classifier
   |
   v
FrameResult + BE-owned student mapping
```

`/v1/analyze/frame` là contract chính với backend hiện tại. `/v1/analyze/frames`
là API batch JSON cho cùng model pipeline, dùng khi backend muốn gom nhiều frame
và giữ ánh xạ `student_id -> result` mà không để AI tự đoán danh tính.

## Luồng video batch

```text
Recording uploaded to Cloudinary
   |
   v
Backend creates signed download URL + analysis job id
   |
   v
POST /v1/analyze/video with Idempotency-Key
   |
   v
OpenEmotion downloads to temp file
   |
   v
VideoProcessor samples frames at VIDEO_SAMPLE_FPS
   |
   v
Detect every visible face -> classify emotion
   |
   v
BatchResult sample_counts + capped timeline
   |
   v
Backend persists immutable analysis result
```

Video jobs are cached in SQLite by idempotency key after success. Replays with a
rotated signed URL return the cached result; mismatched recording or duration is
rejected.

`/v1/analyze/student-video` là biến thể phục vụ video record của một học sinh.
API này nhận Cloudinary URL, `recording_id`, `student_id` tùy chọn, `duration`
tùy chọn và trả timeline đã gắn `student_id` cùng `emotion_timestamps` được group
theo emotion. Contract `/v1/analyze/video` vẫn giữ nguyên để backend worker cũ
không bị đổi schema.

## Model and preprocessing

- Backbone: EfficientNet-B0 with a 7-class classifier head.
- Dataset class order: `angry, disgust, fear, happy, neutral, sad, surprise`.
- Backend label mapping happens only at the adapter boundary:
  `disgust -> disgusted`, `fear -> fearful`, `surprise -> surprised`.
- Input transform: RGB conversion, resize to `224x224`, ImageNet normalization.
- Face detector: OpenCV Haar cascade. Realtime chooses the largest face per frame;
  video records every detected face sample.

## Confidence and calibration

The public contract does not expose an `unknown` emotion. Low-confidence handling
is therefore kept as an internal `uncertain` flag for tuning and diagnostics while
the API continues returning valid contract emotions. The threshold is configurable
with `AI_LOW_CONFIDENCE_THRESHOLD` and defaults to `0.4`.

Use validation/test metrics to decide whether this threshold should change. If the
product later needs a visible unknown state, update both BE schema and FE display
before changing the AI response enum.

## Operational boundaries

- AI never stores frames; frame bytes live only in memory.
- Signed video URLs are redacted from validation/log responses.
- HTTP auth uses `Authorization: Bearer <AI_API_KEY>`.
- `/health` succeeds only after FastAPI lifespan has loaded the model.
- CPU is supported. GPU/MPS can be selected through `AI_DEVICE`.
- Inference is serialized around model/detector calls with a lock, but video
  processing yields between sampled frames so realtime requests can interleave.

## Known limitations

- Haar detection is lightweight but weaker under blur, side faces, masks, low
  light and crowded classrooms.
- The model currently expects a local `.pth` state dict. Missing weights fail
  startup intentionally.
- No face identity tracking is performed. Student mapping must come from backend
  meeting context or a future explicit identity pipeline.
- Baseline metrics depend on the provided FER-2013 split and are not recalculated
  unless `scripts/evaluate.py` is run against a real dataset/checkpoint.
