# Báo cáo task AI

Tài liệu này ánh xạ các task AI-01 đến AI-09 sang artifact hiện có trong repo và
các bước cần chạy khi có dataset/checkpoint thật.

## AI-01 Baseline evaluation model hiện tại

Artifact hiện có:

- `evaluation/metrics.json`
- `evaluation/classification_report.txt`
- `evaluation/confusion_matrix.png`
- `scripts/evaluate.py`

Chạy lại baseline:

```bash
python scripts/evaluate.py \
  --data /path/to/fer2013 \
  --model models/emotion_model_best.pth \
  --output-dir evaluation
```

Output gồm accuracy, macro F1, F1 theo class và confusion matrix. Latency realtime
được đo bằng `scripts/benchmark_api.py`.

## AI-02 Data cleanup, augmentation, imbalance và fine-tune

`scripts/train.py` đã có augmentation cơ bản, `AdamW`, scheduler và tùy chọn
`--use-class-weights` để xử lý class imbalance.

Khuyến nghị chạy:

```bash
python scripts/train.py \
  --data /path/to/fer2013 \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --use-class-weights \
  --output models/emotion_model_best.pth
```

Checklist trước khi fine-tune:

- Kiểm tra đủ 7 thư mục class và class order không đổi.
- Loại ảnh hỏng/không đọc được khỏi dataset.
- Ghi lại seed, command, metrics và checksum checkpoint.

## AI-03 Chọn checkpoint và confidence calibration

Checkpoint tốt nhất nên chọn theo validation macro F1, không chỉ accuracy. Ngưỡng
confidence thấp dùng `AI_LOW_CONFIDENCE_THRESHOLD`, mặc định `0.4`.

Hiện API public chưa có emotion `unknown`; vì vậy low-confidence chỉ được gắn cờ
nội bộ là `uncertain`. Nếu cần hiển thị unknown, phải đổi schema BE/FE trước.

## AI-04 Tối ưu preprocessing, inference và batch

Đã có:

- Model load một lần trong lifespan, không load mỗi request.
- Validate image trước inference.
- Lock quanh model/detector để tránh race.
- Video sample frame theo `VIDEO_SAMPLE_FPS`.
- API `/v1/analyze/frames` cho batch frame JSON.

Benchmark:

```bash
python scripts/benchmark_api.py \
  --url http://127.0.0.1:9000 \
  --api-key "$AI_API_KEY" \
  --image /path/to/frame.jpg \
  --requests 100 \
  --concurrency 4
```

Output gồm p50/p95 latency, throughput FPS và số request lỗi.

## AI-05 AI batch API và mapping student -> result

`POST /v1/analyze/frames` nhận nhiều frame dạng base64. Mỗi item có thể mang
`student_id`, và response giữ nguyên mapping:

```json
{
  "session_id": "meeting-1",
  "results": [
    {"student_id": "student-1", "result": {"frame_id": "f-1"}}
  ]
}
```

AI không tự đoán danh tính từ mặt. Backend vẫn là nguồn sự thật về student và
meeting membership.

## AI-06 Video emotion analysis pipeline

`POST /v1/analyze/video` tải signed URL, lấy mẫu frame, detect/crop mọi face và
trả `sample_counts`, `timeline`, `summary`. Idempotency cache tránh xử lý lặp khi
backend retry cùng job.

## AI-07 Benchmark multi-student throughput/latency

Realtime benchmark dùng `scripts/benchmark_api.py`. Video throughput phụ thuộc số
face mỗi frame; đo trên sample meeting thật bằng cách chạy `/v1/analyze/video` và
ghi CPU/GPU/RAM ngoài process bằng công cụ hệ thống.

Chỉ số cần báo cáo:

- p50/p95 latency cho frame.
- Throughput frame/giây theo batch/concurrency.
- Thời gian xử lý video so với duration.
- CPU/GPU/RAM và thiết bị chạy.

## AI-08 Integration test live frames/video và tuning cuối

Contract tests nằm trong `tests/test_backend_api.py`, gồm realtime, no-face,
video, idempotency, timeout, signed URL redaction và batch frame mapping.

Chạy:

```bash
python -m pytest -q
```

Khi test với live camera/video thật, dùng thêm các case ảnh khó: blur, low-light,
no-face, nhiều face, crop lệch và frame lỗi.

## AI-09 Model freeze, reproducibility docs và regression

Sau khi chọn checkpoint:

```bash
python scripts/freeze_artifact.py \
  --model models/emotion_model_best.pth \
  --metrics evaluation/metrics.json \
  --output evaluation/artifact_manifest.json \
  --notes "final checkpoint for classroom integration"
```

Manifest lưu checksum checkpoint, class order, image size, phiên bản Python/PyTorch
và metrics. Regression tối thiểu cần chạy lại `pytest` và một sample set cố định
trước khi thay checkpoint production.
