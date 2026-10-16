# Phân tích cảm xúc qua video

Tài liệu này mô tả contract HTTP, luồng xử lý, cấu hình và cách kiểm thử nhóm
chức năng phân tích biểu cảm khuôn mặt từ video trong OpenEmotion.

> Kết quả là dự đoán biểu cảm khuôn mặt của mô hình, không phải phép đo chắc chắn
> về trạng thái cảm xúc bên trong của một người.

## Phạm vi

Nhóm chức năng video cung cấp hai API:

| API | Mục đích | Idempotency |
| --- | --- | --- |
| `POST /v1/analyze/video` | Contract ổn định cho backend worker hiện tại | Bắt buộc header `Idempotency-Key` |
| `POST /v1/analyze/student-video` | Phân tích video đã gắn với một học sinh và trả timeline được nhóm theo cảm xúc | Không cache; backend chịu trách nhiệm lưu kết quả |

Cả hai API đều:

- tải video từ HTTPS URL thuộc allowlist;
- giới hạn kích thước, duration và thời gian xử lý;
- lấy mẫu frame, phát hiện mọi khuôn mặt và phân loại 7 biểu cảm;
- trả timestamp theo số giây tính từ đầu video;
- không suy đoán danh tính học sinh từ `face_id`;
- xóa file video tạm sau khi xử lý hoặc khi xảy ra lỗi.

## Luồng xử lý

```text
Backend tạo signed video URL
        ↓
OpenEmotion xác thực request và URL
        ↓
Tải video vào thư mục tạm có giới hạn
        ↓
Đọc metadata và kiểm tra duration
        ↓
Lấy mẫu frame → phát hiện khuôn mặt → crop
        ↓
EfficientNet-B0 dự đoán emotion + confidence
        ↓
Adapter chuẩn hóa nhãn và dựng JSON response
        ↓
Schema kiểm tra counts, timeline và các nhóm emotion
```

## Xác thực

Mọi API phân tích yêu cầu:

```http
Authorization: Bearer <AI_API_KEY>
```

Sai hoặc thiếu API key trả HTTP `401`.

## API tương thích backend: `/v1/analyze/video`

### Request

```http
POST /v1/analyze/video
Authorization: Bearer <AI_API_KEY>
Idempotency-Key: <analysis-job-id>
Content-Type: application/json
```

```json
{
  "recording_id": "recording-1",
  "video_url": "https://api.cloudinary.com/v1_1/demo/video/download?format=mp4&signature=...",
  "duration": 120.5
}
```

### Response `200`

```json
{
  "sample_counts": {
    "happy": 45,
    "neutral": 40,
    "sad": 15
  },
  "timeline": [
    {
      "emotion": "happy",
      "confidence": 0.92,
      "mock": false,
      "timestamp": 0.0,
      "student_id": null
    }
  ],
  "summary": "100 classified samples.",
  "mock": false
}
```

`sample_counts` là số mẫu nguyên, không phải phần trăm. Timeline có thể là tập
con đại diện nếu kết quả vượt 20.000 điểm. Backend tự tính phần trăm từ counts.

Kết quả thành công được cache theo `Idempotency-Key`. Signed URL có thể thay đổi
khi retry, nhưng `recording_id` và `duration` phải giữ nguyên.

## API video theo học sinh: `/v1/analyze/student-video`

### Request

```http
POST /v1/analyze/student-video
Authorization: Bearer <AI_API_KEY>
Content-Type: application/json
```

```json
{
  "recording_id": "recording-1",
  "student_id": "student-1",
  "video_url": "https://api.cloudinary.com/v1_1/demo/video/download?format=mp4&signature=...",
  "duration": 120.5,
  "sample_fps": 1.0
}
```

| Field | Bắt buộc | Quy tắc |
| --- | --- | --- |
| `recording_id` | Có | Chuỗi 1–36 ký tự |
| `student_id` | Không | Chuỗi tối đa 36 ký tự; được echo vào timeline |
| `video_url` | Có | HTTPS URL tối đa 16.384 ký tự, host phải thuộc allowlist |
| `duration` | Không | Số dương; nếu bỏ qua sẽ dùng duration từ metadata video |
| `sample_fps` | Không | `0 < sample_fps <= 5`, mặc định `1.0` |

### JSON response chuẩn

API trả trực tiếp object dưới đây, không bọc trong `data`:

```json
{
  "recording_id": "recording-1",
  "student_id": "student-1",
  "duration": 120.5,
  "sample_counts": {
    "happy": 1
  },
  "timeline": [
    {
      "timestamp": 0.0,
      "confidence": 0.91,
      "face_id": "1",
      "bbox": [120, 80, 160, 160],
      "uncertain": false,
      "emotion": "happy",
      "student_id": "student-1"
    }
  ],
  "emotion_timestamps": {
    "happy": [
      {
        "timestamp": 0.0,
        "confidence": 0.91,
        "face_id": "1",
        "bbox": [120, 80, 160, 160],
        "uncertain": false
      }
    ],
    "neutral": [],
    "sad": [],
    "angry": [],
    "surprised": [],
    "fearful": [],
    "disgusted": []
  },
  "summary": "1 classified samples.",
  "mock": false
}
```

### Quy tắc response

- Emotion public chỉ gồm: `happy`, `neutral`, `sad`, `angry`, `surprised`,
  `fearful`, `disgusted`.
- `sample_counts` có thể chỉ chứa các emotion đã xuất hiện.
- `emotion_timestamps` luôn có đủ 7 key, kể cả các danh sách rỗng.
- Mỗi phần tử trong `emotion_timestamps` phải khớp chính xác với một phần tử
  cùng emotion trong `timeline`.
- Timestamp không được âm hoặc vượt `duration`.
- `student_id` trong mọi timeline point phải khớp `student_id` cấp response.
- `confidence` nằm trong `[0, 1]`.
- `face_id` chỉ là định danh phát hiện trong frame, không phải định danh người
  hoặc tracking ID xuyên suốt video.
- `bbox` có dạng `[x, y, width, height]`.
- `uncertain=true` khi confidence thấp hơn ngưỡng cấu hình; emotion vẫn là một
  trong 7 nhãn public.

Nếu không phát hiện khuôn mặt, API vẫn trả `200`, với `sample_counts={}`,
`timeline=[]`, đủ 7 nhóm rỗng và summary `No faces detected in video.`.

## Mã lỗi HTTP

| Status | Ý nghĩa |
| --- | --- |
| `200` | Phân tích thành công, kể cả trường hợp không phát hiện khuôn mặt |
| `401` | API key thiếu hoặc không hợp lệ |
| `409` | Xung đột idempotency hoặc job đang xử lý; chỉ áp dụng `/v1/analyze/video` |
| `413` | HTTP request body vượt giới hạn byte |
| `422` | Request, video tải về, metadata hoặc kết quả phân tích sai contract |
| `500` | Lỗi suy luận nội bộ |
| `502` | Không tải được video từ nguồn |
| `504` | Tải hoặc phân tích video quá thời gian |

Response lỗi không phản chiếu signed URL, token, đường dẫn nội bộ hoặc thông báo
gốc từ thư viện native/model.

## Cấu hình

Các biến môi trường chính dùng prefix `AI_`:

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `AI_API_KEY` | Không có | Bearer token dùng chung với backend |
| `AI_MODEL_PATH` | `models/emotion_model_best.pth` | Checkpoint 7 lớp |
| `AI_DEVICE` | `cpu` | Thiết bị PyTorch, ví dụ `cpu` hoặc `cuda` |
| `AI_CACHE_PATH` | `outputs/ai-jobs.sqlite3` | Cache idempotency bền vững |
| `AI_DOWNLOAD_HOSTS` | Cloudinary API và CDN | JSON array chứa các host tải video được tin cậy |
| `AI_MAX_VIDEO_BYTES` | `94371840` | Giới hạn video 90 MiB |
| `AI_MAX_VIDEO_SECONDS` | `7200` | Duration tối đa 2 giờ |
| `AI_PROCESSING_TIMEOUT_SECONDS` | `110` | Ngân sách tải và xử lý |
| `AI_DOWNLOAD_TIMEOUT_SECONDS` | `30` | Timeout mỗi request tải video |
| `AI_LOW_CONFIDENCE_THRESHOLD` | `0.4` | Ngưỡng đánh dấu `uncertain` |

Allowlist host mặc định gồm `api.cloudinary.com` và `res.cloudinary.com`.

## Chạy API

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
cp .env.example .env
python -m uvicorn openemotion.api.app:create_app \
  --factory --app-dir src --host 127.0.0.1 --port 9000
```

API chỉ khởi động khi checkpoint tồn tại và load thành công. `GET /health` trả:

```json
{"status": "ok"}
```

## Kiểm thử

Chạy toàn bộ suite:

```bash
python -m pytest -q
```

Chạy contract test với backend nằm cạnh repo OpenEmotion:

```bash
BACKEND_ROOT=../BE/cnpm-be python -m pytest -q tests/test_backend_api.py
```

Test video bao phủ:

- download video và kiểm tra định dạng;
- decode MP4, sampling frame, face detection và inference;
- mapping nhãn model sang contract backend;
- nhiều khuôn mặt, không có khuôn mặt và timeline;
- giới hạn duration trước inference;
- schema JSON, auth, validation và redaction lỗi;
- timeout, cleanup file tạm và idempotency;
- tương thích `HttpAIClient`/schema của backend thật.

## Vị trí mã nguồn

| Thành phần | File |
| --- | --- |
| HTTP routes và request validation | `src/openemotion/api/app.py` |
| JSON/Pydantic contracts | `src/openemotion/api/schema.py` |
| Chuẩn hóa nhãn và dựng response | `src/openemotion/api/adapter.py` |
| Điều phối model và video processor | `src/openemotion/service/ai_service.py` |
| Decode, sampling và inference video | `src/openemotion/video/video_processor.py` |
| Download an toàn | `src/openemotion/api/download.py` |
| Idempotency cache | `src/openemotion/api/jobs.py` |
| API/contract tests | `tests/test_backend_api.py` |

Chi tiết tích hợp backend và kiến trúc tổng thể xem thêm tại
[`../backend-integration.md`](../backend-integration.md) và
[`../architecture.md`](../architecture.md).
