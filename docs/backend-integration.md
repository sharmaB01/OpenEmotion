# Tích hợp AI với backend

API tương thích `face-emotion-backend/app/integrations/ai.py` và
`app/integrations/ai_schema.py`. Backend không cần đổi request/response.
Các hàm Python `process_realtime_batch`, `process_video_meeting` cũ vẫn phục vụ
script nội bộ; HTTP API dùng `AIService.analyze_frame` và `analyze_video`.

## Chạy trên Windows

Từ thư mục có `pyproject.toml`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Đặt `AI_API_KEY` trong `.env` thành cùng giá trị với backend. Đặt trọng số
EfficientNet-B0 đã huấn luyện tại `models/emotion_model_best.pth`, hoặc sửa
`AI_MODEL_PATH` thành đường dẫn thực tế. Đây phải là state dict có đầu ra 7 lớp
theo thứ tự `angry, disgust, fear, happy, neutral, sad, surprise`.
API dừng khởi động nếu thiếu/sai trọng số; không tự tạo dự đoán ngẫu nhiên.

```powershell
.\.venv\Scripts\python.exe -m uvicorn openemotion.api.app:create_app --factory --app-dir src --host 127.0.0.1 --port 9000
```

Backend cần cấu hình:

```dotenv
AI_MODE=http
AI_BASE_URL=http://localhost:9000
AI_API_KEY=<cùng khóa với AI>
AI_TIMEOUT_SECONDS=120
FRAME_TIMEOUT_SECONDS=10
```

Chạy cả API backend và worker backend (`python -m app.worker`) để job video được
xử lý. Production dùng HTTPS theo quy tắc cấu hình của backend. Khi chạy trên máy
khác/container, thay hostname bằng địa chỉ AI mà backend truy cập được.

`GET /health` chỉ trả thành công sau khi model đã load. Các endpoint phân tích
yêu cầu `Authorization: Bearer <AI_API_KEY>`.

## Realtime

`POST /v1/analyze/frame` nhận multipart: `frame` JPEG/PNG, `frame_id` chuỗi,
`timestamp` ISO 8601 có timezone. Trả trực tiếp `FrameResult`, echo metadata.

- Chọn mặt lớn nhất; trả đủ 7 xác suất, confidence bằng xác suất cao nhất.
- Ánh xạ `disgust → disgusted`, `fear → fearful`, `surprise → surprised` chỉ tại
  adapter; không đổi thứ tự output model hoặc nhãn dataset.
- Không thấy mặt: HTTP 200, `face_detected=false`, `emotion=fail_detection`,
  `failure_reason=no_face`, confidence và mọi xác suất bằng 0, `face_id=null`.
- Ảnh lỗi: HTTP 422; lỗi suy luận: HTTP 500. Backend chuyển lỗi HTTP thành
  `failure_reason=ai_error` theo logic hiện tại.
- `face_id` là số thứ tự phát hiện trong frame chuyển thành chuỗi, không phải
  danh tính hay ID theo dõi xuyên các frame. Student/meeting do backend xác định.
- Giới hạn mặc định 512 KiB và 1920×1080 pixel. Frame chỉ nằm trong RAM.

`POST /v1/analyze/frames` là biến thể batch JSON cho nhiều frame realtime. Mỗi
item có `frame_id`, `timestamp`, `image` base64, `content_type` và `student_id`
tùy chọn. Response trả `{session_id, results}`; từng result giữ nguyên
`student_id` và bọc `FrameResult`. Endpoint này dành cho tối ưu batch hoặc luồng
mapping student từ backend, không thay thế contract multipart đơn frame hiện tại.

## Video

`POST /v1/analyze/video` nhận JSON `{recording_id, video_url, duration}` và header
`Idempotency-Key` là ID job của backend. Trả HTTP 200 trực tiếp `BatchResult`,
không trả 202/job_id và không bọc envelope.

- Tải ngay URL có chữ ký về thư mục tạm. Hỗ trợ endpoint Cloudinary
  `/video/download?format=...` và redirect đến host được phép. HTTPS, giới hạn
  byte và thời gian được kiểm tra. File tạm được xóa cả khi thành công hoặc lỗi.
- `AI_DOWNLOAD_HOSTS` mặc định là `api.cloudinary.com`, `res.cloudinary.com`.
  Thêm chính xác host CDN đáng tin cậy khi dùng cấu hình Cloudinary khác.
- Lấy 1 frame/giây, mỗi khuôn mặt hợp lệ là một mẫu. Đếm toàn bộ mẫu thành số
  nguyên; phần trăm do backend tính. Không suy đoán `student_id` từ `face_id`.
- Timeline chứa timestamp giây và confidence của từng mẫu. Tối đa 20.000 điểm
  trải đều trên danh sách mẫu; số đếm vẫn đầy đủ. Timestamp vượt duration do BE
  gửi sẽ bị từ chối. Không thấy mặt trả `{sample_counts: {}, timeline: []}`.
- SQLite tại `AI_CACHE_PATH` lưu kết quả thành công theo idempotency key, tồn tại
  qua restart. Retry với URL ký mới vẫn trả kết quả cũ cho cùng recording/duration.
  Dùng cùng key cho recording/duration khác: HTTP 409. Lỗi cho phép thử lại cùng key.
  Job đang chạy: HTTP 409; job bị crash có thể được nhận lại sau khi lease hết hạn.
  Worker cũ không được ghi đè kết quả của lease mới. Giữ file SQLite trên ổ bền vững.
- Mặc định ngân sách xử lý video 110 giây, dưới timeout BE 120 giây. Pipeline kiểm
  tra deadline giữa các thao tác/frame; không cưỡng bức ngắt một lệnh suy luận native
  đang chạy. Video dài/CPU chậm có thể timeout: tăng timeout cả hai bên và
  `JOB_LEASE_SECONDS` của backend tương ứng, hoặc dùng GPU.
- HTTP download lỗi: 502; timeout: 504; video/schema sai: 422; suy luận lỗi: 500.
  Response/log không chứa URL ký hay nội dung lỗi bên trong.

### Video record theo học sinh

`POST /v1/analyze/student-video` dùng khi backend đã biết video record thuộc về
một học sinh và muốn lấy timeline trực tiếp theo cảm xúc. Endpoint vẫn tải video
từ Cloudinary giống `/v1/analyze/video`, yêu cầu `Authorization: Bearer <AI_API_KEY>`
và nhận JSON:

```json
{
  "recording_id": "recording-1",
  "student_id": "student-1",
  "video_url": "https://api.cloudinary.com/v1_1/.../video/download?...",
  "duration": 120.5,
  "sample_fps": 1
}
```

`student_id` và `duration` là tùy chọn; nếu không gửi `duration`, AI dùng duration
đọc từ metadata video. Response trả:

- `timeline`: danh sách mẫu theo thời gian, gồm `timestamp`, `emotion`,
  `confidence`, `face_id`, `bbox`, `uncertain`, `student_id`.
- `emotion_timestamps`: cùng timeline nhưng group theo từng emotion để backend/FE
  dễ hiển thị các mốc thời gian của cảm xúc.
- `sample_counts`: số mẫu phân loại theo từng emotion trên toàn bộ video.

JSON response chuẩn (trả trực tiếp, không bọc `data`) có dạng:

```json
{
  "recording_id": "recording-1",
  "student_id": "student-1",
  "duration": 120.5,
  "sample_counts": {"happy": 1, "neutral": 1},
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
    "happy": [{"timestamp": 0.0, "confidence": 0.91, "face_id": "1", "bbox": [120, 80, 160, 160], "uncertain": false}],
    "neutral": [],
    "sad": [],
    "angry": [],
    "surprised": [],
    "fearful": [],
    "disgusted": []
  },
  "summary": "2 classified samples.",
  "mock": false
}
```

`emotion_timestamps` luôn có đủ 7 key theo template trên, kể cả khi danh sách
rỗng. Mỗi phần tử trong các nhóm phải khớp chính xác với phần tử tương ứng trong
`timeline`; response sai contract bị từ chối thay vì trả dữ liệu không nhất quán.
Giới hạn duration được kiểm tra từ metadata ngay sau khi mở video và trước khi
chạy face detection/model inference.

Endpoint này không cache theo `Idempotency-Key`; backend nên lưu kết quả nếu cần
tránh phân tích lại cùng recording. Contract cũ `/v1/analyze/video` vẫn giữ nguyên
để tương thích worker/backend hiện tại.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Test tích hợp gọi HTTP API bằng chính `HttpAIClient` và schema của backend khi repo
backend nằm cạnh repo AI theo cấu trúc workspace. Có thể đặt `BACKEND_ROOT` thành
thư mục backend chứa `app/` ở máy khác. Nếu thiếu backend, test đó được đánh dấu skip.
Test API dùng logits cố định; vẫn chạy decode JPEG/MP4, tiền xử lý, softmax,
adapter, download, cache và HTTP thật trong ASGI. Test model huấn luyện được đánh
dấu skip khi thiếu file trọng số; không coi test logits cố định là kiểm định độ chính xác AI.
