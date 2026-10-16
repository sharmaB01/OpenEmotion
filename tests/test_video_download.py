import time

import httpx
import pytest

from openemotion.api.download import VideoDownloader
from openemotion.api.settings import APISettings


@pytest.fixture
def settings():
    return APISettings(_env_file=None, api_key="test-key", max_video_bytes=1024)


def download(
    settings,
    tmp_path,
    handler,
    url="https://api.cloudinary.com/video/download?format=webm&signature=private",
):
    return VideoDownloader(settings, httpx.MockTransport(handler)).download(
        url, tmp_path, time.monotonic() + 10
    )


def test_signed_download_redirect_and_extension(settings, tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        assert "authorization" not in request.headers
        if len(seen) == 1:
            return httpx.Response(
                302, headers={"location": "https://res.cloudinary.com/video/asset"}
            )
        return httpx.Response(
            200,
            content=b"webm-video",
            headers={"content-type": "application/octet-stream"},
        )

    path = download(settings, tmp_path, handler)
    assert path.suffix == ".webm" and path.read_bytes() == b"webm-video"
    assert len(seen) == 2


@pytest.mark.parametrize(
    "url",
    [
        "http://api.cloudinary.com/video.mp4",
        "https://127.0.0.1/video.mp4",
        "file:///private.mp4",
        "https://api.cloudinary.com.evil.test/video.mp4",
        "https://user:pass@api.cloudinary.com/video.mp4",
        "https://api.cloudinary.com:8443/video.mp4",
    ],
)
def test_invalid_download_target(settings, tmp_path, url):
    def no_request(request):
        pytest.fail("Disallowed URLs must be rejected before networking")

    with pytest.raises(ValueError, match="allowed download host"):
        download(settings, tmp_path, no_request, url)


def test_redirect_cannot_escape_allowed_hosts(settings, tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://127.0.0.1/private"})

    with pytest.raises(ValueError, match="allowed download host"):
        download(settings, tmp_path, handler)
    assert len(requests) == 1


@pytest.mark.parametrize("with_length", [True, False])
def test_download_byte_limit(settings, tmp_path, with_length):
    def handler(request):
        if with_length:
            return httpx.Response(200, content=b"a" * 1025)
        return httpx.Response(200, stream=httpx.ByteStream(b"a" * 1025))

    with pytest.raises(ValueError, match="byte limit"):
        download(settings, tmp_path, handler)


def test_empty_download_and_deadline(settings, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        download(settings, tmp_path, lambda request: httpx.Response(200, content=b""))
    downloader = VideoDownloader(
        settings, httpx.MockTransport(lambda request: httpx.Response(200))
    )
    with pytest.raises(TimeoutError):
        downloader.download(
            "https://api.cloudinary.com/video.mp4", tmp_path, time.monotonic() - 1
        )
