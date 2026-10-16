"""Bounded downloads of backend-generated Cloudinary URLs."""

import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx

from openemotion.config import SUPPORTED_VIDEO_EXTENSIONS


def checked_url(url: str, hosts: list[str]) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Video URL must use HTTPS on an allowed download host")
    return url


def video_suffix(url: str, content_type: str) -> str:
    parsed = urlsplit(url)
    # private_download_url uses /video/download?format=mp4 rather than a file path.
    format_value = parse_qs(parsed.query).get("format", [""])[0]
    candidates = ["." + format_value.lower(), Path(parsed.path).suffix.lower()]
    types = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
    }
    candidates.append(types.get(content_type.split(";")[0].strip().lower(), ""))
    for suffix in candidates:
        if suffix in SUPPORTED_VIDEO_EXTENSIONS:
            return suffix
    raise ValueError("Unsupported video format")


class VideoDownloader:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.transport = transport

    def download(self, url: str, directory: Path, deadline: float) -> Path:
        initial_url = checked_url(url, self.settings.download_hosts)
        with httpx.Client(
            transport=self.transport, follow_redirects=False, trust_env=False
        ) as client:
            for _ in range(5):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Video processing deadline exceeded")
                timeout = min(self.settings.download_timeout_seconds, remaining)
                with client.stream("GET", url, timeout=timeout) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        url = checked_url(
                            urljoin(url, response.headers["location"]),
                            self.settings.download_hosts,
                        )
                        continue
                    response.raise_for_status()
                    if (
                        int(response.headers.get("content-length", "0"))
                        > self.settings.max_video_bytes
                    ):
                        raise ValueError("Video download exceeds byte limit")
                    suffix = video_suffix(
                        initial_url, response.headers.get("content-type", "")
                    )
                    destination = directory / ("recording" + suffix)
                    size = 0
                    with destination.open("wb") as output:
                        for chunk in response.iter_bytes(chunk_size=64 * 1024):
                            if time.monotonic() > deadline:
                                raise TimeoutError("Video processing deadline exceeded")
                            size += len(chunk)
                            if size > self.settings.max_video_bytes:
                                raise ValueError("Video download exceeds byte limit")
                            output.write(chunk)
                    if not size:
                        raise ValueError("Video download is empty")
                    return destination
        raise ValueError("Too many video download redirects")
