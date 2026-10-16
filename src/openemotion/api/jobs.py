"""Durable idempotency for successful video results; failures may be retried."""

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException

from openemotion.api.schema import BatchResult


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS video_jobs (
                key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                owner TEXT NOT NULL, expires REAL NOT NULL, result TEXT
            )""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def claim(self, key: str, recording_id: str, duration: float, lease: float):
        # Signed URLs expire/rotate on backend retries and are not the asset identity.
        fingerprint = hashlib.sha256(
            json.dumps([recording_id, duration]).encode()
        ).hexdigest()
        owner = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT fingerprint, expires, result FROM video_jobs WHERE key=?",
                (key,),
            ).fetchone()
            if row:
                if row[0] != fingerprint:
                    raise HTTPException(
                        409, "Idempotency key belongs to another request"
                    )
                if row[2] is not None:
                    return None, BatchResult.model_validate_json(row[2])
                if row[1] > time.time():
                    raise HTTPException(
                        409, "Video analysis is already processing; retry later"
                    )
            db.execute(
                "INSERT OR REPLACE INTO video_jobs VALUES (?, ?, ?, ?, NULL)",
                (key, fingerprint, owner, time.time() + lease),
            )
        return owner, None

    def finish(self, key: str, owner: str, result: BatchResult):
        with self.connect() as db:
            changed = db.execute(
                "UPDATE video_jobs SET result=? WHERE key=? AND owner=?",
                (result.model_dump_json(), key, owner),
            ).rowcount
            if changed != 1:
                raise HTTPException(409, "Video analysis lease expired; retry request")

    def release(self, key: str, owner: str):
        with self.connect() as db:
            db.execute(
                "DELETE FROM video_jobs WHERE key=? AND owner=? AND result IS NULL",
                (key, owner),
            )
