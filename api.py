"""Standalone HTTP wrapper around dcv_vision.analyze_micrograph.

Stateless — nothing is written to disk or a database. This is a thin FastAPI
shell around a pure function; run it if you want to try the pipeline over
HTTP instead of calling analyze_micrograph() directly.

No auth, no rate limiting. Fine for local/trusted use; put it behind your
own auth and a reverse proxy before exposing it to the internet.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile

from dcv_vision.dcv import VisionAnalysisError, analyze_micrograph

app = FastAPI(title="dcv-vision")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
BAD_FILE_MESSAGE = "File must be a JPG or PNG image."


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Reads in fixed-size chunks and aborts as soon as the running total
    exceeds max_bytes, so an oversized upload is rejected without ever
    buffering more than ~max_bytes + one chunk in memory."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=400, detail=BAD_FILE_MESSAGE)
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/analyze-microscope")
async def analyze_microscope(file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=BAD_FILE_MESSAGE)

    image_bytes = await _read_capped(file, MAX_UPLOAD_BYTES)

    try:
        return analyze_micrograph(image_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail=BAD_FILE_MESSAGE)
    except VisionAnalysisError as e:
        raise HTTPException(status_code=422, detail=str(e))
