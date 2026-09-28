"""Optional local/trusted HTTP demo. Add authentication and rate limits
before exposing it publicly. Uploads may be temporarily spooled by the
framework, but no research data or results are persistently stored.
"""
from __future__ import annotations

from fastapi import FastAPI, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from dcv_vision.config import MAX_UPLOAD_BYTES
from dcv_vision.dcv import BAD_FILE_MESSAGE, VisionAnalysisError, analyze_micrograph

app = FastAPI(title="dcv-vision")
# Counts received bytes before multipart parsing, including absent or
# understated Content-Length. The limit includes multipart overhead.
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_UPLOAD_BYTES)

READ_CHUNK_BYTES = 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Also bound the endpoint's own buffer, independently of middleware."""
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
async def analyze_microscope(file: UploadFile, particles: str = Form("dark")):
    if particles not in ("dark", "bright"):
        raise HTTPException(status_code=400, detail="particles must be 'dark' or 'bright'.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=BAD_FILE_MESSAGE)
    image_bytes = await _read_capped(file, MAX_UPLOAD_BYTES)
    try:
        return await run_in_threadpool(analyze_micrograph, image_bytes, particles)
    except ValueError:
        raise HTTPException(status_code=400, detail=BAD_FILE_MESSAGE) from None
    except VisionAnalysisError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
