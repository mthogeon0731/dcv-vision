"""Standalone HTTP wrapper around dcv_vision.analyze_micrograph.

Stateless — nothing is written to disk or a database. This is a thin FastAPI
shell around a pure function; run it if you want to try the pipeline over
HTTP instead of calling analyze_micrograph() directly.

No auth, no rate limiting, no real ceiling on request body size — see the
two caveats on MAX_UPLOAD_BYTES below. Fine for local/trusted use; put it
behind your own auth and a reverse proxy before exposing it to the internet.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from dcv_vision.dcv import VisionAnalysisError, analyze_micrograph

app = FastAPI(title="dcv-vision")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
BAD_FILE_MESSAGE = "File must be a JPG or PNG image."


@app.middleware("http")
async def reject_oversized_content_length(request: Request, call_next):
    """Rejects on the declared Content-Length before Starlette parses the
    multipart body — cheap, but only catches clients that send an honest
    Content-Length. A client using chunked transfer-encoding (no
    Content-Length) or one that just lies about a smaller value isn't
    caught here; enforcing that needs a reverse proxy or platform-level
    body-size limit in front of this process, not application code.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = None
        if declared is not None and declared > MAX_UPLOAD_BYTES:
            return JSONResponse({"detail": BAD_FILE_MESSAGE}, status_code=413)
    return await call_next(request)


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Reads in fixed-size chunks and aborts as soon as the running total
    exceeds max_bytes. This bounds how much *this function* buffers — it
    does not stop Starlette's multipart parser from having already received
    the full request body onto disk (a SpooledTemporaryFile) before this
    endpoint even runs; reject_oversized_content_length above is what
    prevents that for well-behaved clients."""
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
