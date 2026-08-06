"""Micrograph -> quadrat D_CV extraction. A pure function with no framework
or storage dependency — feed it bytes, get a dict back.

The scientific definition (quadrat CV, no clipping) and every processing
parameter live in dcv_vision/config.py, not here, because changing them
silently changes what a previously-computed D_CV means.
"""
from __future__ import annotations

import base64
import struct

import cv2
import numpy as np

from dcv_vision.config import (
    GAUSSIAN_KERNEL,
    GRID_N,
    MAX_IMAGE_PIXELS,
    MIN_PARTICLE_AREA_PX,
    MORPH_OPEN_KERNEL_PX,
    RESIZE_LONG_EDGE_PX,
    THUMBNAIL_MAX_PX,
    TOPHAT_KERNEL_PX,
)


class VisionAnalysisError(Exception):
    """Image decoded fine, but no particles could be detected in it."""


def _peek_image_size(b: bytes) -> tuple[int, int] | None:
    """(width, height) read straight from a PNG/JPEG header, no pixel decode.

    Exists so a decompression-bomb upload (a small file whose declared
    dimensions decode to a huge canvas) can be rejected before cv2.imdecode
    allocates the full decoded array — checking dimensions on the decoded
    array is too late, the memory's already spent by then. Returns None for
    anything it doesn't recognize as PNG/JPEG; the caller falls back to the
    post-decode MAX_IMAGE_PIXELS check for those.
    """
    if b[:8] == b"\x89PNG\r\n\x1a\n" and b[12:16] == b"IHDR":
        return struct.unpack(">II", b[16:24])
    if b[:2] == b"\xff\xd8":  # JPEG: walk markers to the SOF segment
        i, n = 2, len(b)
        while i + 9 < n:
            if b[i] != 0xFF:
                i += 1
                continue
            m = b[i + 1]
            if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:  # no length field
                i += 2
                continue
            seg = struct.unpack(">H", b[i + 2 : i + 4])[0]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):  # SOF marker
                h, w = struct.unpack(">HH", b[i + 5 : i + 9])
                return w, h
            i += 2 + seg
    return None


def _resize_long_edge(img: np.ndarray, target_px: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_px / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


def _segment(corrected: np.ndarray, open_kernel: np.ndarray) -> tuple[np.ndarray, int]:
    """Otsu-threshold a background-corrected (top-hat or black-hat) image and
    clean up noise. Returns (final binary mask, count of components at or
    above MIN_PARTICLE_AREA_PX).
    """
    _, mask = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    final_mask = np.zeros_like(cleaned)
    n_particles = 0
    for label in range(1, n_labels):  # label 0 = background
        if stats[label, cv2.CC_STAT_AREA] >= MIN_PARTICLE_AREA_PX:
            final_mask[labels == label] = 255
            n_particles += 1
    return final_mask, n_particles


def _too_large_error(w: int, h: int) -> ValueError:
    return ValueError(f"Image is too large ({w}x{h} = {w * h:,}px, max {MAX_IMAGE_PIXELS:,}px).")


def analyze_micrograph(image_bytes: bytes) -> dict:
    # Checked from the header alone, before cv2 touches the bytes at all —
    # this is the actual defense against a decompression-bomb upload (a
    # small file that decodes to a huge canvas). Checking dimensions after
    # cv2.imdecode is too late: the full array is already allocated by then.
    peeked = _peek_image_size(image_bytes)
    if peeked is not None and peeked[0] * peeked[1] > MAX_IMAGE_PIXELS:
        raise _too_large_error(*peeked)

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("File must be a JPG or PNG image.")

    # Record the original resolution before touching anything — if input
    # resolution varied run to run, every pixel-unit kernel constant below
    # would be meaningless, so scale must be fixed before any kernel runs.
    orig_h, orig_w = img.shape[:2]

    # Fallback for formats _peek_image_size doesn't recognize (or a header
    # that lied about its own dimensions) — the memory's already spent at
    # this point, but this still stops the resize/blur/morphology below,
    # which all scale with pixel count, from running on a huge array.
    if orig_h * orig_w > MAX_IMAGE_PIXELS:
        raise _too_large_error(orig_w, orig_h)

    resized = _resize_long_edge(img, RESIZE_LONG_EDGE_PX)
    blurred = cv2.GaussianBlur(resized, GAUSSIAN_KERNEL, 0)

    # Flat/saturated image: with zero texture, no downstream morphology can
    # mean anything.
    if float(blurred.std()) < 1e-6:
        raise VisionAnalysisError("No particles detected. Check focus and lighting.")

    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_PX, TOPHAT_KERNEL_PX))
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL_PX, MORPH_OPEN_KERNEL_PX)
    )

    # Background (illumination) correction and polarity detection (which
    # side is the particle) are folded into one symmetric step. Using only
    # top-hat (bright particles) makes dark particles structurally
    # undetectable — their response vanishes — so black-hat (dark particles)
    # is always computed alongside it.
    #
    # The polarity decision is made on contrast (std of the corrected image),
    # not raw component count. The losing channel's background is nearly
    # flat, so its Otsu threshold degenerates toward zero — and at that
    # point, the handful of 1-to-a-few-pixel rings and noise left over from
    # resize/blur/morphology all get counted as "components," swamping the
    # real particle count. That was observed directly: comparing by
    # component count flipped polarity on the losing channel. A channel with
    # real particles, by contrast, has a clear brightness difference from
    # its background, which shows up as a reliably higher std. So the
    # higher-contrast channel is the one taken forward into Otsu + cleanup.
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, tophat_kernel)
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, tophat_kernel)
    bright_strength, dark_strength = float(tophat.std()), float(blackhat.std())

    if bright_strength >= dark_strength:
        corrected, polarity = tophat, "particles_bright"
    else:
        corrected, polarity = blackhat, "particles_dark"

    final_mask, n_particles = _segment(corrected, open_kernel)
    polarity_evidence = (
        f"Compared bright-region contrast (std={bright_strength:.2f}) against "
        f"dark-region contrast (std={dark_strength:.2f}); classified the "
        f"{'bright' if polarity == 'particles_bright' else 'dark'} side as particles."
    )

    if n_particles == 0:
        raise VisionAnalysisError("No particles detected. Check focus and lighting.")

    particle_bool = final_mask > 0
    area_fraction = float(particle_bool.mean())

    # Quadrat D_CV: coefficient of variation (std/mean) of per-cell particle
    # area fraction over a GRID_N x GRID_N grid. Not clipped — values above
    # 1 are possible and expected for highly clustered dispersion.
    h, w = particle_bool.shape
    cell_fractions = np.array(
        [
            particle_bool[h * i // GRID_N : h * (i + 1) // GRID_N, w * j // GRID_N : w * (j + 1) // GRID_N].mean()
            for i in range(GRID_N)
            for j in range(GRID_N)
        ]
    )
    mean_frac = float(cell_fractions.mean())
    d_cv = float(cell_fractions.std() / mean_frac) if mean_frac > 0 else 0.0

    thumb = _resize_long_edge(final_mask, THUMBNAIL_MAX_PX)
    ok, jpg = cv2.imencode(".jpg", thumb)
    processed_image_base64 = base64.b64encode(jpg.tobytes()).decode("ascii") if ok else ""

    return {
        "d_cv": round(d_cv, 4),
        "area_fraction": area_fraction,
        "n_grid": GRID_N,
        "polarity": polarity,
        "polarity_evidence": polarity_evidence,
        "processed_image_base64": processed_image_base64,
        "original_width": int(orig_w),
        "original_height": int(orig_h),
    }
