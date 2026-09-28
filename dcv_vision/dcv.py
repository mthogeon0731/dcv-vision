"""PNG/JPEG bytes to versioned minority-phase quadrat D_CV, without storage."""
from __future__ import annotations

import base64
import struct

import cv2
import numpy as np

from dcv_vision.config import (
    DCV_DEFINITION_VERSION,
    GAUSSIAN_KERNEL,
    GRID_N,
    MAX_IMAGE_EDGE_PX,
    MAX_IMAGE_PIXELS,
    MAX_PROCESSING_PIXELS,
    MAX_UPLOAD_BYTES,
    MIN_CONTRAST,
    MIN_PARTICLE_AREA_PX,
    MORPH_OPEN_KERNEL_PX,
    RESIZE_LONG_EDGE_PX,
    SCALE_BAR_MAX_FRAC,
    SCALE_BAR_MIN_FRAC,
    SCALE_BAR_UM,
    TARGET_PX_PER_UM,
    THUMBNAIL_MAX_PX,
)

BAD_FILE_MESSAGE = "File must be a JPG or PNG image."


class VisionAnalysisError(Exception):
    """The image decoded, but meaningful particles could not be detected."""


def _peek_image_size(b: bytes) -> tuple[int, int] | None:
    """Read PNG/JPEG dimensions before decoding; reject truncated headers.

    This is an early resource guard, not a replacement for a full decoder.
    Unsupported formats return None and are rejected by the caller.
    """
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        if len(b) < 24 or b[8:16] != b"\x00\x00\x00\x0dIHDR":
            raise ValueError(BAD_FILE_MESSAGE)
        return struct.unpack(">II", b[16:24])
    if b[:2] != b"\xff\xd8":
        return None

    i, n = 2, len(b)
    while i < n:
        if b[i] != 0xFF:
            raise ValueError(BAD_FILE_MESSAGE)
        while i < n and b[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = b[i]
        i += 1
        if marker in (0x00, 0xD8, 0xD9, 0xDA):
            break  # no SOF before end-of-image or start-of-scan
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue  # standalone markers have no length field
        if i + 2 > n:
            break
        length = struct.unpack(">H", b[i:i + 2])[0]
        if length < 2 or i + length > n:
            break
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if length < 8 or not b[i + 7] or length != 8 + 3 * b[i + 7]:
                break
            h, w = struct.unpack(">HH", b[i + 3:i + 7])
            return w, h
        i += length
    raise ValueError(BAD_FILE_MESSAGE)


def _validate_dimensions(w: int, h: int, max_pixels: int) -> None:
    if w < GRID_N or h < GRID_N:
        raise ValueError("Image dimensions must each cover the 8 x 8 grid.")
    if w * h > max_pixels or max(w, h) > MAX_IMAGE_EDGE_PX:
        raise ValueError("Image is too large for analysis.")


def _resize_checked(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    target_w, target_h = round(w * scale), round(h * scale)
    _validate_dimensions(target_w, target_h, MAX_PROCESSING_PIXELS)
    return cv2.resize(
        img, (target_w, target_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )


def _resize_long_edge(img: np.ndarray, target_px: int) -> np.ndarray:
    return _resize_checked(img, target_px / max(img.shape[:2]))


def _segment(corrected: np.ndarray, open_kernel: np.ndarray) -> tuple[np.ndarray, int]:
    """Global Otsu followed by opening and small-component removal."""
    _, mask = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    # One lookup per pixel avoids rescanning the entire image per component.
    keep = stats[:, cv2.CC_STAT_AREA] >= MIN_PARTICLE_AREA_PX
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255, int(keep.sum())


def _remove_overlay(bgr: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Remove lower-right red annotations; measure the longest red bar run."""
    h, w = bgr.shape[:2]
    # Channel views preserve uint8 instead of allocating three integer arrays.
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    red = ((r > 200) & (g < 60) & (b < 60)).astype(np.uint8)
    red[:int(h * 0.85), :] = 0
    red[:, :int(w * 0.6)] = 0
    if not red.any():
        return bgr, None
    length = 0.0
    for row in red[int(h * 0.85):]:
        d = np.diff(np.concatenate(([0], row, [0])))
        runs = np.where(d == -1)[0] - np.where(d == 1)[0]
        if len(runs):
            length = max(length, float(runs.max()))
    bar_px = length if SCALE_BAR_MIN_FRAC * w <= length <= SCALE_BAR_MAX_FRAC * w else None
    clean = cv2.inpaint(bgr, cv2.dilate(red, np.ones((9, 9), np.uint8)), 5, cv2.INPAINT_TELEA)
    return clean, bar_px


def analyze_micrograph(image_bytes: bytes, particles: str = "dark") -> dict:
    """Analyze a micrograph with explicit dark/bright particle polarity.

    v2 values are not directly comparable to v1. No image, filename, or
    metadata is stored; the returned preview is encoded from the binary mask.
    """
    if particles not in ("dark", "bright"):
        raise ValueError("particles must be 'dark' or 'bright'.")
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(BAD_FILE_MESSAGE)
    peeked = _peek_image_size(image_bytes)
    if peeked is None:
        raise ValueError(BAD_FILE_MESSAGE)
    _validate_dimensions(*peeked, MAX_IMAGE_PIXELS)
    try:
        bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        raise ValueError(BAD_FILE_MESSAGE) from None
    if bgr is None:
        raise ValueError(BAD_FILE_MESSAGE)
    orig_h, orig_w = bgr.shape[:2]
    _validate_dimensions(orig_w, orig_h, MAX_IMAGE_PIXELS)

    bgr, bar_px = _remove_overlay(bgr)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = (TARGET_PX_PER_UM / (bar_px / SCALE_BAR_UM)
             if bar_px else RESIZE_LONG_EDGE_PX / max(orig_h, orig_w))
    resized = _resize_checked(img, scale)
    blurred = cv2.GaussianBlur(resized, GAUSSIAN_KERNEL, 0)
    contrast = float(np.percentile(blurred, 99) - np.percentile(blurred, 1))
    if float(blurred.std()) < 1e-6 or contrast < MIN_CONTRAST:
        raise VisionAnalysisError("No particles detected. Check focus and lighting.")

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL_PX, MORPH_OPEN_KERNEL_PX)
    )
    # Top/black-hat can erase aggregates larger than its kernel. v2 instead
    # thresholds the blurred grayscale image with user-selected polarity.
    corrected = 255 - blurred if particles == "dark" else blurred
    final_mask, n_particles = _segment(corrected, open_kernel)
    if n_particles == 0:
        raise VisionAnalysisError("No particles detected. Check focus and lighting.")

    particle_bool = final_mask > 0
    area_fraction = float(particle_bool.mean())
    h, w = particle_bool.shape
    cell_fractions = np.array([
        particle_bool[h * i // GRID_N:h * (i + 1) // GRID_N,
                      w * j // GRID_N:w * (j + 1) // GRID_N].mean()
        for i in range(GRID_N) for j in range(GRID_N)
    ])
    minority = "particle" if area_fraction <= 0.5 else "void"
    if minority == "void":
        cell_fractions = 1.0 - cell_fractions
    mean_frac = float(cell_fractions.mean())
    d_cv = float(cell_fractions.std() / mean_frac) if mean_frac > 0 else 0.0

    thumb = _resize_long_edge(final_mask, THUMBNAIL_MAX_PX)
    ok, jpg = cv2.imencode(".jpg", thumb)
    return {
        "d_cv": round(d_cv, 4),
        "dcv_version": DCV_DEFINITION_VERSION,
        "area_fraction": area_fraction,
        "n_grid": GRID_N,
        "minority_phase": minority,
        "polarity": "particles_dark" if particles == "dark" else "particles_bright",
        "polarity_evidence": f"User selected {particles} particles.",
        "processed_image_base64": base64.b64encode(jpg.tobytes()).decode("ascii") if ok else "",
        "original_width": int(orig_w),
        "original_height": int(orig_h),
    }
