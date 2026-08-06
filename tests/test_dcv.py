"""Self-check for dcv_vision. No pytest — run with `python tests/test_dcv.py`.

Synthetic images are generated in-code with numpy/cv2; no external image
files are used.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  OK {msg}")
    else:
        print(f"  FAIL {msg}")
        failures.append(msg)


# ---- synthetic micrograph helpers ---------------------------------------

def _grid_centers(size: int, n_side: int, span: tuple[float, float] = (0.05, 0.95)) -> list[tuple[int, int]]:
    """Place n_side x n_side particle centers, spanning the whole canvas
    (span=(0.05, 0.95)) or a narrow sub-region for a clustered layout."""
    lo, hi = span
    xs = np.linspace(size * lo, size * hi, n_side)
    ys = np.linspace(size * lo, size * hi, n_side)
    return [(int(x), int(y)) for y in ys for x in xs]


def _make_micrograph(
    centers: list[tuple[int, int]],
    radius: int = 10,
    size: int = 640,
    bg: int = 60,
    fg: int = 220,
    gradient: int = 20,
) -> bytes:
    """Draw particles (fg) on a background (bg), add a gentle illumination
    gradient (vignetting stand-in), and PNG-encode."""
    canvas = np.full((size, size), bg, dtype=np.float64)
    grad = np.linspace(0, gradient, size)[None, :]  # brightens gently left -> right
    canvas += grad
    canvas = canvas.astype(np.uint8)
    for cx, cy in centers:
        cv2.circle(canvas, (cx, cy), radius, fg, thickness=-1)
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


UNIFORM_CENTERS = _grid_centers(640, 8, span=(0.07, 0.93))  # 64, spread evenly
CLUSTERED_CENTERS = _grid_centers(640, 8, span=(0.55, 0.72))  # 64, packed in one corner


def _extreme_bytes() -> bytes:
    """One of 64 grid cells nearly full, the rest empty — an extreme clustering case."""
    return _make_micrograph([(560, 560)], radius=35, size=640)


def _fake_png_header(width: int, height: int) -> bytes:
    """PNG signature + IHDR chunk only — enough for _peek_image_size to read
    width/height, but not a decodable image (no IDAT). Lets a bomb-sized
    declared resolution be tested without allocating real pixel data for it."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    length = struct.pack(">I", len(ihdr_data))
    return sig + length + b"IHDR" + ihdr_data


# ---- tests ----------------------------------------------------------------

def t_ordering():
    from dcv_vision import analyze_micrograph

    r_uniform = analyze_micrograph(_make_micrograph(UNIFORM_CENTERS))
    r_clustered = analyze_micrograph(_make_micrograph(CLUSTERED_CENTERS))
    check(
        r_uniform["d_cv"] < r_clustered["d_cv"],
        f"D_CV(uniform)={r_uniform['d_cv']} < D_CV(clustered)={r_clustered['d_cv']}",
    )


def t_determinism():
    from dcv_vision import analyze_micrograph

    img_bytes = _make_micrograph(CLUSTERED_CENTERS)
    r1 = analyze_micrograph(img_bytes)
    r2 = analyze_micrograph(img_bytes)
    check(r1["d_cv"] == r2["d_cv"], f"same image analyzed twice gives same d_cv: {r1['d_cv']} == {r2['d_cv']}")


def t_scale_over_one():
    from dcv_vision import analyze_micrograph

    r = analyze_micrograph(_extreme_bytes())
    check(r["d_cv"] > 1.0, f"extreme clustering gives d_cv={r['d_cv']} > 1.0 (not clipped)")


def t_error_non_image():
    from dcv_vision import analyze_micrograph

    try:
        analyze_micrograph(b"this is not an image file at all")
        check(False, "non-image bytes should raise ValueError")
    except ValueError as e:
        check("JPG or PNG" in str(e), f"non-image bytes error message: {e}")
    except Exception as e:
        check(False, f"wrong exception type for non-image bytes: {type(e).__name__}")


def t_error_blank_image():
    from dcv_vision import VisionAnalysisError, analyze_micrograph

    blank = np.full((640, 640), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    try:
        analyze_micrograph(buf.tobytes())
        check(False, "flat solid-color image should raise VisionAnalysisError")
    except VisionAnalysisError as e:
        check("No particles detected" in str(e), f"no-particle error message: {e}")


def t_polarity_symmetry():
    from dcv_vision import analyze_micrograph

    bright = analyze_micrograph(_make_micrograph(CLUSTERED_CENTERS, bg=60, fg=220))
    dark = analyze_micrograph(_make_micrograph(CLUSTERED_CENTERS, bg=220, fg=60))

    check(bright["polarity"] == "particles_bright", f"bright particles polarity={bright['polarity']}")
    check(dark["polarity"] == "particles_dark", f"dark particles polarity={dark['polarity']}")
    check(bright["area_fraction"] > 0.01, f"bright particles detected, area_fraction={bright['area_fraction']}")
    check(dark["area_fraction"] > 0.01, f"dark particles detected (black-hat path), area_fraction={dark['area_fraction']}")
    check(
        abs(bright["d_cv"] - dark["d_cv"]) < 0.05,
        f"d_cv is similar under brightness inversion: bright={bright['d_cv']} dark={dark['d_cv']}",
    )


def t_original_resolution():
    from dcv_vision import analyze_micrograph

    canvas = np.full((300, 500), 60, dtype=np.uint8)
    for cx, cy in _grid_centers(300, 5):
        cv2.circle(canvas, (int(cx * 500 / 300), cy), 8, 220, thickness=-1)
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    r = analyze_micrograph(buf.tobytes())
    check(r["original_width"] == 500, f"original_width={r['original_width']}")
    check(r["original_height"] == 300, f"original_height={r['original_height']}")


def t_header_peek_matches_real_image():
    """Sanity check that _peek_image_size actually parses PNG headers,
    rather than the oversized-header test below passing vacuously because
    it always returns None and everything falls through to the (working,
    but too-late) post-decode check."""
    from dcv_vision.dcv import _peek_image_size

    dims = _peek_image_size(_make_micrograph(UNIFORM_CENTERS))  # 640x640 PNG
    check(dims == (640, 640), f"_peek_image_size on a real 640x640 PNG: {dims}")


def t_header_peek_blocks_before_decode():
    """The actual defense against a decompression bomb: a bomb-sized image
    must be rejected from the header alone, before cv2.imdecode ever runs
    (checking the decoded array's shape is too late — decoding it is the
    expensive part). Proven here by making cv2.imdecode raise if called at
    all, using a fake header with no real pixel data behind it."""
    import dcv_vision.dcv as dcv_module
    from dcv_vision.config import MAX_IMAGE_PIXELS

    side = int((MAX_IMAGE_PIXELS * 1.2) ** 0.5)  # comfortably over the cap
    fake_bomb = _fake_png_header(side, side)

    def _fail_if_called(*a, **kw):
        raise AssertionError("cv2.imdecode must not be called for an oversized header")

    original_imdecode = dcv_module.cv2.imdecode
    dcv_module.cv2.imdecode = _fail_if_called
    try:
        try:
            dcv_module.analyze_micrograph(fake_bomb)
            check(False, f"{side}x{side} declared header should be rejected before decode")
        except ValueError as e:
            check("too large" in str(e), f"oversized-header error message: {e}")
        except AssertionError as e:
            check(False, str(e))
    finally:
        dcv_module.cv2.imdecode = original_imdecode


def t_pixel_cap_rejected_real_image():
    """End-to-end version with a real, fully decodable oversized PNG (as
    opposed to the fake-header unit test above) — confirms the whole path
    behaves the same way on genuine image bytes, not just a crafted header."""
    from dcv_vision import analyze_micrograph
    from dcv_vision.config import MAX_IMAGE_PIXELS

    side = int((MAX_IMAGE_PIXELS * 1.2) ** 0.5)
    huge = np.full((side, side), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", huge)
    assert ok
    try:
        analyze_micrograph(buf.tobytes())
        check(False, f"{side}x{side} real image (> MAX_IMAGE_PIXELS) should be rejected")
    except ValueError as e:
        check("too large" in str(e), f"oversized real-image error message: {e}")


def t_oversized_content_length_rejected():
    """A large enough upload carries a Content-Length header Starlette would
    otherwise use to fully buffer the multipart body before the endpoint
    even runs; the middleware in api.py checks that header first."""
    from fastapi.testclient import TestClient

    from api import app, MAX_UPLOAD_BYTES

    client = TestClient(app)
    oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    resp = client.post(
        "/analyze-microscope",
        files={"file": ("huge.png", oversized, "image/png")},
    )
    check(
        resp.status_code == 413,
        f"upload with Content-Length over MAX_UPLOAD_BYTES rejected before body parsing: {resp.status_code}",
    )


def t_endpoint_passthrough():
    from fastapi.testclient import TestClient

    from api import app

    client = TestClient(app)
    img_bytes = _make_micrograph(UNIFORM_CENTERS)
    resp = client.post(
        "/analyze-microscope",
        files={"file": ("micrograph.png", img_bytes, "image/png")},
    )
    check(resp.status_code == 200, f"endpoint returns 200: {resp.status_code} {resp.text[:200]}")
    body = resp.json()
    for key in (
        "d_cv",
        "area_fraction",
        "n_grid",
        "polarity",
        "polarity_evidence",
        "processed_image_base64",
        "original_width",
        "original_height",
    ):
        check(key in body, f"response includes {key}")

    bad = client.post(
        "/analyze-microscope",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    check(bad.status_code == 400, f"non-image upload returns 400: {bad.status_code}")


tests = [
    ("1. ordering (uniform < clustered)", t_ordering),
    ("2. determinism (same input -> same d_cv)", t_determinism),
    ("3. scale check (d_cv > 1.0 allowed)", t_scale_over_one),
    ("4a. error path (non-image -> 400-class)", t_error_non_image),
    ("4b. error path (flat solid color -> 422-class)", t_error_blank_image),
    ("5. polarity symmetry (top-hat and black-hat both work)", t_polarity_symmetry),
    ("6. original resolution in response", t_original_resolution),
    ("7a. header peek matches a real image's dimensions", t_header_peek_matches_real_image),
    ("7b. header peek blocks a bomb before cv2.imdecode runs", t_header_peek_blocks_before_decode),
    ("7c. real oversized image rejected end-to-end", t_pixel_cap_rejected_real_image),
    ("8. oversized Content-Length rejected before body parsing", t_oversized_content_length_rejected),
    ("9. endpoint passthrough", t_endpoint_passthrough),
]

for name, fn in tests:
    print(f"\n[{name}]")
    try:
        fn()
    except Exception as e:
        print(f"  ERROR {type(e).__name__}: {e}")
        failures.append(f"{name}: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} failed:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All dcv_vision tests passed.")
    sys.exit(0)
