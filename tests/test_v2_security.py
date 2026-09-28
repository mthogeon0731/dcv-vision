"""Regression and hostile-input checks; all fixtures are generated in memory."""
from __future__ import annotations

import asyncio
import base64
import io
import math
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from api import MAX_UPLOAD_BYTES, _read_capped, app
from dcv_vision import VisionAnalysisError, analyze_micrograph
from dcv_vision import dcv as pipeline
from dcv_vision.config import MAX_PROCESSING_PIXELS


def encode(img: np.ndarray, extension: str = ".png") -> bytes:
    ok, data = cv2.imencode(extension, img)
    if not ok:
        raise RuntimeError("Fixture encoding failed")
    return data.tobytes()


def micrograph() -> np.ndarray:
    img = np.full((384, 512, 3), 225, np.uint8)
    for y in (60, 140, 220, 290):
        for x in (60, 150, 240, 330, 410):
            cv2.circle(img, (x, y), 12, (30, 30, 30), -1)
    return img


def png_header(w: int, h: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h)


class MeasurementTests(unittest.TestCase):
    def test_default_dark_and_version(self):
        result = analyze_micrograph(encode(micrograph()))
        self.assertEqual(result["polarity"], "particles_dark")
        self.assertEqual(result["dcv_version"], 2)
        self.assertEqual(result["minority_phase"], "particle")
        self.assertTrue(math.isfinite(result["d_cv"]))

    def test_minority_phase_formula_with_known_grid(self):
        # Exactly one of 64 cells occupied (or vacant) has CV sqrt(63).
        for dense in (False, True):
            with self.subTest(dense=dense):
                mask = np.full((64, 64), 255 if dense else 0, np.uint8)
                mask[:8, :8] = 0 if dense else 255
                with patch.object(pipeline, "_segment", return_value=(mask, 1)):
                    result = analyze_micrograph(encode(micrograph()))
                self.assertAlmostEqual(result["d_cv"], math.sqrt(63), places=4)
                self.assertEqual(result["minority_phase"], "void" if dense else "particle")

    def test_half_coverage_uses_particle_phase(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[:, :32] = 255
        with patch.object(pipeline, "_segment", return_value=(mask, 1)):
            result = analyze_micrograph(encode(micrograph()))
        self.assertEqual(result["minority_phase"], "particle")
        self.assertEqual(result["d_cv"], 1.0)

    def test_dense_micrograph_uses_void(self):
        img = np.full((512, 512), 30, np.uint8)
        img[64:192, 64:192] = 225
        result = analyze_micrograph(encode(img))
        self.assertGreater(result["area_fraction"], 0.5)
        self.assertEqual(result["minority_phase"], "void")
        self.assertGreater(result["d_cv"], 1.0)

    def test_large_aggregate_is_retained(self):
        img = np.full((512, 512), 225, np.uint8)
        img[64:320, 64:320] = 30
        result = analyze_micrograph(encode(img))
        self.assertAlmostEqual(result["area_fraction"], 0.25, delta=0.01)

    def test_low_contrast_rejected(self):
        img = np.random.default_rng(4).integers(120, 137, (256, 256), dtype=np.uint8)
        with self.assertRaises(VisionAnalysisError):
            analyze_micrograph(encode(img))

    def test_red_bar_alone_is_not_a_particle(self):
        img = np.full((384, 512, 3), 225, np.uint8)
        img[350:354, 370:470] = (0, 0, 255)
        with self.assertRaises(VisionAnalysisError):
            analyze_micrograph(encode(img))

    def test_scale_bar_measurement_and_magnification(self):
        # Same field sampled at two resolutions. Both should reach 546x410.
        for factor in (1, 2):
            with self.subTest(factor=factor):
                img = cv2.resize(micrograph(), None, fx=factor, fy=factor)
                img[350 * factor:354 * factor, 370 * factor:470 * factor] = (0, 0, 255)
                clean, bar_px = pipeline._remove_overlay(img)
                self.assertEqual(bar_px, 100 * factor)
                self.assertFalse(np.any((clean[..., 2] > 200) & (clean[..., 1] < 60)))
                with patch.object(pipeline.cv2, "GaussianBlur", wraps=cv2.GaussianBlur) as blur:
                    result = analyze_micrograph(encode(img))
                self.assertEqual(blur.call_args.args[0].shape, (410, 546))
                self.assertEqual(result["original_width"], 512 * factor)

    def test_unrecognized_bar_falls_back(self):
        img = micrograph()
        img[350:354, 450:455] = (0, 0, 255)  # too short to be a valid bar
        self.assertIsNone(pipeline._remove_overlay(img)[1])
        with patch.object(pipeline.cv2, "GaussianBlur", wraps=cv2.GaussianBlur) as blur:
            analyze_micrograph(encode(img))
        self.assertEqual(blur.call_args.args[0].shape, (768, 1024))

    def test_jpeg_and_png(self):
        for ext in (".png", ".jpg"):
            with self.subTest(ext=ext):
                payload = encode(micrograph(), ext)
                self.assertEqual(pipeline._peek_image_size(payload), (512, 384))
                self.assertTrue(math.isfinite(analyze_micrograph(payload)["d_cv"]))

    def test_preview_does_not_copy_input_metadata(self):
        # A JPEG comment simulates identifying source metadata.
        payload = encode(micrograph(), ".jpg")
        marker = b"SYNTHETIC_METADATA_SENTINEL"
        payload = payload[:2] + b"\xff\xfe" + struct.pack(">H", len(marker) + 2) + marker + payload[2:]
        result = analyze_micrograph(payload)
        preview = base64.b64decode(result["processed_image_base64"])
        self.assertNotIn(marker, preview)
        self.assertNotIn(marker.decode(), str(result))
        decoded = cv2.imdecode(np.frombuffer(preview, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertLessEqual(max(decoded.shape), 512)


class InputSecurityTests(unittest.TestCase):
    def test_invalid_polarity_before_decode(self):
        for particles in ("auto", "", "DARK"):
            with self.subTest(particles=particles):
                with patch.object(pipeline.cv2, "imdecode") as decode:
                    with self.assertRaises(ValueError):
                        analyze_micrograph(encode(micrograph()), particles)
                    decode.assert_not_called()

    def test_malformed_headers_are_value_errors(self):
        fixtures = [
            b"", b"garbage", b"\x89PNG\r\n\x1a\n",
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR",
            b"\xff\xd8", b"\xff\xd8\xff",
            b"\xff\xd8\xff\xe0\x00\x01",  # invalid segment length
            b"\xff\xd8\xff\xe0\xff\xff",  # segment exceeds available data
            b"\xff\xd8\xff\xda\x00\x02",  # scan before dimensions
            png_header(0, 100), png_header(1, 100), png_header(100, 0),
        ]
        for payload in fixtures:
            with self.subTest(length=len(payload)):
                with self.assertRaises(ValueError):
                    analyze_micrograph(payload)

    def test_unsupported_format_even_if_decodable(self):
        with self.assertRaises(ValueError):
            analyze_micrograph(encode(micrograph(), ".bmp"))

    def test_input_caps_before_decode(self):
        fixtures = [
            png_header(8000, 8000),
            png_header(20001, 100),
            b"x" * (MAX_UPLOAD_BYTES + 1),
        ]
        for payload in fixtures:
            with patch.object(pipeline.cv2, "imdecode") as decode:
                with self.assertRaises(ValueError):
                    analyze_micrograph(payload)
                decode.assert_not_called()

    def test_post_decode_cap_remains(self):
        oversized = np.zeros((8, 20001, 3), np.uint8)
        with patch.object(pipeline.cv2, "imdecode", return_value=oversized):
            with self.assertRaises(ValueError):
                analyze_micrograph(encode(micrograph()))

    def test_native_decode_exception_is_normalized(self):
        with patch.object(pipeline.cv2, "imdecode", side_effect=cv2.error("bad input")):
            with self.assertRaises(ValueError):
                analyze_micrograph(encode(micrograph()))

    def test_processing_cap_before_resize(self):
        with patch.object(pipeline.cv2, "resize") as resize:
            with self.assertRaises(ValueError):
                pipeline._resize_checked(np.zeros((64, 64), np.uint8), 100)
            resize.assert_not_called()
        self.assertLess(MAX_PROCESSING_PIXELS, 64_000_000)

    def test_scale_bar_path_cannot_bypass_processing_cap(self):
        img = np.full((1200, 100, 3), 225, np.uint8)
        img[1100:1104, 80:83] = (0, 0, 255)
        with patch.object(pipeline.cv2, "resize") as resize:
            with self.assertRaises(ValueError):
                analyze_micrograph(encode(img))
            resize.assert_not_called()

    def test_extreme_aspect_ratio_rejected(self):
        with self.assertRaises(ValueError):
            analyze_micrograph(encode(np.zeros((8, 10000), np.uint8)))

    def test_component_cleanup_preserves_large_components(self):
        img = np.zeros((100, 100), np.uint8)
        img[10:30, 10:30] = 255
        img[50, 50] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask, count = pipeline._segment(img, kernel)
        self.assertEqual(count, 1)
        self.assertEqual(mask[20, 20], 255)
        self.assertEqual(mask[50, 50], 0)


class HttpSecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_polarity_and_result_metadata(self):
        response = self.client.post("/analyze-microscope",
            files={"file": ("sample.png", encode(micrograph()), "image/png")},
            data={"particles": "dark"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dcv_version"], 2)
        self.assertEqual(response.json()["polarity"], "particles_dark")

    def test_invalid_polarity_is_400(self):
        response = self.client.post("/analyze-microscope",
            files={"file": ("sample.png", encode(micrograph()), "image/png")},
            data={"particles": "auto"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("particles", response.json()["detail"])

    def test_malformed_uploads_do_not_return_500(self):
        for payload in (b"", b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR", b"\xff\xd8\xff"):
            response = self.client.post("/analyze-microscope",
                files={"file": ("sample.png", payload, "image/png")})
            self.assertEqual(response.status_code, 400)

    def test_low_contrast_is_422(self):
        response = self.client.post("/analyze-microscope",
            files={"file": ("sample.png", encode(np.full((64, 64), 128, np.uint8)), "image/png")})
        self.assertEqual(response.status_code, 422)

    def test_endpoint_reader_has_independent_cap(self):
        upload = UploadFile(filename="fixture", file=io.BytesIO(b"a" * 32))
        try:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(_read_capped(upload, 16))
            self.assertEqual(error.exception.status_code, 400)
        finally:
            asyncio.run(upload.close())

    def test_actual_body_limit_without_or_with_false_length(self):
        # Use ASGI directly so a client cannot repair the deliberately false
        # headers. The parser must stop before the rest of the body is read.
        async def request(headers):
            sent, reads = [], 0
            part = (b"--boundary\r\nContent-Disposition: form-data; "
                    b'name="file"; filename="fixture.png"\r\n'
                    b"Content-Type: image/png\r\n\r\n")
            chunks = [part] + [b"x" * (1024 * 1024)] * 12
            async def receive():
                nonlocal reads
                reads += 1
                if reads <= len(chunks):
                    return {"type": "http.request", "body": chunks[reads - 1], "more_body": True}
                return {"type": "http.request", "body": b"\r\n--boundary--\r\n", "more_body": False}
            async def send(message):
                sent.append(message)
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": "POST", "scheme": "http", "path": "/analyze-microscope",
                "raw_path": b"/analyze-microscope", "query_string": b"",
                "headers": [(b"content-type", b"multipart/form-data; boundary=boundary")] + headers,
                "client": ("127.0.0.1", 1), "server": ("testserver", 80), "root_path": ""}
            await app(scope, receive, send)
            status = next(m["status"] for m in sent if m["type"] == "http.response.start")
            return status, reads, len(chunks)
        for headers in ([], [(b"content-length", b"1")]):
            with self.subTest(headers=headers):
                status, reads, total = asyncio.run(request(headers))
                self.assertEqual(status, 413)
                self.assertLess(reads, total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
