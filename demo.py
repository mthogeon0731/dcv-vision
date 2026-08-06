"""Runs the pipeline on two synthetic micrographs (uniform vs. clustered
particle placement) and writes demo_dispersion.png comparing the input to
what got detected.

No lab equipment or real photos needed — particles are drawn with cv2.circle
the same way the test suite builds its synthetic fixtures.
"""
from __future__ import annotations

import cv2
import numpy as np

from dcv_vision import analyze_micrograph

SIZE = 640


def _grid_centers(n_side: int, span: tuple[float, float]) -> list[tuple[int, int]]:
    lo, hi = span
    xs = np.linspace(SIZE * lo, SIZE * hi, n_side)
    ys = np.linspace(SIZE * lo, SIZE * hi, n_side)
    return [(int(x), int(y)) for y in ys for x in xs]


def _make_micrograph(centers: list[tuple[int, int]], radius: int = 10) -> tuple[np.ndarray, bytes]:
    canvas = np.full((SIZE, SIZE), 60, dtype=np.uint8)
    for cx, cy in centers:
        cv2.circle(canvas, (cx, cy), radius, 220, thickness=-1)
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return canvas, buf.tobytes()


def _labeled_panel(img: np.ndarray, label: str) -> np.ndarray:
    resized = cv2.resize(img, (300, 300))
    bordered = cv2.copyMakeBorder(resized, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    panel = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return panel


def main() -> None:
    uniform_img, uniform_bytes = _make_micrograph(_grid_centers(8, (0.07, 0.93)))
    clustered_img, clustered_bytes = _make_micrograph(_grid_centers(8, (0.55, 0.72)))

    uniform_result = analyze_micrograph(uniform_bytes)
    clustered_result = analyze_micrograph(clustered_bytes)

    print(f"uniform:   d_cv={uniform_result['d_cv']:.4f}  area_fraction={uniform_result['area_fraction']:.4f}")
    print(f"clustered: d_cv={clustered_result['d_cv']:.4f}  area_fraction={clustered_result['area_fraction']:.4f}")
    assert uniform_result["d_cv"] < clustered_result["d_cv"]

    def _decode_thumb(result: dict) -> np.ndarray:
        import base64

        raw = base64.b64decode(result["processed_image_base64"])
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    top = np.hstack(
        [
            _labeled_panel(uniform_img, "input: uniform"),
            _labeled_panel(_decode_thumb(uniform_result), f"detected  D_CV={uniform_result['d_cv']:.3f}"),
        ]
    )
    bottom = np.hstack(
        [
            _labeled_panel(clustered_img, "input: clustered"),
            _labeled_panel(_decode_thumb(clustered_result), f"detected  D_CV={clustered_result['d_cv']:.3f}"),
        ]
    )
    grid = np.vstack([top, bottom])
    cv2.imwrite("demo_dispersion.png", grid)
    print("wrote demo_dispersion.png")


if __name__ == "__main__":
    main()
