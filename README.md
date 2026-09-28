# dcv-vision

**Measure spatial dispersion from a microscope image.**

A deterministic OpenCV pipeline that converts a PNG or JPEG micrograph into
**D_CV**, a measure of how unevenly particles or voids are distributed across
an 8 × 8 grid. Lower values mean more even coverage; higher values mean more
spatial variation. Values can exceed 1.

Use it as a Python function or through an optional local HTTP API. The core
analysis has no database, network, or application-framework dependency. It
was built for the same materials project as
[formulation-bo](https://github.com/mthogeon0731/formulation-bo).

![Synthetic micrographs: uniform and clustered inputs with their detected masks](demo_dispersion.png)

*The example above is generated from synthetic images, not research photos.*

[Quick start](#quick-start) · [Python API](#python-api) ·
[HTTP API](#optional-http-api) · [D_CV definition](#how-d_cv-is-calculated) ·
[Tests](#tests) · [Security](SECURITY.md)

## What changed in v2

Real-photo observations motivated changes to particle selection, large
aggregate detection, and the handling of dense samples.

| Area | Previous behavior | Definition v2 |
|---|---|---|
| Particle selection | Automatic bright/dark detection | Explicit `"dark"` or `"bright"`; default: `"dark"` |
| Segmentation | Top-hat/black-hat background correction, then Otsu | Global Otsu after blur and polarity selection |
| Image scale | Fixed 1024-pixel long edge | Supported scale-bar normalization, with fixed-size fallback |
| Dense samples | Particle-fraction CV only | Void-fraction CV when particle coverage exceeds 50% |
| Result metadata | No definition-version field | `dcv_version=2` and `minority_phase` |

**v1 and v2 values are not directly interchangeable.** Reanalyze original
images with the same definition and acquisition conditions before combining
datasets. See [Migration from v1](#migration-from-v1) and the
[changelog](CHANGELOG.md).

## Quick start

Requires **Python 3.12 or later** for the pinned dependencies.

```bash
git clone https://github.com/mthogeon0731/dcv-vision
cd dcv-vision
python -m venv .venv
```

Activate the environment for your platform:

| Platform | Command |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

Then install and run the synthetic demo:

```bash
python -m pip install -r requirements.txt
python demo.py
```

The demo writes `demo_dispersion.png`. With the pinned versions:

| Synthetic fixture | D_CV |
|---|---:|
| Uniform particle placement | ≈ 0.0031 |
| Clustered particle placement | ≈ 3.9167 |

These numbers illustrate the metric; they are not real-photo accuracy results.

## Python API

```python
from pathlib import Path
from dcv_vision import analyze_micrograph

result = analyze_micrograph(
    Path("micrograph.jpg").read_bytes(),
    particles="dark",
)

print(result["d_cv"])
print(result["dcv_version"], result["minority_phase"], result["polarity"])
```

Choose `particles="dark"` for dark particles on a brighter background, or
`particles="bright"` for bright particles on a darker background. There is
no automatic polarity mode.

| Result field | Meaning |
|---|---|
| `d_cv` | Quadrat coefficient of variation, rounded to four decimals |
| `dcv_version` | Measurement definition; currently `2` |
| `minority_phase` | `"particle"` or `"void"`: the phase used to calculate CV |
| `area_fraction` | Overall **particle** coverage, even when the metric uses voids |
| `n_grid` | Grid size per side; currently `8` |
| `polarity` / `polarity_evidence` | Selected particle polarity and a readable explanation |
| `processed_image_base64` | Base64-encoded JPEG mask preview, maximum long edge 512 px |
| `original_width` / `original_height` | Input image dimensions in pixels |

The function returns a dictionary without storing the input or result.
It raises `ValueError` for invalid input and `VisionAnalysisError` when
meaningful particles cannot be detected.

## Optional HTTP API

Start the local server:

```bash
uvicorn api:app --host 127.0.0.1
```

In another terminal, send a multipart request:

```bash
curl -F "file=@micrograph.jpg" -F "particles=dark" http://127.0.0.1:8000/analyze-microscope
```

On Windows PowerShell, use `curl.exe` if `curl` resolves to a PowerShell alias.
The response contains the same fields as the Python API.

| Status | Meaning |
|---|---|
| `200` | Analysis completed |
| `400` | Invalid image input or particle polarity |
| `413` | Total request body exceeds 10 MiB, including multipart overhead |
| `422` | Low-contrast/no-particle image, or a missing required file |

The wrapper is intended for **local or trusted use**. It has no authentication
or rate limiting. The framework may temporarily spool uploads to disk; the
application does not persist images or results. Deployment requirements and
the limits of image-input protections are documented in [SECURITY.md](SECURITY.md).

## How D_CV is calculated

For each grid cell, let `p_i` be the particle area fraction. Select the phase
using the overall particle coverage:

- Coverage **≤ 50%**: `q_i = p_i`, measuring particle fractions.
- Coverage **> 50%**: `q_i = 1 - p_i`, measuring void fractions.

```text
D_CV = std(q_i, ddof=0) / mean(q_i)
```

The calculation uses population standard deviation and does not clip the
result at 1. The implementation returns 0 if the selected phase has zero
mean coverage. Always inspect the mask when interpreting a measurement.

Using the minority phase makes the metric more sensitive to uneven empty
regions in dense samples, where particle fractions can approach saturation.

### Image processing

1. **Validate:** check PNG/JPEG headers and input resource limits.
2. **Normalize scale:** remove lower-right red annotations and use a
   supported scale bar; otherwise resize the long edge to 1024 px.
3. **Check contrast:** apply Gaussian blur and reject images with a
   99th-to-1st percentile grayscale range below 40.
4. **Segment:** apply global Otsu using the selected particle polarity.
5. **Clean and measure:** apply morphological opening, remove small
   components, and calculate the grid statistic.

### Acquisition assumptions

- **Scale bar:** a red bar in the bottom 15% and rightmost 40% of the image,
  assumed to represent **100 µm**. Its numeric label is not read.
  Normalization targets `1024 / 960` pixels per µm.
- **Sampling:** consistent magnification, field of view, and sampling protocol
  are still needed. Pixel normalization does not make different physical
  grid-cell sizes interchangeable.
- **Segmentation:** one grayscale particle phase with adequate contrast.
  Uneven illumination or multiple phases can invalidate global Otsu results.
- **Cleanup:** the 3-pixel opening and 20-pixel minimum component area remain
  provisional settings for a reference capture setup. Validate masks for
  your own optics.

No reproducible real-photo benchmark is included, and this repository makes
no accuracy or correlation claim for real samples.

## Migration from v1

1. Pass particle polarity explicitly, especially `particles="bright"` for
   bright-particle images; omitted polarity now means dark particles.
2. Save `dcv_version`, `minority_phase`, and `polarity` with each measurement.
3. Reanalyze original images before pooling previous and new results.
4. Recheck masks and update downstream thresholds or models for the new
   definition rather than assuming old numerical cutoffs still apply.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python tests/test_dcv.py
python tests/test_v2_security.py
```

The suite contains 12 baseline scenarios and 27 additional regression tests.
Fixtures are generated in memory and cover measurement ordering, determinism,
explicit polarity, dense samples, large aggregates, scale-bar behavior,
malformed images, resource limits, HTTP body limits, and metadata exclusion
from result previews.

The pinned environment was verified on Windows with Python 3.12. Other
operating systems have not been execution-tested for this update.

## License

[MIT](LICENSE).
