# Changes

## Unreleased — D_CV definition v2

This is a measurement-definition change, not a drop-in numerical update.

- Replace automatic particle polarity with particles="dark" or "bright"
  (default: dark).
- Remove top-hat/black-hat background correction and use global Otsu.
- Remove lower-right red annotations and normalize magnification when a
  supported 100 um red scale bar is present; otherwise use long-edge resize.
- Reject low-contrast images before segmentation.
- Use void-fraction CV above 50% particle coverage and particle-fraction CV
  otherwise. Add dcv_version=2 and minority_phase to results.
- Preserve input size protection and empty-file handling. Harden truncated
  headers, restrict decoded formats to PNG/JPEG, bound dimensions and
  normalized image size, and normalize native decode errors.
- Count actual HTTP body bytes, including absent or false Content-Length,
  through Starlette's request body limit middleware.
- Avoid repeated whole-image scans during component filtering.
- Update synthetic examples, tests, and dependency pins. The pinned NumPy
  version requires Python 3.12 or newer.
- Document the real-micrograph check behind v2 in the README: process,
  results on 13 photos, and an open global-Otsu segmentation failure under
  uneven illumination. No code or definition change.

### Migration

Pass particle polarity explicitly, store the definition version with each
result, and reanalyze original images before combining old and new datasets.
Bright-particle callers must pass particles="bright". The response's
area_fraction remains the particle coverage for both minority phases.

## Previous public version

The earlier pipeline used automatic polarity, top/black-hat background
correction, fixed long-edge resizing, and particle-fraction CV only.
Its results had no explicit definition-version field.
