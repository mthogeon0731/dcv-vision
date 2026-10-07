# Security and data handling

The library processes image bytes locally. It does not make network calls,
load credential files, or store input images, filenames, metadata, or results.
The returned JPEG preview is encoded from a binary mask; source EXIF and
comments are not copied. The synthetic demo writes its own generated PNG.

The optional HTTP wrapper is for local or trusted use. Multipart uploads
can be temporarily spooled by the framework. Deployment logs, temporary
storage, access control, and retention are the operator's responsibility.

## Input protections

- Only PNG and JPEG signatures are accepted by the library.
- A 10 MiB limit applies to library input bytes. HTTP requests have a
  separate 10 MiB total-body limit, including multipart overhead.
- Starlette's body-limit middleware counts bytes actually received, even
  when Content-Length is absent or understates the request.
- The endpoint also caps its own read buffer.
- Header dimensions are checked before native decoding: at most 50 million
  pixels and 20,000 pixels on either edge, with each edge at least 8 pixels.
- Dimensions are checked again after decoding. Normalized processing images
  are limited to 16 million pixels and must cover the analysis grid.
- Empty files, truncated headers, unsupported formats, and native decode
  errors are rejected as input errors rather than exposing stack traces.

These are application-level guards, not complete isolation of a native image
decoder. The preliminary header parser is not a full format validator; a
malicious file could exploit parser differences or decoder vulnerabilities
before a post-decode check. The pixel limit is not a promise of a particular
peak-memory ceiling, and concurrent requests multiply resource use.

## Before exposing an HTTP service

Use authentication, rate and concurrency limits, a reverse-proxy body limit,
timeouts, and worker memory/CPU limits. For adversarial uploads, run decoding
in an isolated worker with enforced resource limits. Keep native image and
HTTP parsing dependencies updated and rerun tests after upgrades.

Do not commit original research images, credential files, logs, local
environments, or generated audit reports. Review the actual staged file list
and diff before publishing; ignore patterns alone are not a security boundary.

The two micrographs in the README's real-micrograph figure are an exception
approved by the repository owner. They are downscaled, re-encoded copies
that carry no source filename or metadata.

To review dependencies and Python source independently:

~~~bash
python -m pip install pip-audit bandit
python -m pip_audit -r requirements.txt
python -m bandit -r dcv_vision api.py
~~~

A scanner's clean result covers its rules and known advisories at the time
of the scan; it is not a guarantee that all vulnerabilities are absent.
