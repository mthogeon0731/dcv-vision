"""Processing parameters for D_CV vision analysis.

Changing these values shifts the definition/scale of D_CV relative to any
D_CV computed with the old values — the two are no longer comparable.
"""
from __future__ import annotations

# Hard cap on image size (width * height). Enforced twice: from the file
# header alone before cv2 ever touches the bytes (dcv._peek_image_size),
# and again on the decoded array's actual shape as a fallback for formats
# the header parser doesn't recognize. 50 megapixels is generous for any
# real micrograph capture.
#
# OpenCV has its own OPENCV_IO_MAX_IMAGE_PIXELS env var for this, but it's
# not used here: it's latched at cv2's native-extension load time, not at
# first decode, so setting it from Python after `import cv2` has already
# run anywhere in the process (including inside this package) is a no-op —
# confirmed empirically, not just from the docs. Depending on it would also
# make this library's safety contingent on import order in whatever
# process embeds it, which isn't something a library should ask its caller
# to get right.
MAX_IMAGE_PIXELS = 50_000_000

# Long edge is locked to this value the moment an image enters the pipeline
# (upscale or downscale). If input resolution varied run to run, every
# pixel-unit kernel constant below would be meaningless — so scale is fixed
# before any kernel is applied.
RESIZE_LONG_EDGE_PX = 1024

# Quadrat grid size (N x N) for the D_CV calculation. Project-fixed definition.
GRID_N = 8

# Gaussian blur kernel for grayscale noise suppression (odd, square).
GAUSSIAN_KERNEL = (5, 5)

# ---------------------------------------------------------------------------
# The three constants below (TOPHAT_KERNEL_PX / MORPH_OPEN_KERNEL_PX /
# MIN_PARTICLE_AREA_PX) are PROVISIONAL — pending approval against a real
# micrograph. Do not adjust them until someone has visually confirmed the
# binarization quality on an actual photo.
#
# Capture SOP (fixed): 10X objective, 960 x 720um field of view, 2048x1536
# raw sensor export (not a screen capture). Under this SOP, px/um = 2048 /
# 960 = 2.133. A 20um target particle (alumina) is therefore 20 * 2.133 =
# 42.7px at native resolution, and 42.7 * 0.5 = 21.3px after the
# RESIZE_LONG_EDGE_PX=1024 resize (scale = 1024/2048 = 0.5).
# ---------------------------------------------------------------------------

# Structuring element for top-hat (bright particles) / black-hat (dark
# particles) background correction, which doubles as the polarity test.
# Provisional. About 2.4x (51/21.3) the SOP particle size — large enough to
# fully enclose a particle while staying smaller than typical illumination
# gradients, which is theoretically reasonable but not yet confirmed against
# a real photo.
TOPHAT_KERNEL_PX = 51

# Morphological opening kernel used to remove salt-and-pepper noise after
# thresholding. Provisional, pending real-photo confirmation.
MORPH_OPEN_KERNEL_PX = 3

# Connected components smaller than this pixel area are treated as noise and
# discarded. Provisional. The SOP particle (diameter 21.3px, radius
# 10.65px) has area pi*10.65^2 ~= 356px^2 — this value (20px^2) is set
# conservatively low, well below that, so it only filters out noise smaller
# than any real particle. Final value pending real-photo confirmation.
MIN_PARTICLE_AREA_PX = 20

# Max long-edge pixels for the binarization thumbnail returned in the
# response (keeps the payload small).
THUMBNAIL_MAX_PX = 512
