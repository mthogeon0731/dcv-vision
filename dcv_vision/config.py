"""Fixed scientific parameters and resource limits for micrograph analysis.

Changing segmentation or measurement parameters changes the meaning of D_CV.
Results from different definition versions must not be pooled directly.
"""
from __future__ import annotations

# v2: user-selected polarity, global Otsu, minority-phase quadrat CV.
DCV_DEFINITION_VERSION = 2

# These input limits also apply to direct library calls.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_IMAGE_EDGE_PX = 20_000
# Magnification normalization can enlarge an image; check before allocating it.
MAX_PROCESSING_PIXELS = 16_000_000

# Used only when a supported scale bar is not detected.
RESIZE_LONG_EDGE_PX = 1024
GRID_N = 8
GAUSSIAN_KERNEL = (5, 5)

# Cleanup constants remain provisional; confirm masks for your optics.
# Reference capture: 10X, 960 x 720 um FOV, 2048 x 1536 raw export.
# A 20 um particle is about 21.3 px at the reference processing scale.
MORPH_OPEN_KERNEL_PX = 3
MIN_PARTICLE_AREA_PX = 20
THUMBNAIL_MAX_PX = 512

# Only a red bar in the lower-right corner is recognized. The numeric label
# is NOT read: its length is assumed to represent 100 um.
SCALE_BAR_UM = 100
TARGET_PX_PER_UM = 1024 / 960
SCALE_BAR_MIN_FRAC = 0.03
SCALE_BAR_MAX_FRAC = 0.4

# 99th minus 1st percentile of the blurred grayscale image.
MIN_CONTRAST = 40
