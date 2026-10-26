"""Image loading/saving and color-space helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

IMAGE_EXTS = (".tga", ".png", ".jpg", ".jpeg")  # Minecraft's lookup priority order


def load_rgba(path: str | Path) -> np.ndarray:
    """Load any supported image as an ``(H, W, 4)`` uint8 RGBA array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"), dtype=np.uint8)


def find_image(base: str | Path) -> Path | None:
    """Resolve an extensionless texture reference the way Minecraft does (.tga > .png > .jpg)."""
    base = Path(base)
    for ext in IMAGE_EXTS:
        p = base.with_name(base.name + ext)
        if p.exists():
            return p
    return None


def save_map(path: str | Path, arr: np.ndarray, channels: int = 4) -> None:
    """Save a MER (3 channel) or MERS (4 channel) map. Format follows the extension."""
    path = Path(path)
    mode = "RGBA" if channels == 4 else "RGB"
    im = Image.fromarray(np.ascontiguousarray(arr[..., :channels]).astype(np.uint8), mode)
    if path.suffix.lower() == ".tga":
        im.save(path, compression=None)
    else:
        im.save(path)


def gray(rgba: np.ndarray) -> np.ndarray:
    """Vanilla-matching grayscale: the plain mean of R, G and B.

    Of the luminance formulas tested against vanilla MERS maps (Rec.601,
    Rec.709, linear-light, max-channel, mean) the mean correlated best.
    """
    return rgba[..., :3].astype(np.float32).mean(axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, ``(..., 3)``) to CIE L*a*b* (D65)."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = c @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def fill_transparent(arr: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Copy each transparent pixel's value from its nearest opaque pixel.

    Keeps mip-maps and texture filtering from bleeding black into edges.
    """
    if opaque.all() or not opaque.any():
        return arr
    _, (iy, ix) = ndimage.distance_transform_edt(~opaque, return_indices=True)
    return arr[iy, ix]


def border_mask(shape: tuple[int, int], tile: int | None = None) -> np.ndarray:
    """Pixels on the outer ring of each tile (frames of a flipbook count as tiles)."""
    h, w = shape
    tile = tile or min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    ty, tx = yy % tile, xx % tile
    return (ty == 0) | (tx == 0) | (ty == tile - 1) | (tx == tile - 1)


def local_relief(g: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """How much brighter (+) or darker (-) a pixel is than its 3x3 neighborhood."""
    gf = np.where(opaque, g, np.nan)
    pad = np.pad(gf, 1, mode="edge")
    stack = np.stack([pad[dy:dy + g.shape[0], dx:dx + g.shape[1]]
                      for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)])
    with np.errstate(all="ignore"):
        mean = np.nanmean(stack, axis=0)
    out = np.where(np.isnan(mean), 0.0, g - mean)
    return np.where(opaque, out, 0.0).astype(np.float32)
