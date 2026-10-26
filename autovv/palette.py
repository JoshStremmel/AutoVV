"""Color swatches detected from a texture, used to edit channels and assign materials per color."""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

from .imaging import gray
from .regions import TextureContext, color_clusters, name_seed

MAX_EXACT = 256        # up to this many distinct colors, every exact color is its own swatch
GROUPED_SWATCHES = 64  # beyond that (photo-like textures), similar colors are grouped


@dataclass
class Swatch:
    index: int
    rgb: tuple[int, int, int]
    share: float                  # fraction of opaque pixels
    grouped: bool = False         # True when it stands for several similar colors

    @property
    def hex(self) -> str:
        return "#%02x%02x%02x" % self.rgb


def detect_palette(rgba: np.ndarray, name: str = "") -> tuple[np.ndarray, list[Swatch]]:
    """Every color of a texture as swatches. Returns ``(labels, swatches)``.

    ``labels`` maps each pixel to a swatch index (-1 = transparent). Pixel-art textures
    get one swatch per exact color; textures with more than ``MAX_EXACT`` colors get
    ``GROUPED_SWATCHES`` groups of similar colors. Swatches are sorted most common first,
    and the result is deterministic, so an edit made in the GUI maps to the same pixels
    when a batch runs later.
    """
    opaque = rgba[..., 3] > 0
    labels = np.full(opaque.shape, -1, dtype=np.int32)
    if not opaque.any():
        return labels, []
    colors, inverse, counts = np.unique(rgba[opaque][:, :3], axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    grouped = len(colors) > MAX_EXACT
    if grouped:
        ctx = TextureContext(rgba, gray(rgba), opaque)
        raw = color_clusters(ctx, GROUPED_SWATCHES, seed=name_seed("palette:" + name))[opaque]
        n = raw.max() + 1
        counts = np.bincount(raw, minlength=n)
        colors = np.array([rgba[opaque][raw == c][:, :3].mean(0) if counts[c] else (0, 0, 0)
                            for c in range(n)])
        inverse = raw
    # Most common first; ties broken by brightness, then color, so the order is stable.
    order = sorted(range(len(colors)),
                   key=lambda i: (-counts[i], -float(np.mean(colors[i])), tuple(colors[i])))
    rank = np.empty(len(colors), dtype=np.int32)
    rank[order] = np.arange(len(colors))
    labels[opaque] = rank[inverse]
    total = int(opaque.sum())
    swatches = [Swatch(new, tuple(int(round(v)) for v in colors[old]), counts[old] / total, grouped)
                for new, old in enumerate(order) if counts[old]]
    return labels, swatches


def sorted_by_hue(swatches: list[Swatch]) -> list[Swatch]:
    """Display order: colors around the hue wheel (red → yellow → green → blue → purple),
    darker to lighter within a hue, then grays from dark to light at the end."""
    def key(sw: Swatch):
        h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in sw.rgb))
        if s < 0.12 or v < 0.08:                      # grays / near-black have no useful hue
            return (1, 0.0, v)
        return (0, round(h * 24) / 24, v)             # 24 hue bands, then brightness
    return sorted(swatches, key=key)
