"""Region features and the per-region "levels" model.

The core finding from studying vanilla MERS maps: inside one material
region every channel is (almost exactly) a linear function of the color
texture's grayscale value, normalized to that region's own min..max range:

    channel(px) = dark + t(px) * (bright - dark)
    t(px)       = (gray(px) - gray_min) / (gray_max - gray_min)

So a region is fully described by eight numbers (dark/bright for M, E, R, S)
plus an optional per-pixel roughness noise amount. Roughness almost always
has dark > bright (brighter pixels are smoother).
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.cluster.vq import kmeans2

from .imaging import rgb_to_lab

CHANNELS = ("M", "E", "R", "S")
PARAM_NAMES = ("M_dark", "M_bright", "E_dark", "E_bright",
               "R_dark", "R_bright", "S_dark", "S_bright", "R_noise")
N_PARAMS = len(PARAM_NAMES)

FEATURE_NAMES = ("lightness", "a", "b", "chroma", "lightness_std", "gray_range",
                 "rel_brightness", "area", "rel_chroma", "delta_e_rest")
# Divisors that bring each feature to a comparable scale for k-NN distance.
FEATURE_SCALE = np.array([22.0, 14.0, 14.0, 12.0, 10.0, 0.25, 0.12, 0.35, 14.0, 15.0], dtype=np.float32)
MIN_RANGE = 3.0  # gray range below which a region is treated as flat


class TextureContext:
    """Per-texture values reused when computing region features."""

    def __init__(self, rgba: np.ndarray, gray: np.ndarray, opaque: np.ndarray):
        self.rgba = rgba
        self.gray = gray
        self.opaque = opaque
        self.lab = rgb_to_lab(rgba[..., :3])
        self.chroma = np.hypot(self.lab[..., 1], self.lab[..., 2])
        self.gray_mean = float(gray[opaque].mean()) if opaque.any() else 0.0
        self.chroma_mean = float(self.chroma[opaque].mean()) if opaque.any() else 0.0
        self.n_opaque = max(int(opaque.sum()), 1)


def region_features(ctx: TextureContext, mask: np.ndarray) -> np.ndarray:
    lab = ctx.lab[mask]
    g = ctx.gray[mask]
    ch = ctx.chroma[mask]
    rest = ctx.opaque & ~mask
    # How far this region's color is from everything else in the texture.
    delta_e = float(np.linalg.norm(lab.mean(0) - ctx.lab[rest].mean(0))) if rest.any() else 0.0
    return np.array([
        lab[:, 0].mean(), lab[:, 1].mean(), lab[:, 2].mean(), ch.mean(), lab[:, 0].std(),
        (g.max() - g.min()) / 255.0,
        (g.mean() - ctx.gray_mean) / 255.0,
        mask.sum() / ctx.n_opaque,
        ch.mean() - ctx.chroma_mean,
        delta_e,
    ], dtype=np.float32)


def texture_features(ctx: TextureContext) -> np.ndarray:
    """Whole-texture descriptor used by the color-based material classifier."""
    feats = region_features(ctx, ctx.opaque)
    coverage = ctx.opaque.mean()
    n_colors = len(np.unique(ctx.rgba[ctx.opaque][:, :3], axis=0))
    return np.concatenate([feats[:6], [coverage, min(n_colors, 256) / 256.0]]).astype(np.float32)


TEXTURE_FEATURE_SCALE = np.array([22.0, 14.0, 14.0, 12.0, 10.0, 0.25, 0.3, 0.3], dtype=np.float32)


def normalized_t(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-pixel position of each pixel within its region's gray range (0 = darkest)."""
    g = gray[mask]
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < MIN_RANGE:
        return np.full(g.shape, 0.5, dtype=np.float32)
    return ((g - lo) / (hi - lo)).astype(np.float32)


def fit_region(gray: np.ndarray, mers: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the levels model to a vanilla region. Returns ``(params, mean_abs_error)``."""
    t = normalized_t(gray, mask)
    vals = mers[mask].astype(np.float32)
    params = np.zeros(N_PARAMS, dtype=np.float32)
    err = 0.0
    flat = t.std() < 1e-6
    for i in range(4):
        y = vals[:, i]
        if flat:
            d = b = float(y.mean())
        else:
            slope, icpt = np.polyfit(t, y, 1)
            d, b = float(icpt), float(icpt + slope)
        params[2 * i], params[2 * i + 1] = np.clip(d, 0, 255), np.clip(b, 0, 255)
        err += float(np.abs(y - (d + t * (b - d))).mean())
    params[8] = roughness_noise(gray[mask], vals[:, 2])
    return params, err / 4.0


def roughness_noise(g: np.ndarray, r: np.ndarray) -> float:
    """Std-dev of roughness among pixels that share the same gray value.

    Pure levels maps give 0 here; hand-noised maps such as dirt give 5-10.
    """
    keys = np.round(g).astype(np.int32)
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if counts.max() < 2:
        return 0.0
    means = np.bincount(inv, r) / counts
    resid = r - means[inv]
    multi = counts[inv] > 1
    return float(resid[multi].std()) if multi.any() else 0.0


def predict_channels(gray: np.ndarray, mask: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate the levels model for one region -> ``(n, 4)`` float values."""
    t = normalized_t(gray, mask)[:, None]
    dark = params[[0, 2, 4, 6]][None, :]
    bright = params[[1, 3, 5, 7]][None, :]
    return dark + t * (bright - dark)


def color_clusters(ctx: TextureContext, k: int, seed: int = 0) -> np.ndarray:
    """K-means over L*a*b* (hue weighted above lightness). Returns a label map (-1 = transparent)."""
    labels = np.full(ctx.gray.shape, -1, dtype=np.int32)
    pts = ctx.lab[ctx.opaque].astype(np.float64)
    if len(pts) == 0:
        return labels
    uniq = np.unique(pts, axis=0)
    k = int(min(k, len(uniq)))
    if k <= 1:
        labels[ctx.opaque] = 0
        return labels
    w = np.array([0.6, 1.0, 1.0])
    rng = np.random.default_rng(seed)
    sample = pts if len(pts) <= 20000 else pts[rng.choice(len(pts), 20000, replace=False)]
    centroids, _ = kmeans2(sample * w, k, minit="++", seed=seed)
    d = ((pts * w)[:, None, :] - centroids[None, :, :]) ** 2
    lab = d.sum(-1).argmin(1)
    # Re-index to drop empty clusters.
    _, lab = np.unique(lab, return_inverse=True)
    labels[ctx.opaque] = lab
    return labels


def param_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Scaled difference between two parameter vectors (< 1 means "same material look")."""
    mid = lambda p, i: (p[i] + p[i + 1]) / 2  # noqa: E731
    return max(abs(mid(a, 0) - mid(b, 0)) / 40.0,
               abs(mid(a, 2) - mid(b, 2)) / 10.0,
               abs(a[4] - b[4]) / 20.0,
               abs(a[5] - b[5]) / 20.0,
               abs(mid(a, 6) - mid(b, 6)) / 40.0)


def name_seed(name: str) -> int:
    """Stable per-texture seed so regeneration is deterministic."""
    return int.from_bytes(hashlib.md5(name.encode()).digest()[:4], "little")
