"""Manual adjustments to a generated MERS map, per channel and per detected color.

For each channel (M, E, R, S) the user can:

* set an **overall** value: the channel's learned values are rescaled so their
  peak equals it, keeping the learned variation. If the channel was empty
  (e.g. nothing emitted) it is filled with that value instead;
* set a **color** value: every pixel of that detected color (palette swatch)
  gets exactly that value. Colors set this way are not touched by the
  overall value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imaging import fill_transparent

CHANNELS = ("M", "E", "R", "S")
CHANNEL_NAMES = {"M": "Metalness", "E": "Emissive", "R": "Roughness", "S": "Subsurface"}


@dataclass
class ChannelEdit:
    overall: int | None = None                     # None = learned
    colors: dict[int, int] = field(default_factory=dict)   # swatch index -> value

    @property
    def active(self) -> bool:
        return self.overall is not None or bool(self.colors)


@dataclass
class TextureEdits:
    channels: dict[str, ChannelEdit] = field(default_factory=lambda: {c: ChannelEdit() for c in CHANNELS})

    @property
    def active(self) -> bool:
        return any(c.active for c in self.channels.values())

    def describe(self) -> list[str]:
        out = []
        for c in CHANNELS:
            e = self.channels[c]
            parts = []
            if e.overall is not None:
                parts.append(f"overall {e.overall}")
            if e.colors:
                parts.append(f"{len(e.colors)} color(s) set")
            if parts:
                out.append(f"{CHANNEL_NAMES[c]}: " + ", ".join(parts))
        return out


def apply_edits(mers: np.ndarray, opaque: np.ndarray, labels: np.ndarray,
                edits: TextureEdits) -> np.ndarray:
    """Return a copy of ``mers`` with ``edits`` applied (transparent pixels re-bled)."""
    out = mers.astype(np.float32).copy()
    for ci, c in enumerate(CHANNELS):
        e = edits.channels.get(c)
        if e is None or not e.active:
            continue
        ch = out[..., ci]
        manual = np.isin(labels, list(e.colors)) & opaque if e.colors else np.zeros_like(opaque)
        rest = opaque & ~manual
        if e.overall is not None and rest.any():
            peak = ch[rest].max()
            if peak >= 1.0:
                ch[rest] = ch[rest] * (e.overall / peak)
            else:
                ch[rest] = e.overall
        for swatch, value in e.colors.items():
            ch[(labels == swatch) & opaque] = value
    out = fill_transparent(out, opaque)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
