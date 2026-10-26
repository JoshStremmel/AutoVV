"""Generate vanilla-style MERS maps from a color texture using the learned knowledge base.

Pipeline:
1. Resolve the material (user choice > name keywords > color classifier).
2. Split the texture into color clusters (k-means in L*a*b*).
3. Ask the knowledge base which clusters look like vanilla *accent* regions
   (ore specks, metal fittings) for that material; the rest
   form the base region.
4. For every region, predict dark/bright values for M, E, R and S from the
   nearest vanilla regions and apply the levels model per pixel.
5. Apply learned border/crevice roughness offsets and roughness noise,
   then bleed values into transparent pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imaging import border_mask, fill_transparent, gray, local_relief
from .knowledge import NOT_FROM_COLOR, KnowledgeBase
from .materials import BY_KEY, classify_name
from .regions import (TextureContext, color_clusters, name_seed, param_distance,
                      predict_channels, region_features)

MAX_CLUSTERS = 6
ACCENT_THRESHOLD = 0.5
MIN_ACCENT_PIXELS = 2


@dataclass
class Region:
    mask: np.ndarray
    params: np.ndarray
    role: str                       # "base" or "accent"
    neighbors: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    mers: np.ndarray                # (H, W, 4) uint8
    materials: list[str]
    material_reason: str
    regions: list[Region]
    notes: list[str] = field(default_factory=list)

    @property
    def mer(self) -> np.ndarray:
        return self.mers[..., :3]

    def summary(self) -> str:
        names = ", ".join(BY_KEY[m].label if m in BY_KEY else m for m in self.materials) or "Auto"
        lines = [f"Material: {names} ({self.material_reason})"]
        for i, r in enumerate(self.regions):
            p = r.params
            lines.append(
                f"  region {i + 1} [{r.role}, {int(r.mask.sum())} px]: "
                f"M {p[0]:.0f}-{p[1]:.0f}  E {p[2]:.0f}-{p[3]:.0f}  "
                f"R {p[4]:.0f}-{p[5]:.0f}  S {p[6]:.0f}-{p[7]:.0f}"
                + (f"  noise {p[8]:.1f}" if p[8] >= 2.5 else ""))
        lines += [f"  {n}" for n in self.notes]
        return "\n".join(lines)


def resolve_materials(kb: KnowledgeBase, ctx: TextureContext, name: str,
                      materials: list[str] | None, folder_hint: str = "") -> tuple[list[str], str]:
    if materials:
        return list(materials), "selected"
    key, why = classify_name(name, folder_hint)
    if key and key in kb.known_categories:
        return [key], f"name: {why}"
    # No preset: learn from every vanilla texture (nearest by color and name) instead of
    # forcing a color-guessed preset, which fits textures like slime or melon badly.
    return [], "auto: nearest vanilla textures"


def generate(kb: KnowledgeBase, rgba: np.ndarray, name: str = "texture",
             materials: list[str] | None = None, folder_hint: str = "",
             noise: bool = True, edge_effects: bool = True,
             name_matching: bool = True, chosen: bool = False) -> GenerationResult:
    """Generate a MERS map. ``chosen`` means the user picked ``materials`` themselves
    (so an explicit Metal really is metal), rather than it being guessed from the name."""
    opaque = rgba[..., 3] > 0
    g = gray(rgba)
    ctx = TextureContext(rgba, g, opaque)
    out = np.zeros(rgba.shape[:2] + (4,), dtype=np.float32)
    if not opaque.any():
        # Fully transparent texture: nothing to shade, emit an all-zero map.
        return GenerationResult(out.astype(np.uint8), list(materials or []), "fully transparent", [])
    cats, reason = resolve_materials(kb, ctx, name, materials, folder_hint)

    # With no preset, search all of vanilla except materials that add metalness or glassiness:
    # guessing those from color alone looks far worse than missing them.
    search = cats or sorted(kb.known_categories - (NOT_FROM_COLOR - {"unknown"}))
    regions = _segment(kb, ctx, search, name, name if name_matching else "",
                       metallic_only=chosen and cats == ["metal"])

    rng = np.random.default_rng(name_seed(name))
    for reg in regions:
        vals = predict_channels(g, reg.mask, reg.params)
        if noise and reg.params[8] >= 2.5:
            vals[:, 2] += rng.normal(0.0, reg.params[8], len(vals))
        out[reg.mask] = vals

    if edge_effects:
        if cats:
            _apply_edge_effects(kb, cats[0], out, g, opaque)

    out = fill_transparent(out, opaque)
    mers = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return GenerationResult(mers, cats, reason, regions)


def _segment(kb: KnowledgeBase, ctx: TextureContext, cats: list[str], seed_name: str,
             name: str, metallic_only: bool = False) -> list[Region]:
    whole = ctx.opaque
    labels = color_clusters(ctx, MAX_CLUSTERS, seed=name_seed(seed_name))
    n = labels.max() + 1

    base_pred = kb.predict_region(region_features(ctx, whole), cats, role="base", name=name,
                                  metallic_only=metallic_only)

    # Clusters that behave like a separate accent material in comparable vanilla textures.
    accents: list[Region] = []
    for c in range(n):
        m = labels == c
        if m.sum() < MIN_ACCENT_PIXELS or m.sum() == whole.sum():
            continue
        pred = kb.predict_cluster(region_features(ctx, m), cats, name=name, metallic_only=metallic_only)
        if pred.accent_prob >= ACCENT_THRESHOLD and param_distance(pred.params, base_pred.params) >= 1.0:
            accents.append(Region(m, pred.params, "accent", pred.neighbors))

    base_mask = whole.copy()
    for a in accents:
        base_mask &= ~a.mask
    if not base_mask.any():
        return [Region(whole, base_pred.params, "base", base_pred.neighbors)]

    base = (kb.predict_region(region_features(ctx, base_mask), cats, role="base", name=name,
                              metallic_only=metallic_only)
            if accents else base_pred)
    regions = [Region(base_mask, base.params, "base", base.neighbors)]

    # Accents with the same look share one levels range, as in vanilla.
    for a in accents:
        if param_distance(a.params, base.params) < 1.0:
            regions[0].mask |= a.mask
            continue
        for r in regions[1:]:
            if param_distance(a.params, r.params) < 1.0:
                r.mask = r.mask | a.mask
                break
        else:
            regions.append(a)
    return regions


def _apply_edge_effects(kb: KnowledgeBase, category: str, out: np.ndarray,
                        g: np.ndarray, opaque: np.ndarray) -> None:
    fx = kb.edge_effects(category)
    if not fx:
        return
    h, w = g.shape
    tile = w if h % w == 0 else min(h, w)  # vertical flipbooks are stacks of square frames
    rough = out[..., 2]
    if abs(fx.get("border_offset", 0.0)) >= 2.0:
        rough += np.where(border_mask((h, w), tile) & opaque, fx["border_offset"], 0.0)
    if abs(fx.get("relief_coef", 0.0)) >= 0.05:
        rough += fx["relief_coef"] * local_relief(g, opaque)
