"""The learned knowledge base: vanilla exemplars plus per-material statistics.

Two exemplar sets are stored:

* **regions**: true material regions of vanilla textures (found from their
  real MERS maps) with their fitted levels parameters. Used to predict the
  parameters of a region.
* **clusters**: k-means color clusters of vanilla textures (computed exactly
  the way the generator clusters a new texture), labeled with whether that
  cluster is a separate accent material in vanilla and, if so, its
  parameters. Used to decide which clusters of a new texture are accents.

Produced by :mod:`autovv.trainer`, consumed by :mod:`autovv.generator`.
Everything is plain JSON so it can be inspected and version controlled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .materials import tokenize
from .regions import FEATURE_SCALE, N_PARAMS, TEXTURE_FEATURE_SCALE

# Tokens that say where a face is, not what it is made of.
_GENERIC_TOKENS = {"block", "side", "top", "bottom", "front", "back", "end", "inner", "outer",
                   "carried", "stage", "upper", "lower", "left", "right", "base", "north",
                   "south", "east", "west", "horizontal", "vertical", "opaque", "mipmap", "a",
                   "b", "c", "d", "e", "f", "g", "h", "blocks", "textures", "texture"}
NAME_WEIGHT = 0.8  # distance shrink factor for a perfect name match


def name_tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in tokenize(name.replace("/", "_")) if t not in _GENERIC_TOKENS)


def _name_similarity(query: frozenset[str], others: list[frozenset[str]]) -> np.ndarray:
    if not query:
        return np.zeros(len(others), dtype=np.float32)
    return np.array([len(query & o) / len(query | o) if o else 0.0 for o in others], dtype=np.float32)

DEFAULT_KB_PATH = Path(__file__).parent / "data" / "knowledge_base.json"
# Materials that add metalness or glassiness. When no preset applies, the generator learns
# from every other vanilla texture but never from these: guessing them from color alone
# looks far worse than missing them.
NOT_FROM_COLOR = {"metal", "gem", "glass", "ice", "ore", "unknown"}


@dataclass
class Prediction:
    params: np.ndarray            # N_PARAMS values
    accent_prob: float            # weighted share of neighbors that are accents
    neighbors: list[str]         # vanilla textures the prediction was drawn from
    distance: float               # mean scaled feature distance to those neighbors


@dataclass
class ExemplarSet:
    features: np.ndarray          # (n, n_features)
    params: np.ndarray            # (n, N_PARAMS)
    category: np.ndarray          # (n,) material key
    is_accent: np.ndarray         # (n,) bool
    texture: list[str]            # (n,) source texture

    def __post_init__(self):
        self.tokens = [name_tokens(t) for t in self.texture]

    def to_json(self) -> dict:
        r = lambda a: np.round(a.astype(float), 3).tolist()  # noqa: E731
        return {"features": r(self.features), "params": r(self.params),
                "category": self.category.tolist(), "is_accent": self.is_accent.astype(int).tolist(),
                "texture": self.texture}

    @classmethod
    def from_json(cls, d: dict) -> "ExemplarSet":
        return cls(np.array(d["features"], dtype=np.float32),
                   np.array(d["params"], dtype=np.float32).reshape(-1, N_PARAMS),
                   np.array(d["category"]), np.array(d["is_accent"], dtype=bool), list(d["texture"]))

    def subset(self, keep: np.ndarray) -> "ExemplarSet":
        return ExemplarSet(self.features[keep], self.params[keep], self.category[keep],
                           self.is_accent[keep], [t for t, k in zip(self.texture, keep) if k])

    def neighbors(self, feats: np.ndarray, categories: list[str] | None, role: str | None,
                   k: int, name: str = "", metallic_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        sel = np.ones(len(self.features), dtype=bool)
        if categories:
            cat_sel = np.isin(self.category, categories)
            if cat_sel.sum() >= 3:
                sel &= cat_sel
        if role is not None:
            role_sel = self.is_accent if role == "accent" else ~self.is_accent
            if (sel & role_sel).sum() >= 3:
                sel &= role_sel
        if metallic_only:
            # Some "metal"-named textures are mostly non-metal (rail sleepers, cauldron water);
            # when the user picks Metal, learn only from surfaces that really are metallic.
            metallic = self.params[:, :2].max(1) >= 20
            if (sel & metallic).sum() >= 3:
                sel &= metallic
        idx = np.flatnonzero(sel)
        d = np.sqrt((((self.features[idx] - feats) / FEATURE_SCALE) ** 2).sum(1))
        if name:
            # Textures from the same family (e.g. red_candle / blue_candle, or a pack
            # retexture sharing a vanilla name) are pulled closer.
            sim = _name_similarity(name_tokens(name), [self.tokens[i] for i in idx])
            d = d * (1.0 - NAME_WEIGHT * sim)
        order = np.argsort(d)[:k]
        return idx[order], d[order]


@dataclass
class KnowledgeBase:
    regions: ExemplarSet
    clusters: ExemplarSet
    tex_features: np.ndarray                  # (m, n_tex_features)
    tex_category: np.ndarray                  # (m,)
    categories: dict = field(default_factory=dict)   # per-material stats/edge effects
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path = DEFAULT_KB_PATH) -> "KnowledgeBase":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tx = d["texture_classifier"]
        return cls(ExemplarSet.from_json(d["regions"]), ExemplarSet.from_json(d["clusters"]),
                   np.array(tx["features"], dtype=np.float32), np.array(tx["category"]),
                   d.get("categories", {}), d.get("meta", {}))

    def save(self, path: str | Path = DEFAULT_KB_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "meta": self.meta,
            "categories": self.categories,
            "regions": self.regions.to_json(),
            "clusters": self.clusters.to_json(),
            "texture_classifier": {
                "features": np.round(self.tex_features.astype(float), 3).tolist(),
                "category": self.tex_category.tolist(),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, separators=(",", ":"))

    def subset(self, keep_regions: np.ndarray, keep_clusters: np.ndarray,
               keep_tex: np.ndarray) -> "KnowledgeBase":
        """A KB restricted to some exemplars (used for hold-out evaluation)."""
        return KnowledgeBase(self.regions.subset(keep_regions), self.clusters.subset(keep_clusters),
                             self.tex_features[keep_tex], self.tex_category[keep_tex],
                             self.categories, self.meta)

    # ------------------------------------------------------------- queries
    @property
    def known_categories(self) -> set[str]:
        return set(self.regions.category.tolist())

    def predict_region(self, feats: np.ndarray, categories: list[str] | None,
                       role: str | None = None, k: int = 5, name: str = "",
                       metallic_only: bool = False) -> Prediction:
        """Weighted-median parameters of the k nearest vanilla regions."""
        nn, dist = self.regions.neighbors(feats, categories, role, k, name, metallic_only)
        w = 1.0 / (dist + 0.15)
        params = _weighted_median_rows(self.regions.params[nn], w)
        accent = float((w * self.regions.is_accent[nn]).sum() / w.sum())
        return Prediction(params, accent, [self.regions.texture[i] for i in nn], float(dist.mean()))

    def predict_cluster(self, feats: np.ndarray, categories: list[str] | None,
                        k: int = 5, name: str = "", metallic_only: bool = False) -> Prediction:
        """Is a color cluster a separate accent material, and if so with what parameters?"""
        nn, dist = self.clusters.neighbors(feats, categories, None, k, name, metallic_only)
        w = 1.0 / (dist + 0.15)
        acc = self.clusters.is_accent[nn]
        prob = float((w * acc).sum() / w.sum())
        src = nn[acc] if acc.any() else nn
        ws = w[acc] if acc.any() else w
        params = _weighted_median_rows(self.clusters.params[src], ws)
        return Prediction(params, prob, [self.clusters.texture[i] for i in nn], float(dist.mean()))

    def edge_effects(self, category: str) -> dict:
        return self.categories.get(category, {}).get("edge_effects", {})


def _weighted_median_rows(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-channel weighted median that keeps each (dark, bright) pair from one exemplar.

    Taking the two ends from different exemplars would turn e.g. a flat metalness of 191
    into an invented 128->191 gradient; vanilla's structure is kept by choosing the
    exemplar whose midpoint is the weighted median midpoint.
    """
    out = np.empty(rows.shape[1], dtype=np.float32)
    for j in range(0, rows.shape[1] - 1, 2):
        mid = (rows[:, j] + rows[:, j + 1]) / 2
        target = _weighted_median(mid, weights)
        pick = np.flatnonzero(mid == target)
        best = pick[np.argmax(weights[pick])]
        out[j], out[j + 1] = rows[best, j], rows[best, j + 1]
    if rows.shape[1] % 2:
        out[-1] = _weighted_median(rows[:, -1], weights)
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])
