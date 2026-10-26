"""Learning session: study vanilla color/MERS pairs and build the knowledge base.

For every vanilla ``*.texture_set.json`` with an image MER/MERS layer:

* the texture is labeled with a material from its name,
* its MERS map is split into material regions (pixels sharing
  metal/emissive/subsurface usage, recursively split wherever one linear
  levels model does not fit, e.g. ore specks inside stone),
* each region's levels parameters are fitted and stored with its color
  features as an exemplar,
* roughness residuals on tile borders and in crevices are aggregated per
  material ("edge effects"),
* per-material statistics are summarized for the report.

A hold-out evaluation then regenerates unseen vanilla textures and measures
how close the output is to the real vanilla MERS.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.cluster.vq import kmeans2

from . import __version__
from .generator import MAX_CLUSTERS, generate
from .imaging import border_mask, find_image, gray, load_rgba, local_relief
from .knowledge import DEFAULT_KB_PATH, ExemplarSet, KnowledgeBase
from .materials import BY_KEY, classify_name
from .regions import (PARAM_NAMES, TextureContext, color_clusters, fit_region, name_seed,
                      param_distance, predict_channels, region_features, texture_features)
from .texture_set import MER_KEY, MERS_KEY, read_texture_set

DEFAULT_VANILLA = Path(r"C:\Users\joshs\OneDrive\Desktop\VanillaFiles\bedrock-samples-main"
                       r"\resource_pack\textures\blocks")
SPLIT_ERROR = 5.0       # mean abs error (0-255) above which a region is split further
MAX_DEPTH = 3
MIN_REGION = 3

Progress = Callable[[int, int, str], None]


def source_label(path) -> str:
    """How a training folder is shown: the built-in vanilla folder is just "Vanilla Files"."""
    try:
        if Path(path).resolve() == DEFAULT_VANILLA.resolve():
            return "Vanilla Files"
    except OSError:
        pass
    return str(path)


@dataclass
class Pair:
    name: str
    rel_folder: str
    color: Path
    mers: Path
    has_subsurface: bool


# ---------------------------------------------------------------- dataset
def find_pairs(roots: list[Path]) -> list[Pair]:
    pairs = []
    for root in roots:
        for js in sorted(root.rglob("*.texture_set.json")):
            ts = read_texture_set(js)
            if not ts:
                continue
            col = ts.get("color")
            ref = ts.get(MERS_KEY) or ts.get(MER_KEY)
            if not isinstance(col, str) or not isinstance(ref, str):
                continue
            cp, mp = find_image(js.parent / col), find_image(js.parent / ref)
            if cp and mp:
                rel = js.parent.relative_to(root).as_posix()
                pairs.append(Pair(col, rel if rel != "." else "", cp, mp, isinstance(ts.get(MERS_KEY), str)))
    return pairs


# ------------------------------------------------------------ segmentation
def segment_vanilla(ctx: TextureContext, mers: np.ndarray) -> list[np.ndarray]:
    """Split a vanilla texture into material regions using its real MERS map."""
    m = mers.astype(np.float32)
    key = (m[..., 0] > 0) * 1 + (m[..., 1] > 0) * 2 + (m[..., 3] > 0) * 4
    regions = []
    for k in np.unique(key[ctx.opaque]):
        regions.extend(_split(ctx, m, ctx.opaque & (key == k), 0))
    return regions


def _split(ctx: TextureContext, m: np.ndarray, mask: np.ndarray, depth: int) -> list[np.ndarray]:
    if mask.sum() < 2 * MIN_REGION or depth >= MAX_DEPTH:
        return [mask]
    _, err = fit_region(ctx.gray, m, mask)
    if err <= SPLIT_ERROR:
        return [mask]
    # Cluster on the real MERS values plus hue so that e.g. gold specks separate from stone.
    lab = ctx.lab[mask]
    feats = np.column_stack([m[mask] / 20.0, lab[:, 1:] / 8.0, lab[:, :1] / 25.0]).astype(np.float64)
    if len(np.unique(feats, axis=0)) < 2:
        return [mask]
    _, lab_ids = kmeans2(feats, 2, minit="++", seed=depth + 1)
    idx = np.flatnonzero(mask.ravel())
    parts = []
    for c in (0, 1):
        pm = np.zeros(mask.size, dtype=bool)
        pm[idx[lab_ids == c]] = True
        parts.append(pm.reshape(mask.shape))
    if min(p.sum() for p in parts) < MIN_REGION:
        return [mask]
    child_err = sum(fit_region(ctx.gray, m, p)[1] * p.sum() for p in parts) / mask.sum()
    if child_err > 0.6 * err:
        return [mask]
    return [r for p in parts for r in _split(ctx, m, p, depth + 1)]


# ---------------------------------------------------------------- training
@dataclass
class TrainingData:
    kb: KnowledgeBase
    pairs: list[Pair]
    exemplar_pair: np.ndarray       # pair index per region exemplar
    cluster_pair: np.ndarray        # pair index per cluster exemplar
    tex_pair: np.ndarray            # pair index per texture-classifier row
    fit_errors: list[float]
    skipped: list[str]


def learn(roots: list[Path], progress: Progress | None = None) -> TrainingData:
    pairs = find_pairs(roots)
    feats, params, cats, accent, texname, ex_pair = [], [], [], [], [], []
    cfeats, cparams, ccats, caccent, ctex, cl_pair = [], [], [], [], [], []
    tfeats, tcats, t_pair = [], [], []
    fit_errors: list[float] = []
    skipped: list[str] = []
    glowing = 0
    # edge statistics per category: border residuals, interior residuals, (relief, residual)
    border_res, inner_res = defaultdict(list), defaultdict(list)
    relief_x, relief_y = defaultdict(list), defaultdict(list)

    for i, p in enumerate(pairs):
        if progress:
            progress(i, len(pairs), p.name)
        cat, _ = classify_name(p.name, p.rel_folder)
        try:
            rgba, mers = load_rgba(p.color), load_rgba(p.mers)
        except OSError as e:
            skipped.append(f"{p.name}: {e}")
            continue
        if rgba.shape != mers.shape:
            skipped.append(f"{p.name}: color {rgba.shape[:2]} vs MERS {mers.shape[:2]}")
            continue
        if not p.has_subsurface:
            mers[..., 3] = 0
        # Emissive is never guessed: what glows is the user's choice (Emissive editor),
        # so vanilla's glow is left out and cannot leak into e.g. stone via furnace_front_on.
        glowing += int(mers[..., 1].max() > 0)
        mers[..., 1] = 0
        opaque = rgba[..., 3] > 0
        if opaque.sum() < MIN_REGION:
            continue
        cat = cat or "unknown"
        g = gray(rgba)
        ctx = TextureContext(rgba, g, opaque)
        regions = segment_vanilla(ctx, mers)
        biggest = max(range(len(regions)), key=lambda j: regions[j].sum())
        tname = f"{p.rel_folder}/{p.name}".lstrip("/")
        pred = np.zeros(rgba.shape[:2], dtype=np.float32)
        region_id = np.full(rgba.shape[:2], -1, dtype=np.int32)
        fitted = []
        for j, r in enumerate(regions):
            prm, err = fit_region(g, mers, r)
            fitted.append(prm)
            region_id[r] = j
            fit_errors.append(err)
            feats.append(region_features(ctx, r))
            params.append(prm)
            cats.append(cat)
            accent.append(j != biggest)
            texname.append(tname)
            ex_pair.append(i)
            pred[r] = predict_channels(g, r, prm)[:, 2]

        # Cluster exemplars: cluster the colors exactly like the generator will,
        # then record which clusters are really a distinct accent material.
        labels = color_clusters(ctx, MAX_CLUSTERS, seed=name_seed(p.name))
        for c in range(labels.max() + 1):
            cm = labels == c
            if cm.sum() == 0 or cm.sum() == opaque.sum():
                continue
            owner = np.bincount(region_id[cm], minlength=len(regions)).argmax()
            is_acc = owner != biggest and param_distance(fitted[owner], fitted[biggest]) >= 1.0
            cfeats.append(region_features(ctx, cm))
            cparams.append(fitted[owner])
            ccats.append(cat)
            caccent.append(is_acc)
            ctex.append(tname)
            cl_pair.append(i)
        tfeats.append(texture_features(ctx))
        tcats.append(cat)
        t_pair.append(i)

        # Edge effects: what vanilla does to roughness beyond the levels model.
        res = mers[..., 2].astype(np.float32) - pred
        h, w = g.shape
        tile = w if h % w == 0 else min(h, w)
        if tile >= 8:
            b = border_mask((h, w), tile) & opaque
            if b.any() and (opaque & ~b).any():
                border_res[cat].append(float(res[b].mean()))
                inner_res[cat].append(float(res[opaque & ~b].mean()))
        rel = local_relief(g, opaque)
        relief_x[cat].append(rel[opaque])
        relief_y[cat].append(res[opaque])

    if progress:
        progress(len(pairs), len(pairs), "summarizing")

    feats_a = np.array(feats, dtype=np.float32)
    params_a = np.array(params, dtype=np.float32)
    cats_a = np.array(cats)
    accent_a = np.array(accent, dtype=bool)

    categories = {}
    for c in sorted(set(cats)):
        sel = cats_a == c
        base = sel & ~accent_a
        acc = sel & accent_a
        entry = {
            "label": BY_KEY[c].label if c in BY_KEY else c,
            "textures": int(sum(1 for x in tcats if x == c)),
            "regions": int(sel.sum()),
            "accent_regions": int(acc.sum()),
            "base": _param_stats(params_a[base]),
            "accent": _param_stats(params_a[acc]) if acc.any() else None,
            "edge_effects": _edge_stats(border_res[c], inner_res[c], relief_x[c], relief_y[c]),
        }
        categories[c] = entry

    meta = {
        "autovv_version": __version__,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": [source_label(r) for r in roots],
        "pairs": len(pairs),
        "textures_used": len(tfeats),
        "regions": len(feats),
        "skipped": len(skipped),
        "emissive_textures_ignored": glowing,
        "median_fit_error": float(np.median(fit_errors)) if fit_errors else None,
        "param_names": list(PARAM_NAMES),
    }
    meta["clusters"] = len(cfeats)
    region_set = ExemplarSet(feats_a, params_a, cats_a, accent_a, texname)
    cluster_set = ExemplarSet(np.array(cfeats, dtype=np.float32), np.array(cparams, dtype=np.float32),
                              np.array(ccats), np.array(caccent, dtype=bool), ctex)
    kb = KnowledgeBase(region_set, cluster_set, np.array(tfeats, dtype=np.float32), np.array(tcats),
                       categories, meta)
    return TrainingData(kb, pairs, np.array(ex_pair), np.array(cl_pair), np.array(t_pair),
                        fit_errors, skipped)


def _param_stats(p: np.ndarray) -> dict:
    if len(p) == 0:
        return {}
    out = {}
    for j, n in enumerate(PARAM_NAMES):
        q = np.percentile(p[:, j], [10, 50, 90])
        out[n] = [round(float(x), 1) for x in q]
    out["metal_share"] = round(float((p[:, [0, 1]].max(1) > 20).mean()), 3)
    out["emissive_share"] = round(float((p[:, [2, 3]].max(1) > 2).mean()), 3)
    out["subsurface_share"] = round(float((p[:, [6, 7]].max(1) > 20).mean()), 3)
    out["noisy_share"] = round(float((p[:, 8] >= 2.5).mean()), 3)
    return out


def _edge_stats(border: list, inner: list, rx: list, ry: list) -> dict:
    out = {}
    if len(border) >= 5:
        diff = np.array(border) - np.array(inner)
        out["border_offset"] = round(float(np.median(diff)), 2)
        out["border_samples"] = len(border)
    if rx:
        x, y = np.concatenate(rx), np.concatenate(ry)
        keep = np.abs(y) < 60  # ignore pixels from mis-segmented regions
        if keep.sum() > 50 and x[keep].std() > 0:
            coef = float(np.polyfit(x[keep], y[keep], 1)[0])
            corr = float(np.corrcoef(x[keep], y[keep])[0, 1])
            # Only keep a relief effect that is actually consistent.
            out["relief_coef"] = round(coef, 3) if abs(corr) >= 0.15 else 0.0
            out["relief_corr"] = round(corr, 3)
    return out


# -------------------------------------------------------------- evaluation
def evaluate(data: TrainingData, folds: int = 5, limit: int | None = None,
             progress: Progress | None = None, name_matching: bool = True) -> dict:
    """K-fold hold-out: regenerate unseen vanilla textures and compare to the real MERS."""
    pairs = data.pairs
    usable = sorted(set(data.tex_pair.tolist()))
    if limit:
        rng = np.random.default_rng(0)
        usable = sorted(rng.choice(usable, min(limit, len(usable)), replace=False).tolist())
    fold_of = {i: (hash_name(pairs[i].name) % folds) for i in usable}
    errs = defaultdict(list)            # channel -> abs errors per texture
    per_cat = defaultdict(lambda: defaultdict(list))
    base_errs = defaultdict(list)
    done = 0
    for f in range(folds):
        test = [i for i in usable if fold_of[i] == f]
        test_set = set(test)
        sub = data.kb.subset(np.array([pi not in test_set for pi in data.exemplar_pair]),
                             np.array([pi not in test_set for pi in data.cluster_pair]),
                             np.array([pi not in test_set for pi in data.tex_pair]))
        for i in test:
            p = pairs[i]
            done += 1
            if progress:
                progress(done, len(usable), f"evaluating {p.name}")
            rgba, mers = load_rgba(p.color), load_rgba(p.mers)
            if rgba.shape != mers.shape:
                continue
            if not p.has_subsurface:
                mers[..., 3] = 0
            cat, _ = classify_name(p.name, p.rel_folder)
            res = generate(sub, rgba, p.name, [cat] if cat else None, p.rel_folder,
                           name_matching=name_matching)
            op = rgba[..., 3] > 0
            truth, out = effective(mers), effective(res.mers)
            # Baseline: flat per-material median, what a naive tool would do.
            stats = data.kb.categories.get(cat or "unknown", {}).get("base", {})
            for ci, ch in enumerate("MERS"):
                e = float(np.abs(out[..., ci][op] - truth[..., ci][op]).mean())
                errs[ch].append(e)
                per_cat[cat or "unknown"][ch].append(e)
                if stats:
                    flat = (stats[f"{ch}_dark"][1] + stats[f"{ch}_bright"][1]) / 2
                    base_errs[ch].append(float(np.abs(flat - truth[..., ci][op]).mean()))
            # Does the roughness preserve vanilla's per-pixel ordering?
            r_true = mers[..., 2][op].astype(float)
            r_pred = res.mers[..., 2][op].astype(float)
            if r_true.std() > 0 and r_pred.std() > 0:
                errs["R_corr"].append(float(np.corrcoef(r_true, r_pred)[0, 1]))
    summary = {
        "textures": len(usable),
        "folds": folds,
        "mae": {ch: round(float(np.mean(v)), 2) for ch, v in errs.items() if ch != "R_corr"},
        "median_ae": {ch: round(float(np.median(v)), 2) for ch, v in errs.items() if ch != "R_corr"},
        "baseline_mae": {ch: round(float(np.mean(v)), 2) for ch, v in base_errs.items()},
        "roughness_correlation": round(float(np.median(errs["R_corr"])), 3) if errs["R_corr"] else None,
        "per_category": {c: {ch: round(float(np.mean(v)), 1) for ch, v in d.items()}
                         for c, d in sorted(per_cat.items())},
    }
    return summary


def effective(mers: np.ndarray) -> np.ndarray:
    """Apply the Vibrant Visuals metal/subsurface rule: the larger wins, ties go to subsurface."""
    m = mers.astype(np.float32).copy()
    metal_wins = m[..., 0] > m[..., 3]
    m[..., 3] = np.where(metal_wins, 0, m[..., 3])
    m[..., 0] = np.where(metal_wins, m[..., 0], 0)
    return m


def hash_name(name: str) -> int:
    return name_seed(name)


# ------------------------------------------------------------------ report
def write_report(data: TrainingData, evaluation: dict | None, path: Path) -> None:
    kb = data.kb
    L = ["# AutoVV training report", "",
         f"Trained {kb.meta['trained_at']} from:", ""]
    L += [f"- {s}" for s in kb.meta["sources"]]
    L += ["", f"- Texture sets with an image MER/MERS layer: **{kb.meta['pairs']}**",
          f"- Textures analyzed: **{kb.meta['textures_used']}** (skipped {kb.meta['skipped']})",
          f"- Material regions learned: **{kb.meta['regions']}**",
          f"- Color clusters labeled (accent detection): **{kb.meta['clusters']}**",
          f"- Median levels-model fit error per region: **{kb.meta['median_fit_error']:.2f}** / 255", "",
          "## What vanilla does", "",
          "Inside one material region, each MERS channel is a linear remap of the color texture's",
          "grayscale (the mean of R, G, B) from the region's darkest to brightest pixel.",
          "Roughness runs from high on dark pixels to low on bright ones, so highlights come out",
          "smoother and cracks rougher. Each material uses a fixed preset band. Accent materials",
          "(ore specks, metal fittings) are separate regions with their own presets.", "",
          f"Emissive is not learned: {kb.meta['emissive_textures_ignored']} vanilla textures glow, but what glows",
          "and how brightly is left to the user (Emissive editor), so the generator never adds glow by itself.",
          "The E column in the evaluation therefore measures vanilla's glow that AutoVV leaves out.", "",
          "## Learned presets per material", "",
          "Values are the median over base regions, with 10th–90th percentiles in brackets (0–255).", "",
          "| Material | Textures | Metal | Emissive | Roughness dark→bright | Subsurface | Noisy | Border offset | Relief coef |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, e in sorted(kb.categories.items(), key=lambda kv: -kv[1]["textures"]):
        if c == "unknown":
            continue
        b = e["base"]
        if not b:
            continue
        fx = e.get("edge_effects", {})
        pm = lambda n: f"{b[n][1]:.0f}"  # noqa: E731
        L.append(f"| {e['label']} | {e['textures']} | {pm('M_dark')}–{pm('M_bright')} "
                 f"({b['metal_share']:.0%} metal) | {pm('E_dark')}–{pm('E_bright')} ({b['emissive_share']:.0%}) | "
                 f"{pm('R_dark')}→{pm('R_bright')} [{b['R_dark'][0]:.0f}–{b['R_dark'][2]:.0f} → "
                 f"{b['R_bright'][0]:.0f}–{b['R_bright'][2]:.0f}] | {pm('S_dark')}–{pm('S_bright')} "
                 f"({b['subsurface_share']:.0%}) | {b['noisy_share']:.0%} | "
                 f"{fx.get('border_offset', 0):+.1f} | {fx.get('relief_coef', 0):+.2f} |")
    L += ["", "## Accent regions per material", "",
          "| Material | Accent regions | Metal | Emissive | Roughness dark→bright | Subsurface |",
          "|---|---|---|---|---|---|"]
    for c, e in sorted(kb.categories.items(), key=lambda kv: -kv[1]["accent_regions"]):
        a = e.get("accent")
        if not a:
            continue
        pm = lambda n: f"{a[n][1]:.0f}"  # noqa: E731
        L.append(f"| {e['label']} | {e['accent_regions']} | {pm('M_dark')}–{pm('M_bright')} ({a['metal_share']:.0%}) | "
                 f"{pm('E_dark')}–{pm('E_bright')} ({a['emissive_share']:.0%}) | {pm('R_dark')}→{pm('R_bright')} | "
                 f"{pm('S_dark')}–{pm('S_bright')} ({a['subsurface_share']:.0%}) |")
    if evaluation:
        ev = evaluation
        L += ["", "## Hold-out evaluation", "",
              f"Each of the {ev['textures']} textures was regenerated by a knowledge base that had never seen it",
              f"({ev['folds']}-fold). The error is the mean absolute difference from the real vanilla map (0–255).", "",
              "*Name-blind* ignores similar vanilla names and relies on material and color only, which is what",
              "a brand-new modded texture gets.", "",
              "| Channel | AutoVV MAE | Median | Name-blind MAE | Flat-preset baseline |", "|---|---|---|---|---|"]
        nb = ev.get("name_blind", {}).get("mae", {})
        for ch in "MERS":
            L.append(f"| {ch} | {ev['mae'].get(ch, 0):.1f} | {ev['median_ae'].get(ch, 0):.1f} | "
                     f"{nb.get(ch, 0):.1f} | {ev['baseline_mae'].get(ch, 0):.1f} |")
        L += ["", f"Median per-texture correlation between generated and vanilla roughness: "
                  f"**{ev['roughness_correlation']}**", "",
              "| Material | M | E | R | S |", "|---|---|---|---|---|"]
        for c, d in ev["per_category"].items():
            L.append(f"| {c} | {d.get('M', 0)} | {d.get('E', 0)} | {d.get('R', 0)} | {d.get('S', 0)} |")
    if data.skipped:
        L += ["", "## Skipped", ""] + [f"- {s}" for s in data.skipped[:50]]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def train(roots: list[Path] | None = None, out: Path = DEFAULT_KB_PATH, evaluate_folds: int = 5,
          eval_limit: int | None = None, progress: Progress | None = None,
          log: Callable[[str], None] = print) -> tuple[KnowledgeBase, dict | None]:
    roots = roots or [DEFAULT_VANILLA]
    log(f"Scanning {', '.join(source_label(r) for r in roots)}")
    data = learn(roots, progress)
    log(f"Learned {len(data.kb.regions.features)} regions from {data.kb.meta['textures_used']} textures")
    ev = None
    if evaluate_folds:
        log("Running hold-out evaluation")
        ev = evaluate(data, evaluate_folds, eval_limit, progress)
        ev["name_blind"] = evaluate(data, evaluate_folds, eval_limit, progress, name_matching=False)
        data.kb.meta["evaluation"] = ev
        log(f"Hold-out MAE: {ev['mae']}  name-blind {ev['name_blind']['mae']}  "
            f"(flat baseline {ev['baseline_mae']})")
    data.kb.save(out)
    report = out.with_name("training_report.md")
    write_report(data, ev, report)
    log(f"Saved knowledge base to {out}\nReport: {report}")
    return data.kb, ev
