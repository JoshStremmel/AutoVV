"""Turn a color texture on disk into a MERS/MER map plus texture_set.json."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .edits import TextureEdits, apply_edits
from .generator import GenerationResult, generate
from .imaging import fill_transparent, load_rgba, save_map
from .knowledge import KnowledgeBase
from .materials import classify_name
from .ollama_client import OllamaClient
from .palette import detect_palette
from .texture_set import map_name, write_texture_set


@dataclass
class OutputOptions:
    mode: str = "mers"            # "mers" or "mer"
    image_format: str = "tga"     # "tga" or "png"
    noise: bool = True
    edge_effects: bool = True
    write_texture_set: bool = True


@dataclass
class MaterialDecision:
    materials: list[str] | None   # None -> let the generator decide from color
    reason: str


@dataclass
class ProcessResult:
    source: Path
    map_path: Path | None
    texture_set_path: Path | None
    result: GenerationResult


def decide_material(name: str, folder_hint: str = "", chosen: list[str] | None = None,
                    ollama: OllamaClient | None = None, verify_keywords: bool = False) -> MaterialDecision:
    """User choice > keyword match > Ollama (optional) > color match inside the generator."""
    if chosen:
        return MaterialDecision(list(chosen), "selected")
    key, why = classify_name(name, folder_hint)
    if ollama and (key is None or verify_keywords):
        o_key, o_why = ollama.classify(name, folder_hint)
        if o_key:
            return MaterialDecision([o_key], o_why)
    if key:
        return MaterialDecision([key], f"name: {why}")
    return MaterialDecision(None, "")


def generate_for(kb: KnowledgeBase, rgba, name: str, folder_hint: str = "",
                 decision: MaterialDecision | None = None,
                 options: OutputOptions | None = None,
                 edits: TextureEdits | None = None,
                 assignment: dict[str, list[int]] | None = None) -> GenerationResult:
    """Generate a map for one texture.

    ``assignment`` maps a material to palette swatches (see :mod:`autovv.palette`): those
    colors are generated as that material, and ``decision`` covers every other color.
    """
    options = options or OutputOptions()
    decision = decision or decide_material(name, folder_hint)
    assignment = {m: list(c) for m, c in (assignment or {}).items() if c}
    if assignment:
        res = _generate_by_color(kb, rgba, name, folder_hint, decision, options, assignment)
    else:
        res = generate(kb, rgba, name, decision.materials, folder_hint,
                       noise=options.noise, edge_effects=options.edge_effects,
                       chosen=decision.reason == "selected")
        if decision.materials and decision.reason:
            res.material_reason = decision.reason
    if edits is not None and edits.active:
        labels, _ = detect_palette(rgba, name)
        res.mers = apply_edits(res.mers, rgba[..., 3] > 0, labels, edits)
        res.notes += edits.describe()
    return res


def _generate_by_color(kb, rgba, name, folder_hint, decision, options,
                        assignment: dict[str, list[int]]) -> GenerationResult:
    """Assigned colors take their values from the whole texture generated as their material.

    Generating the whole texture (not just the chosen colors) keeps every color's value
    tied to its brightness within the full texture, exactly as if the preset had been
    applied to all of it; the chosen colors' pixels are then copied into the result.
    """
    opaque = rgba[..., 3] > 0
    labels, _ = detect_palette(rgba, name)
    groups: list[tuple[list[str] | None, np.ndarray]] = []
    taken = np.zeros(opaque.shape, dtype=bool)
    for material, swatches in assignment.items():
        mask = np.isin(labels, swatches) & opaque & ~taken
        taken |= mask
        groups.append(([material], mask))
    rest = opaque & ~taken
    if rest.any():
        groups.append((decision.materials, rest))   # the ticked material, or auto

    out = np.zeros(rgba.shape[:2] + (4,), dtype=np.float32)
    materials: list[str] = []
    regions = []
    for mats, mask in groups:
        if not mask.any():
            continue
        chosen = mats is not None and (mats != decision.materials or decision.reason == "selected")
        r = generate(kb, rgba, name, mats, folder_hint,
                     noise=options.noise, edge_effects=options.edge_effects, chosen=chosen)
        out[mask] = r.mers[mask]
        materials += [m for m in r.materials if m not in materials]
        regions += r.regions
    out = fill_transparent(out, opaque)
    mers = np.clip(np.round(out), 0, 255).astype(np.uint8)
    parts = [f"{m}: {len(c)} color(s)" for m, c in assignment.items()]
    if rest.any():
        parts.append("rest: " + ("+".join(decision.materials) if decision.materials else "auto"))
    return GenerationResult(mers, materials, "by color — " + ", ".join(parts), regions)


def save_outputs(res: GenerationResult, color_name: str, out_dir: Path,
                 options: OutputOptions) -> tuple[Path, Path | None]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = map_name(color_name, options.mode)
    map_path = out_dir / f"{ref}.{options.image_format}"
    save_map(map_path, res.mers, 4 if options.mode == "mers" else 3)
    ts = write_texture_set(out_dir, color_name, ref, options.mode) if options.write_texture_set else None
    return map_path, ts


def process_file(kb: KnowledgeBase, path: Path, out_dir: Path | None = None,
                 decision: MaterialDecision | None = None, folder_hint: str = "",
                 options: OutputOptions | None = None,
                 edits: TextureEdits | None = None,
                 assignment: dict[str, list[int]] | None = None) -> ProcessResult:
    options = options or OutputOptions()
    path = Path(path)
    rgba = load_rgba(path)
    res = generate_for(kb, rgba, path.stem, folder_hint, decision, options, edits, assignment)
    map_path, ts = save_outputs(res, path.stem, out_dir or path.parent, options)
    return ProcessResult(path, map_path, ts, res)
