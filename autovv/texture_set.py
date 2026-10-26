"""Reading and writing ``*.texture_set.json`` files."""
from __future__ import annotations

import json
import re
from pathlib import Path

MER_KEY = "metalness_emissive_roughness"
MERS_KEY = "metalness_emissive_roughness_subsurface"
FORMAT_MERS = "1.21.30"   # first version that accepts the subsurface layer
FORMAT_MER = "1.16.100"


def loads_lenient(text: str):
    """Parse Bedrock-style JSON: BOMs, // and /* */ comments, trailing commas."""
    text = text.lstrip("\ufeff")
    text = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', lambda m: m.group(1) or "", text, flags=re.S)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


def read_texture_set(path: Path) -> dict | None:
    """Return the ``minecraft:texture_set`` object of a file, or ``None``."""
    try:
        data = loads_lenient(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    ts = data.get("minecraft:texture_set") if isinstance(data, dict) else None
    return ts if isinstance(ts, dict) else None


def texture_set_path(folder: Path, color_name: str) -> Path:
    return Path(folder) / f"{color_name}.texture_set.json"


def map_name(color_name: str, mode: str) -> str:
    """Vanilla naming: ``<name>_mers`` for MERS maps, ``<name>_mer`` for RTX MER maps."""
    return f"{color_name}_{'mers' if mode == 'mers' else 'mer'}"


def write_texture_set(folder: Path, color_name: str, map_ref: str, mode: str = "mers") -> Path:
    """Create or update ``<color_name>.texture_set.json`` to reference ``map_ref``.

    An existing file keeps its other layers (normal, heightmap, uniform color).
    The other MER/MERS key is removed, because a set may not define both.
    """
    path = texture_set_path(folder, color_name)
    ts: dict = {}
    fmt = None
    if path.exists():
        try:
            data = loads_lenient(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                ts = dict(data.get("minecraft:texture_set") or {})
                fmt = data.get("format_version")
        except (OSError, ValueError):
            ts = {}
    if not isinstance(ts.get("color"), (str, list)):
        ts["color"] = color_name
    key, other = (MERS_KEY, MER_KEY) if mode == "mers" else (MER_KEY, MERS_KEY)
    ts.pop(other, None)
    ts[key] = map_ref
    wanted = FORMAT_MERS if mode == "mers" else FORMAT_MER
    if not fmt or _version_tuple(fmt) < _version_tuple(wanted):
        fmt = wanted
    # Keep "color" first, like vanilla files.
    ordered = {"color": ts.pop("color"), **ts}
    out = {"format_version": fmt, "minecraft:texture_set": ordered}
    path.write_text(json.dumps(out, indent=4) + "\n", encoding="utf-8")
    return path


def _version_tuple(v) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)
