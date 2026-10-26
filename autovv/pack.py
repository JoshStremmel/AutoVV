"""Finding Bedrock resource packs and listing the textures inside them."""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .imaging import IMAGE_EXTS
from .texture_set import MER_KEY, MERS_KEY, loads_lenient, read_texture_set

# Folders that hold UI/system art rather than world textures; hidden unless asked for.
SYSTEM_FOLDERS = {"ui", "gui", "colormap", "environment", "persona_thumbnails", "map", "flame_atlas"}
# Suffixes of PBR companion maps; such an image is treated as a map when its base image exists.
MAP_SUFFIXES = ("_mers", "_mer", "_normal", "_heightmap", "_height", "_n", "_h", "_s", "_mre")
MIN_PBR_ENGINE = [1, 21, 120]


@dataclass
class PackInfo:
    name: str
    path: Path
    source: str                       # e.g. "resource_packs", "development_resource_packs"
    icon: Path | None = None
    description: str = ""

    @property
    def display(self) -> str:
        return f"{self.name}  [{self.source}]"


@dataclass
class TextureEntry:
    color_path: Path
    rel_folder: str                   # folder relative to the pack root, posix style
    name: str                         # stem, e.g. "iron_block"
    mer_status: str = "none"          # none | image | uniform | set_without_mer
    mer_path: Path | None = None
    texture_set: Path | None = None

    @property
    def has_mer(self) -> bool:
        return self.mer_status in ("image", "uniform")

    @property
    def key(self) -> str:
        return f"{self.rel_folder}/{self.name}".lstrip("/")


@dataclass
class PackScan:
    root: Path
    textures: list[TextureEntry] = field(default_factory=list)

    def folders(self) -> dict[str, list[TextureEntry]]:
        out: dict[str, list[TextureEntry]] = {}
        for t in self.textures:
            out.setdefault(t.rel_folder, []).append(t)
        return dict(sorted(out.items()))


# ------------------------------------------------------------- discovery
def minecraft_roots() -> list[Path]:
    """Every ``com.mojang`` folder we can find (GDK, per-user, legacy UWP, Preview)."""
    roots: list[Path] = []
    appdata = os.environ.get("APPDATA")
    local = os.environ.get("LOCALAPPDATA")
    if appdata:
        for product in ("Minecraft Bedrock", "Minecraft Bedrock Preview"):
            users = Path(appdata) / product / "Users"
            if users.is_dir():
                for u in sorted(users.iterdir()):
                    p = u / "games" / "com.mojang"
                    if p.is_dir():
                        roots.append(p)
    if local:
        for pkg in ("Microsoft.MinecraftUWP_8wekyb3d8bbwe", "Microsoft.MinecraftWindowsBeta_8wekyb3d8bbwe"):
            p = Path(local) / "Packages" / pkg / "LocalState" / "games" / "com.mojang"
            if p.is_dir():
                roots.append(p)
    return roots


def discover_packs() -> list[PackInfo]:
    packs: list[PackInfo] = []
    seen: set[Path] = set()
    for root in minecraft_roots():
        user = root.parent.parent.name
        for kind in ("resource_packs", "development_resource_packs"):
            folder = root / kind
            if not folder.is_dir():
                continue
            for d in sorted(folder.iterdir()):
                info = read_pack_info(d, kind if user == "Shared" or "Packages" in str(root) else f"{kind} ({user})")
                if info and info.path.resolve() not in seen:
                    seen.add(info.path.resolve())
                    packs.append(info)
    return packs


def read_pack_info(folder: Path, source: str = "folder") -> PackInfo | None:
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        return None
    name, desc = folder.name, ""
    try:
        m = loads_lenient(manifest.read_text(encoding="utf-8-sig"))
        header = m.get("header", {})
        name = str(header.get("name") or folder.name)
        desc = str(header.get("description") or "")
        types = {mod.get("type") for mod in m.get("modules", []) if isinstance(mod, dict)}
        if types and "resources" not in types:
            return None  # behavior pack, skin pack, world template...
    except (OSError, ValueError, AttributeError):
        pass
    if name in ("pack.name", "") or name.startswith("pack."):
        name = _lang_value(folder, "pack.name") or folder.name
    name = re.sub(r"§.", "", name).strip() or folder.name
    icon = folder / "pack_icon.png"
    return PackInfo(name, folder, source, icon if icon.is_file() else None, re.sub(r"§.", "", desc))


def _lang_value(folder: Path, key: str) -> str | None:
    lang = folder / "texts" / "en_US.lang"
    if not lang.is_file():
        return None
    for line in lang.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].split("\t#")[0].strip()
    return None


# --------------------------------------------------------------- scanning
def scan_pack(root: Path, include_system: bool = False) -> PackScan:
    root = Path(root)
    tex_root = root / "textures" if (root / "textures").is_dir() else root
    scan = PackScan(root)
    for folder, _dirs, files in os.walk(tex_root):
        folder = Path(folder)
        rel = folder.relative_to(root).as_posix()
        parts = set(Path(rel).parts[1:] if rel.startswith("textures") else Path(rel).parts)
        if not include_system and parts & SYSTEM_FOLDERS:
            continue
        scan.textures.extend(_scan_folder(folder, rel if rel != "." else "", files))
    scan.textures.sort(key=lambda t: (t.rel_folder, t.name))
    return scan


def _scan_folder(folder: Path, rel: str, files: list[str]) -> list[TextureEntry]:
    images: dict[str, Path] = {}
    for f in files:
        p = Path(f)
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            stem = p.stem
            # Minecraft picks .tga > .png > .jpg > .jpeg for the same name.
            if stem not in images or IMAGE_EXTS.index(ext) < IMAGE_EXTS.index(images[stem].suffix.lower()):
                images[stem] = folder / f
    if not images:
        return []

    sets: dict[str, tuple[Path, dict]] = {}
    map_refs: set[str] = set()
    for f in files:
        if f.lower().endswith(".texture_set.json"):
            ts = read_texture_set(folder / f)
            if ts is None:
                continue
            color = ts.get("color")
            key = color if isinstance(color, str) else f[: -len(".texture_set.json")]
            sets[key] = (folder / f, ts)
            for k in (MER_KEY, MERS_KEY, "normal", "heightmap"):
                if isinstance(ts.get(k), str):
                    map_refs.add(ts[k])

    entries = []
    for stem, path in sorted(images.items()):
        if stem in map_refs or _looks_like_map(stem, images):
            continue
        e = TextureEntry(path, rel, stem)
        if stem in sets:
            ts_path, ts = sets[stem]
            e.texture_set = ts_path
            ref = ts.get(MERS_KEY, ts.get(MER_KEY))
            if isinstance(ref, str):
                e.mer_status = "image"
                e.mer_path = images.get(ref)
            elif isinstance(ref, (list, int, float)) or (isinstance(ref, str) and ref.startswith("#")):
                e.mer_status = "uniform"
            else:
                e.mer_status = "set_without_mer"
        entries.append(e)
    return entries


def _looks_like_map(stem: str, images: dict[str, Path]) -> bool:
    low = stem.lower()
    for suf in MAP_SUFFIXES:
        if low.endswith(suf) and stem[: -len(suf)] in images:
            return True
    return False


# ---------------------------------------------------------------- manifest
def ensure_pbr_capability(pack_root: Path) -> str:
    """Add the ``pbr`` capability (and a new enough min_engine_version) to manifest.json.

    A one-time ``manifest.json.bak`` backup is written before the first change.
    Returns a short human-readable description of what happened.
    """
    manifest = Path(pack_root) / "manifest.json"
    if not manifest.is_file():
        return "no manifest.json found"
    try:
        data = loads_lenient(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        return f"manifest.json could not be parsed ({e}); add \"pbr\" to capabilities manually"
    changes = []
    caps = data.get("capabilities")
    if not isinstance(caps, list):
        caps = []
    if "pbr" not in caps and "raytraced" not in caps:
        caps.append("pbr")
        data["capabilities"] = caps
        changes.append('added "pbr" capability')
    header = data.setdefault("header", {})
    mev = header.get("min_engine_version")
    if isinstance(mev, list) and all(isinstance(x, int) for x in mev) and mev < MIN_PBR_ENGINE:
        header["min_engine_version"] = list(MIN_PBR_ENGINE)
        changes.append(f"min_engine_version {mev} -> {MIN_PBR_ENGINE}")
    if not changes:
        return "manifest already Vibrant Visuals ready"
    backup = manifest.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(manifest, backup)
    manifest.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    return "; ".join(changes)


# ------------------------------------------------------- reusing existing maps
@dataclass
class MapFile:
    path: Path
    rel: str                          # path relative to the pack root, posix style
    kind: str                         # "mers" or "mer"


def find_maps(root: Path) -> list[MapFile]:
    """Every MER/MERS image in a pack: referenced by a texture set, or named ``*_mer`` / ``*_mers``."""
    root = Path(root)
    tex_root = root / "textures" if (root / "textures").is_dir() else root
    found: dict[Path, str] = {}
    for folder, _dirs, files in os.walk(tex_root):
        folder = Path(folder)
        images = {Path(f).stem: folder / f for f in files if Path(f).suffix.lower() in IMAGE_EXTS}
        for f in files:
            if not f.lower().endswith(".texture_set.json"):
                continue
            ts = read_texture_set(folder / f) or {}
            for key, kind in ((MERS_KEY, "mers"), (MER_KEY, "mer")):
                ref = ts.get(key)
                if isinstance(ref, str) and ref in images:
                    found[images[ref]] = kind
        for stem, p in images.items():
            low = stem.lower()
            if p not in found and (low.endswith("_mers") or low.endswith("_mer")):
                found[p] = "mers" if low.endswith("_mers") else "mer"
    return sorted((MapFile(p, p.relative_to(root).as_posix(), k) for p, k in found.items()),
                  key=lambda m: m.rel.lower())


def apply_existing_map(color_path: Path, source: MapFile, out_dir: Path | None = None) -> tuple[Path, Path, str]:
    """Make ``color_path`` use an existing MER/MERS map.

    If the map already sits in the output folder at the texture's size it is referenced
    directly; otherwise it is copied (resized with nearest-neighbor if the sizes differ)
    to ``<texture>_mers``/``_mer`` next to the texture. Returns ``(map, texture_set, note)``.
    """
    from PIL import Image

    from .imaging import load_rgba, save_map
    from .texture_set import map_name, write_texture_set

    color_path = Path(color_path)
    out_dir = Path(out_dir or color_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(color_path) as im:
        size = im.size
    with Image.open(source.path) as im:
        src_size = im.size
    channels = 4 if source.kind == "mers" else 3
    if source.path.parent.resolve() == out_dir.resolve() and src_size == size:
        target, note = source.path, "referenced in place"
    else:
        target = out_dir / f"{map_name(color_path.stem, source.kind)}{source.path.suffix.lower()}"
        arr = load_rgba(source.path)
        if src_size != size:
            arr = np.array(Image.fromarray(arr).resize(size, Image.NEAREST))
            note = f"copied and resized {src_size[0]}x{src_size[1]} -> {size[0]}x{size[1]}"
        else:
            note = "copied"
        save_map(target, arr, channels)
    ts = write_texture_set(out_dir, color_path.stem, target.stem, source.kind)
    return target, ts, note


def find_pack_root(path: Path) -> Path | None:
    """The nearest folder above ``path`` that holds a manifest.json, if any."""
    for folder in [Path(path)] + list(Path(path).parents):
        if (folder / "manifest.json").is_file():
            return folder
    return None


def ensure_pbr_for(path: Path) -> str | None:
    """After writing a texture set: add the "pbr" capability to its pack if missing.

    Returns a message, or None when ``path`` is not inside a pack.
    """
    root = find_pack_root(path)
    return ensure_pbr_capability(root) if root else None
