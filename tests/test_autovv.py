import json

import numpy as np
import pytest
from PIL import Image

from autovv.generator import generate
from autovv.imaging import gray, load_rgba
from autovv.knowledge import KnowledgeBase
from autovv.materials import classify_name
from autovv.pack import ensure_pbr_capability, scan_pack
from autovv.pipeline import OutputOptions, process_file
from autovv.regions import fit_region
from autovv.texture_set import MER_KEY, MERS_KEY, loads_lenient, write_texture_set


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load()


def _stone(seed=0, size=16):
    rng = np.random.default_rng(seed)
    v = rng.choice([103, 116, 126, 142], size=(size, size))
    return np.dstack([v, v, v, np.full_like(v, 255)]).astype(np.uint8)


@pytest.mark.parametrize("name,expected", [
    ("nickel_block", "metal"), ("deepslate_ruby_ore", "ore"), ("maple_planks", "wood"),
    ("light_blue_wool", "fabric"), ("furnace_front_on", "stone"), ("oak_leaves", "foliage"),
    ("hardened_clay_stained_silver", "terracotta"), ("glass_pane_top", "glass"),
    ("grass_top", "grass"), ("grass_side_carried", "grass"), ("double_plant_grass_top", "plant"),
    ("tallgrass", "plant"), ("grass_block_snow", "snow"), ("grass_path_top", "soil"),
])
def test_classify_name(name, expected):
    assert classify_name(name)[0] == expected


def test_levels_model_fits_exact_remap():
    rgba = _stone()
    g = gray(rgba)
    t = (g - g.min()) / (g.max() - g.min())
    mers = np.zeros_like(rgba)
    mers[..., 2] = np.round(227 - t * (227 - 155))
    params, err = fit_region(g, mers, rgba[..., 3] > 0)
    assert err < 0.5
    assert abs(params[4] - 227) < 1 and abs(params[5] - 155) < 1


def test_stone_generation_is_vanilla_like(kb):
    rgba = _stone()
    res = generate(kb, rgba, "stone_variant", ["stone"])
    m = res.mers.astype(int)
    assert m.shape == (16, 16, 4)
    assert m[..., 0].max() == 0 and m[..., 1].max() == 0
    # brighter pixels must be smoother (lower roughness)
    g = gray(rgba)
    assert np.corrcoef(g.ravel(), m[..., 2].ravel())[0, 1] < -0.8
    assert 140 <= m[..., 2].min() and m[..., 2].max() <= 240


def test_metal_generation_is_metallic(kb):
    v = np.linspace(170, 240, 256).reshape(16, 16).astype(np.uint8)
    rgba = np.dstack([v, v, v, np.full_like(v, 255)])
    res = generate(kb, rgba, "nickel_block", ["metal"])
    assert res.mers[..., 0].mean() > 100


def test_fully_transparent_texture(kb):
    rgba = np.zeros((16, 16, 4), np.uint8)
    res = generate(kb, rgba, "invis")
    assert res.mers.max() == 0


def test_deterministic(kb):
    rgba = _stone(3)
    a = generate(kb, rgba, "dirt_like", ["soil"]).mers
    b = generate(kb, rgba, "dirt_like", ["soil"]).mers
    assert np.array_equal(a, b)


def test_texture_set_merge_keeps_normal(tmp_path):
    (tmp_path / "rock.texture_set.json").write_text(json.dumps({
        "format_version": "1.16.100",
        "minecraft:texture_set": {"color": "rock", MER_KEY: "old_mer", "normal": "rock_normal"}}))
    p = write_texture_set(tmp_path, "rock", "rock_mers", "mers")
    d = json.loads(p.read_text())
    ts = d["minecraft:texture_set"]
    assert d["format_version"] == "1.21.30"
    assert ts[MERS_KEY] == "rock_mers" and MER_KEY not in ts and ts["normal"] == "rock_normal"


def test_lenient_json():
    assert loads_lenient('﻿{"a": "http://x", // c\n "b": [1,2,],}') == {"a": "http://x", "b": [1, 2]}


def _make_pack(root):
    blocks = root / "textures" / "blocks"
    blocks.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "format_version": 2, "header": {"name": "Test", "min_engine_version": [1, 20, 0]},
        "modules": [{"type": "resources"}]}))
    Image.fromarray(_stone()).save(blocks / "rock.png")
    Image.fromarray(_stone(1)).save(blocks / "done.png")
    Image.fromarray(_stone(2)).save(blocks / "done_mers.png")
    (blocks / "done.texture_set.json").write_text(json.dumps(
        {"format_version": "1.21.30", "minecraft:texture_set": {"color": "done", MERS_KEY: "done_mers"}}))
    ui = root / "textures" / "ui"
    ui.mkdir()
    Image.fromarray(_stone()).save(ui / "button.png")
    return blocks


def test_scan_pack(tmp_path):
    _make_pack(tmp_path)
    scan = scan_pack(tmp_path)
    names = {t.name: t for t in scan.textures}
    assert set(names) == {"rock", "done"}          # map files and UI folder excluded
    assert names["done"].has_mer and not names["rock"].has_mer
    assert "button" in {t.name for t in scan_pack(tmp_path, include_system=True).textures}


def test_block_sounds_are_ignored(tmp_path, kb):
    """A blocks.json sound of "metal" must not make a stone texture metallic."""
    blocks = _make_pack(tmp_path)
    (tmp_path / "blocks.json").write_text(json.dumps(
        {"format_version": [1, 1, 0], "custom:rock": {"textures": "rock", "sound": "metal"}}))
    r = process_file(kb, blocks / "rock.png", options=OutputOptions())
    assert "metal" not in r.result.materials
    assert load_rgba(r.map_path)[..., 0].max() == 0


def test_process_file_and_manifest(tmp_path, kb):
    blocks = _make_pack(tmp_path)
    r = process_file(kb, blocks / "rock.png", options=OutputOptions())
    assert r.map_path.name == "rock_mers.tga"
    assert load_rgba(r.map_path).shape == (16, 16, 4)
    assert json.loads(r.texture_set_path.read_text())["minecraft:texture_set"][MERS_KEY] == "rock_mers"
    msg = ensure_pbr_capability(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert "pbr" in m["capabilities"] and m["header"]["min_engine_version"] == [1, 21, 120]
    assert (tmp_path / "manifest.json.bak").exists() and "pbr" in msg


def test_mer_mode_writes_three_channels(tmp_path, kb):
    blocks = _make_pack(tmp_path)
    r = process_file(kb, blocks / "rock.png", options=OutputOptions(mode="mer", image_format="png"))
    assert r.map_path.name == "rock_mer.png"
    assert Image.open(r.map_path).mode == "RGB"
    ts = json.loads(r.texture_set_path.read_text())
    assert ts["format_version"] == "1.16.100" and MER_KEY in ts["minecraft:texture_set"]


def _lamp():
    """Gray frame with a bright orange 4x4 center."""
    rgba = _stone(5)
    rgba[6:10, 6:10] = (255, 170, 40, 255)
    return rgba


def test_palette_is_deterministic():
    from autovv.palette import detect_palette
    a_labels, a_sw = detect_palette(_lamp(), "lamp")
    b_labels, b_sw = detect_palette(_lamp(), "lamp")
    assert np.array_equal(a_labels, b_labels) and [s.rgb for s in a_sw] == [s.rgb for s in b_sw]
    shares = [sw.share for sw in a_sw]
    assert shares == sorted(shares, reverse=True)       # most common first


def test_palette_lists_every_color():
    from autovv.palette import detect_palette
    rgba = _lamp()
    labels, swatches = detect_palette(rgba, "lamp")
    exact = {tuple(c) for c in rgba[..., :3].reshape(-1, 3)}
    assert {sw.rgb for sw in swatches} == exact and not any(sw.grouped for sw in swatches)
    for sw in swatches:                                 # each swatch is exactly its color's pixels
        assert (rgba[labels == sw.index][:, :3] == sw.rgb).all()


def test_palette_groups_photo_like_textures():
    from autovv.palette import GROUPED_SWATCHES, detect_palette
    rng = np.random.default_rng(1)
    rgba = np.dstack([rng.integers(0, 256, (64, 64, 3)), np.full((64, 64), 255)]).astype(np.uint8)
    labels, swatches = detect_palette(rgba, "noise")
    assert 1 < len(swatches) <= GROUPED_SWATCHES and all(sw.grouped for sw in swatches)
    assert (labels >= 0).all()


def _orange(swatches):
    return next(sw.index for sw in swatches if sw.rgb == (255, 170, 40))


def test_nothing_glows_unless_asked(kb):
    res = generate(kb, _lamp(), "lamp_on", None)
    assert res.mers[..., 1].max() == 0


def test_color_value_sets_only_that_color(kb):
    from autovv.edits import TextureEdits
    from autovv.palette import detect_palette
    from autovv.pipeline import MaterialDecision, generate_for
    rgba = _lamp()
    _, swatches = detect_palette(rgba, "lamp")
    edits = TextureEdits()
    edits.channels["E"].colors[_orange(swatches)] = 180
    e = generate_for(kb, rgba, "lamp", "", MaterialDecision(["stone"], "selected"), edits=edits).mers[..., 1]
    assert (e[6:10, 6:10] == 180).all()
    e[6:10, 6:10] = 0
    assert e.max() == 0


def test_overall_value_skips_manual_colors(kb):
    from autovv.edits import TextureEdits, apply_edits
    from autovv.palette import detect_palette
    rgba = _lamp()
    labels, swatches = detect_palette(rgba, "lamp")
    auto = generate(kb, rgba, "lamp", ["stone"]).mers
    edits = TextureEdits()
    edits.channels["R"].colors[_orange(swatches)] = 20
    edits.channels["R"].overall = 100
    r = apply_edits(auto, rgba[..., 3] > 0, labels, edits)[..., 2].astype(int)
    assert (r[6:10, 6:10] == 20).all()                    # manual color untouched by overall
    rest = np.ones(r.shape, bool)
    rest[6:10, 6:10] = False
    assert r[rest].max() == 100                             # overall = new peak ...
    a = auto[..., 2].astype(float)[rest]
    assert np.corrcoef(a, r[rest])[0, 1] > 0.99             # ... keeping the learned variation


def test_overall_fills_an_empty_channel(kb):
    from autovv.edits import TextureEdits, apply_edits
    from autovv.palette import detect_palette
    rgba = _stone()
    labels, _ = detect_palette(rgba, "rock")
    edits = TextureEdits()
    edits.channels["E"].overall = 40
    e = apply_edits(generate(kb, rgba, "rock", ["stone"]).mers, rgba[..., 3] > 0, labels, edits)[..., 1]
    assert (e == 40).all()


def test_use_existing_map(tmp_path):
    from autovv.pack import apply_existing_map, find_maps
    blocks = _make_pack(tmp_path)
    maps = find_maps(tmp_path)
    assert [m.rel for m in maps] == ["textures/blocks/done_mers.png"] and maps[0].kind == "mers"
    # same folder, same size -> referenced directly, no copy
    target, ts, note = apply_existing_map(blocks / "rock.png", maps[0])
    assert target == blocks / "done_mers.png" and note == "referenced in place"
    assert json.loads(ts.read_text())["minecraft:texture_set"][MERS_KEY] == "done_mers"
    # different size -> copied and resized next to the texture
    Image.fromarray(np.dstack([_stone()] * 1)[:8, :8]).save(blocks / "small.png")
    target, ts, note = apply_existing_map(blocks / "small.png", maps[0])
    assert target.name == "small_mers.png" and "resized" in note
    assert load_rgba(target).shape == (8, 8, 4)
    assert json.loads(ts.read_text())["minecraft:texture_set"][MERS_KEY] == "small_mers"


def test_material_by_color(kb):
    """Assigned colors get their material; every other color gets the ticked one."""
    from autovv.palette import detect_palette
    from autovv.pipeline import MaterialDecision, generate_for
    rgba = _stone(7)
    rgba[4:8, 4:8] = (200, 200, 205, 255)          # a gray plate
    labels, swatches = detect_palette(rgba, "plate")
    plate = next(s.index for s in swatches if s.rgb == (200, 200, 205))
    res = generate_for(kb, rgba, "plate", "", MaterialDecision(["wood"], "selected"),
                       assignment={"metal": [plate]})
    m = res.mers[..., 0]
    assert (m[4:8, 4:8] > 100).all()                # metal where assigned
    m[4:8, 4:8] = 0
    assert m.max() == 0                             # wood (non-metal) everywhere else
    assert "metal" in res.materials and "wood" in res.materials


def test_by_color_values_match_whole_texture(kb):
    """Assigned colors keep the values they'd get if the whole texture were that material."""
    from autovv.palette import detect_palette
    from autovv.pipeline import MaterialDecision, generate_for
    rgba = _stone(9)
    rgba[2:6, 2:10] = (150, 150, 155, 255)
    rgba[8:12, 2:10] = (210, 210, 215, 255)                # two metal shades
    labels, swatches = detect_palette(rgba, "machine")
    metal = [sw.index for sw in swatches if sw.rgb in ((150, 150, 155), (210, 210, 215))]
    by = generate_for(kb, rgba, "machine", "", MaterialDecision(["stone"], "selected"),
                      assignment={"metal": metal}).mers
    whole = generate_for(kb, rgba, "machine", "", MaterialDecision(["metal"], "selected")).mers
    m = np.isin(labels, metal)
    assert np.array_equal(by[m], whole[m])
    assert (by[..., 0][m] >= 20).all()                     # a chosen Metal really is metallic
    dark, light = by[2:6, 2:10, 2].mean(), by[8:12, 2:10, 2].mean()
    assert dark != light                                   # roughness follows brightness
