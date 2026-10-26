"""Material taxonomy shared by the trainer, the generator and the GUI.

Categories are grouped by the MERS trends observed in vanilla Minecraft
(roughness band, metalness level, subsurface usage). There is deliberately no
"glowing" material: what glows, and how brightly, is set by the user with the
per-channel editor (Emissive), not guessed from a name.
Name classification is purely rule based: texture names are split into
tokens and matched against keyword lists, most specific category first.

Block sound definitions (the ``sound`` field in a pack's blocks.json) are
deliberately never used: Minecraft often gives blocks the sound of a
different material, so a sound says nothing reliable about the surface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    key: str
    label: str
    description: str
    keywords: tuple[str, ...]


# Order matters: earlier categories win when several match.
MATERIALS: tuple[Material, ...] = (
    Material("ore", "Ore", "Stone with embedded mineral specks (two materials).",
             ("ore", "ores", "debris")),
    Material("glass", "Glass", "Glass and panes: metal-tinted, perfectly smooth.",
             ("glass", "pane", "tinted")),
    Material("ice", "Ice", "Smooth, translucent frozen blocks.",
             ("ice", "frosted", "icicle", "frozen")),
    Material("snow", "Snow", "Soft rough snow with light subsurface.",
             ("snow", "snowy", "snowed", "powder_snow")),
    Material("gem", "Gem", "Diamonds, emeralds, amethyst and other crystals.",
             ("diamond", "emerald", "amethyst", "lapis", "quartz", "crystal", "gem", "ruby",
              "sapphire", "topaz", "jade", "opal", "onyx", "peridot", "garnet", "amber",
              "jasper", "citrine", "tourmaline", "aquamarine", "cinnabar", "sulfur", "prismarine_crystals")),
    Material("metal", "Metal", "Iron, gold, copper, netherite and other metals.",
             ("iron", "gold", "golden", "copper", "netherite", "chain", "chains", "anvil", "rail",
              "rails", "bars", "cauldron", "hopper", "bell", "metal", "steel", "nickel", "tin",
              "silver", "platinum", "titanium", "bronze", "brass", "aluminum", "aluminium", "lead",
              "zinc", "cobalt", "tungsten", "uranium", "mithril", "adamantite", "chrome", "pipe",
              "grate", "plate", "ingot", "nugget", "oxidized", "weathered", "exposed", "waxed",
              "spawner", "vault", "heavy", "lightning", "rod", "camera", "hook", "gear", "machine",
              "lantern")),
    Material("foliage", "Leaves / Foliage", "Leaves and vines: strong subsurface scattering.",
             ("leaves", "leaf", "vine", "vines", "azalea", "foliage", "canopy", "hedge")),
    Material("plant", "Plant", "Flowers, saplings, crops and small plants.",
             ("flower", "sapling", "fern", "wheat", "carrots", "potatoes", "beetroot", "beetroots",
              "crop", "crops", "tulip", "rose", "dandelion", "poppy", "orchid", "allium",
              "cornflower", "lily", "daisy", "bush", "kelp", "seagrass", "reeds", "sugarcane",
              "roots", "sprouts", "stem", "tallgrass", "double_plant", "petals", "wildflowers",
              "dripleaf", "propagule", "berry", "berries", "pitcher", "torchflower", "blossom",
              "stage", "herb", "weed", "clover", "deadbush", "chorus",
              "waterlily", "eyeblossom", "vine", "cactus_flower")),
    Material("wood", "Wood", "Planks, logs, doors and wooden furniture.",
             ("planks", "plank", "log", "wood", "wooden", "stripped", "door", "trapdoor", "bamboo",
              "barrel", "bookshelf", "crafting", "chest", "sign", "shelf", "fence", "oak", "spruce",
              "birch", "jungle", "acacia", "mangrove", "cherry", "crimson", "warped", "hyphae",
              "board", "timber", "lumber", "beam", "maple", "redwood", "palm", "willow", "ladder",
              "lectern", "composter", "loom", "cartography", "fletching", "smithing", "jukebox",
              "noteblock", "scaffolding", "fletcher", "lever", "cardboard", "paper", "crate", "box")),
    Material("fabric", "Fabric / Wool", "Wool, carpet, beds and cloth.",
             ("wool", "carpet", "bed", "banner", "cloth", "fabric", "cotton", "silk", "linen",
              "curtain", "rug", "felt")),
    Material("concrete", "Concrete", "Smooth-ish matte concrete.",
             ("concrete", "cement", "plaster", "asphalt")),
    Material("terracotta", "Terracotta", "Terracotta, glazed and brick-like clay.",
             ("terracotta", "glazed", "pottery", "pot", "ceramic", "tile", "tiles")),
    Material("grass", "Grass", "Grass block tops and sides: very rough with a faint sheen (vanilla metalness 26).",
             ("grass", "turf", "lawn", "sod", "meadow")),
    Material("soil", "Soil / Sand", "Dirt, sand, gravel, clay, mud: very rough.",
             ("dirt", "sand", "gravel", "clay", "mud", "podzol", "mycelium", "farmland", "soil",
              "path", "powder", "coarse", "rooted", "suspicious", "ash", "dust", "silt")),
    Material("stone", "Stone", "Stone, bricks, deepslate, rock and masonry.",
             ("stone", "cobblestone", "cobble", "deepslate", "brick", "bricks", "andesite",
              "diorite", "granite", "tuff", "calcite", "basalt", "blackstone", "sandstone",
              "end", "netherrack", "obsidian", "purpur", "prismarine", "dripstone", "rock",
              "marble", "slate", "furnace", "smoker", "dispenser", "dropper", "observer",
              "piston", "stonecutter", "grindstone", "pillar", "chiseled", "polished",
              "cracked", "mossy", "smooth", "quarry", "boulder", "pebble", "masonry",
              "stonebrick", "bedrock", "lodestone", "endframe", "respawn", "anchor", "enchanting",
              "comparator", "repeater", "crafter")),
)

MATERIAL_KEYS: tuple[str, ...] = tuple(m.key for m in MATERIALS)
BY_KEY: dict[str, Material] = {m.key: m for m in MATERIALS}

# Specific multi-word overrides checked on the whole name before tokens.
# A key of None means "no preset" (auto-detect) and stops a misleading keyword from matching.
_PHRASES: tuple[tuple[str, str | None], ...] = (
    ("double_plant", "plant"),
    ("dry_grass", "plant"),
    ("seagrass", "plant"),
    ("tallgrass", "plant"),
    ("grass_path", "soil"),
    ("snow", "snow"),               # snowy grass sides are snow, not grass
    ("grass_block", "grass"),
    ("grass_top", "grass"),
    ("grass_side", "grass"),
    ("grass_carried", "grass"),
    ("moss_carpet", None),
    ("nylium", None),
    ("redstone_block", "gem"),
    ("coal_block", "stone"),
    ("grass_path", "soil"),
    ("mud_brick", "stone"),
    ("sandstone", "stone"),
    ("end_stone", "stone"),
    ("crying_obsidian", "stone"),
    ("concrete_powder", "soil"),
    ("tall_grass", "plant"),
    ("short_grass", "plant"),
    ("dragon_egg", "stone"),
    ("iron_bars", "metal"),
    ("glass_pane", "glass"),
    ("hardened_clay", "terracotta"),
    ("stained_silver", "terracotta"),
    ("shulker", None),
    ("silver_candle", None),
    ("wool_colored", "fabric"),
    ("carpet", "fabric"),
    ("concrete_silver", "concrete"),
)


def tokenize(name: str) -> list[str]:
    """Split a texture name like ``deepslate_diamond_ore_02`` into tokens."""
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()
    return [t for t in re.split(r"[^a-z]+", name) if t]


def classify_name(name: str, folder_hint: str = "") -> tuple[str | None, str]:
    """Return ``(material_key, reason)`` for a texture name, or ``(None, reason)``.

    ``folder_hint`` (e.g. ``textures/blocks/ores``) is used as a lower
    priority source of tokens.
    """
    lname = name.lower()
    for phrase, key in _PHRASES:
        if phrase in lname:
            return key, f"name contains '{phrase}'" if key else f"'{phrase}' has no preset"

    tokens = tokenize(name)
    for mat in MATERIALS:
        for kw in mat.keywords:
            if kw in tokens or ("_" in kw and kw in lname):
                return mat.key, f"keyword '{kw}'"

    if folder_hint:
        ftokens = tokenize(folder_hint)
        for mat in MATERIALS:
            for kw in mat.keywords:
                if kw in ftokens:
                    return mat.key, f"folder keyword '{kw}'"
    return None, "no keyword match"
