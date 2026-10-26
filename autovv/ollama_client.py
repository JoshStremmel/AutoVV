"""Optional Ollama helper that reads a texture *name* and suggests a material.

Ollama never touches pixels or MERS values. It is only asked what material
a name such as "nickel_block" refers to, and its answer is limited to the
material categories AutoVV knows. The MERS maps themselves are always made
by the learned, deterministic generator.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .materials import MATERIALS, MATERIAL_KEYS

DEFAULT_HOST = "http://localhost:11434"

_PROMPT = """You label Minecraft texture files with the physical material they depict.
Texture file name: "{name}"
Folder: "{folder}"

Think about what real-world thing the words refer to. For example "nickel" and "tin" are metals,
"jade" and "ruby" are gems, "maple" and "redwood" are woods, "basalt" is stone, and a name ending in
"_ore" is an ore. Judge only by what the thing is made of. Do not use the sound a block makes
(Minecraft often gives blocks the sound of a different material) or any gameplay property.
Choose exactly one category:
{categories}

Answer with JSON: {{"material": "<category key>", "reason": "<one short sentence>"}}"""


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, model: str = "", timeout: float = 30.0):
        self.host = host.rstrip("/") or DEFAULT_HOST
        self.model = model
        self.timeout = timeout
        self._cache: dict[tuple[str, str], tuple[str | None, str]] = {}

    def _request(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def models(self) -> list[str]:
        """Installed model names; raises ``OSError`` if Ollama is not reachable."""
        return [m["name"] for m in self._request("/api/tags", timeout=4).get("models", [])]

    def available(self) -> bool:
        try:
            self.models()
            return True
        except (OSError, ValueError):
            return False

    def has_answer(self, name: str, folder: str = "") -> bool:
        return (name, folder) in self._cache

    def cached_view(self) -> "_CachedView":
        """A stand-in that answers only from the cache, so UI previews never block."""
        return _CachedView(self)

    def classify(self, name: str, folder: str = "") -> tuple[str | None, str]:
        """Return ``(material_key, reason)``; ``(None, error)`` if Ollama can't answer."""
        key = (name, folder)
        if key in self._cache:
            return self._cache[key]
        if not self.model:
            return None, "no Ollama model selected"
        cats = "\n".join(f"- {m.key}: {m.label} - {m.description}" for m in MATERIALS)
        schema = {"type": "object",
                  "properties": {"material": {"type": "string", "enum": list(MATERIAL_KEYS)},
                                 "reason": {"type": "string"}},
                  "required": ["material", "reason"]}
        payload = {"model": self.model, "stream": False, "format": schema,
                   "prompt": _PROMPT.format(name=name, folder=folder or "-", categories=cats),
                   "options": {"temperature": 0}}
        try:
            resp = self._request("/api/generate", payload)
            ans = json.loads(resp.get("response", "{}"))
            mat = str(ans.get("material", "")).strip().lower()
            reason = str(ans.get("reason", "")).strip()
            result = (mat, f"Ollama: {reason}" if reason else "Ollama") if mat in MATERIAL_KEYS \
                else (None, f"Ollama gave an unknown material '{mat}'")
        except (OSError, ValueError, urllib.error.URLError) as e:
            result = (None, f"Ollama unavailable ({e})")
        self._cache[key] = result
        return result


class _CachedView:
    def __init__(self, client: OllamaClient):
        self._client = client

    def classify(self, name: str, folder: str = "") -> tuple[str | None, str]:
        if self._client.has_answer(name, folder):
            return self._client.classify(name, folder)
        return None, "Ollama not asked yet"


# Names with an unambiguous material, mostly outside the built-in keyword list,
# because those are the names Ollama is actually asked about.
BENCHMARK: tuple[tuple[str, str], ...] = (
    ("osmium_block", "metal"), ("palladium_block", "metal"), ("iridium_casing", "metal"),
    ("electrum_block", "metal"), ("agate_block", "gem"), ("morganite_cluster", "gem"),
    ("moonstone_block", "gem"), ("ebony_planks", "wood"), ("teak_log", "wood"),
    ("sequoia_bark", "wood"), ("limestone", "stone"), ("shale", "stone"),
    ("gneiss_bricks", "stone"), ("travertine", "stone"), ("loam", "soil"),
    ("dune_sand", "soil"), ("velvet_block", "fabric"), ("burlap", "fabric"),
    ("denim_block", "fabric"), ("stained_window", "glass"), ("glacier_ice", "ice"),
    ("lavender_bush", "plant"), ("ivy_leaves", "foliage"), ("willow_foliage", "foliage"),
    ("tin_ore", "ore"),
    ("bauxite_ore", "ore"), ("firn_block", "snow"), ("adobe_tiles", "terracotta"),
)


def benchmark(client: OllamaClient, progress=print) -> dict:
    """Ask ``client`` about every BENCHMARK name; return accuracy, mistakes and timing."""
    import time
    wrong, t0 = [], time.perf_counter()
    for name, expected in BENCHMARK:
        got, why = client.classify(name)
        if got != expected:
            wrong.append((name, expected, got, why))
        progress(f"  {'ok ' if got == expected else 'MISS'} {name:22s} -> {got}")
    secs = time.perf_counter() - t0
    return {"model": client.model, "accuracy": 1 - len(wrong) / len(BENCHMARK),
            "seconds_per_name": secs / len(BENCHMARK), "wrong": wrong}
