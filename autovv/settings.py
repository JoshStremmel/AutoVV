"""Persistent user settings (``%APPDATA%/AutoVV/settings.json``)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def settings_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "AutoVV" / "settings.json"


@dataclass
class Settings:
    ollama_enabled: bool = False
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = ""
    ollama_verify_keywords: bool = False   # also ask Ollama when a keyword already matched
    mode: str = "mers"                     # "mers" (Vibrant Visuals) or "mer" (RTX)
    image_format: str = "tga"              # vanilla ships MERS maps as .tga
    noise: bool = True
    edge_effects: bool = True
    overwrite_existing: bool = False
    training_sources: list[str] = field(default_factory=list)
    slider_units: str = "255"              # "255" (0-255) or "%" (0-100)

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(settings_path().read_text(encoding="utf-8"))
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self) -> None:
        p = settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
