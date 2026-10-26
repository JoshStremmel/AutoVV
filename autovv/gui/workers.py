"""Background jobs so the window stays responsive."""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from ..knowledge import KnowledgeBase
from ..ollama_client import OllamaClient
from ..pack import TextureEntry, ensure_pbr_capability, scan_pack
from ..edits import TextureEdits
from ..pipeline import MaterialDecision, OutputOptions, decide_material, process_file


class Worker(QObject):
    progress = Signal(int, int, str)
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def run(self):  # pragma: no cover - executed in a QThread
        try:
            self.finished.emit(self.work())
        except Exception:  # noqa: BLE001 - surface any failure to the user
            self.failed.emit(traceback.format_exc())

    def work(self):
        raise NotImplementedError


def start(worker: Worker, owner: QObject) -> QThread:
    """Run ``worker`` on a new thread owned by ``owner``; returns the thread."""
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    owner._jobs = getattr(owner, "_jobs", []) + [(thread, worker)]  # keep references alive
    thread.start()
    return thread


class ScanWorker(Worker):
    def __init__(self, root: Path, include_system: bool):
        super().__init__()
        self.root, self.include_system = root, include_system

    def work(self):
        scan = scan_pack(self.root, self.include_system)
        from .widgets import thumbnail
        # QImage (unlike QPixmap) may be built here, off the GUI thread.
        thumbs = {e.key: thumbnail(e.color_path) for e in scan.textures}
        return scan, thumbs


@dataclass
class GenerateJob:
    entry: TextureEntry
    materials: list[str] | None      # per-item or global override
    edits: TextureEdits | None = None  # per-texture channel / color adjustments
    assignment: dict[str, list[int]] | None = None   # per-texture materials by color


class GenerateWorker(Worker):
    def __init__(self, kb: KnowledgeBase, jobs: list[GenerateJob], options: OutputOptions,
                 pack_root: Path, export_dir: Path | None, ollama: OllamaClient | None,
                 verify_keywords: bool):
        super().__init__()
        self.kb, self.jobs, self.options = kb, jobs, options
        self.pack_root, self.export_dir = pack_root, export_dir
        self.ollama, self.verify = ollama, verify_keywords
        self.canceled = False

    def work(self):
        done, errors = 0, 0
        for i, job in enumerate(self.jobs):
            if self.canceled:
                self.log.emit("Canceled.")
                break
            e = job.entry
            self.progress.emit(i, len(self.jobs), e.key)
            try:
                decision = decide_material(e.name, e.rel_folder, job.materials, self.ollama, self.verify)
                out_dir = (self.export_dir / e.rel_folder) if self.export_dir else e.color_path.parent
                r = process_file(self.kb, e.color_path, out_dir, decision, e.rel_folder, self.options,
                                 job.edits, job.assignment)
                mats = "+".join(r.result.materials) or "auto"
                self.log.emit(f"✓ {e.key}  →  {r.map_path.name}   [{mats}; {r.result.material_reason}]")
                done += 1
            except Exception as ex:  # noqa: BLE001 - keep going with the other textures
                errors += 1
                self.log.emit(f"✗ {e.key}: {ex}")
        self.progress.emit(len(self.jobs), len(self.jobs), "done")
        if done:
            # Vibrant Visuals ignores texture sets unless the pack declares "pbr": add it if missing.
            target = self.export_dir or self.pack_root
            if (target / "manifest.json").exists():
                self.log.emit(f"manifest.json: {ensure_pbr_capability(target)}")
            elif self.export_dir:
                self.log.emit("Export folder has no manifest.json; add \"pbr\" to the pack's capabilities.")
        return done, errors


class TrainWorker(Worker):
    def __init__(self, sources: list[Path], evaluate: bool, out: Path):
        super().__init__()
        self.sources, self.evaluate, self.out = sources, evaluate, out

    def work(self):
        from ..trainer import train
        kb, ev = train(self.sources, self.out, evaluate_folds=5 if self.evaluate else 0,
                       progress=lambda i, n, s: self.progress.emit(i, n, s), log=self.log.emit)
        return kb


class OllamaModelsWorker(Worker):
    def __init__(self, host: str):
        super().__init__()
        self.host = host

    def work(self):
        return OllamaClient(self.host).models()


class OllamaClassifyWorker(Worker):
    def __init__(self, client: OllamaClient, names: list[tuple[str, str]]):
        super().__init__()
        self.client, self.names = client, names

    def work(self):
        out = {}
        for i, (name, folder) in enumerate(self.names):
            self.progress.emit(i, len(self.names), name)
            out[(name, folder)] = self.client.classify(name, folder)
        return out


__all__ = ["GenerateJob", "GenerateWorker", "MaterialDecision", "OllamaClassifyWorker",
           "OllamaModelsWorker", "ScanWorker", "TrainWorker", "start"]
