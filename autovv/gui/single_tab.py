"""Single file mode: drop a texture, preview the generated channels, save."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ..imaging import load_rgba
from ..materials import classify_name
from ..edits import TextureEdits
from ..pack import ensure_pbr_for
from ..pipeline import decide_material, save_outputs
from ..texture_set import map_name
from .editing import EditSession
from .widgets import ChannelEditor, ChannelStrip, DropZone, MaterialPicker, OutputOptionsBox
from .workers import OllamaClassifyWorker, start


class SingleFileTab(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self.path: Path | None = None
        self.rgba: np.ndarray | None = None
        self.result = None
        self.edits = TextureEdits()

        root = QHBoxLayout(self)
        root.setSpacing(16)

        # ---- left: drop zone + previews + summary
        left = QVBoxLayout()
        self.drop = DropZone()
        self.drop.fileSelected.connect(self.load)
        left.addWidget(self.drop)
        self.file_label = QLabel("No texture loaded")
        self.file_label.setObjectName("Title")
        left.addWidget(self.file_label)
        self.strip = ChannelStrip(columns=5, min_size=120)
        left.addWidget(self.strip)
        # Description of the selected channel (set by the edit session).
        self.legend = QLabel("")
        self.legend.setObjectName("Hint")
        left.addWidget(self.legend)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Region breakdown appears here after a texture is loaded.")
        self.summary.setMaximumHeight(140)
        left.addWidget(self.summary)
        left.addStretch(1)
        root.addLayout(left, 3)

        # ---- right: material + options + save
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        right.setContentsMargins(0, 0, 0, 0)
        self.picker = MaterialPicker()
        self.picker.changed.connect(self.schedule)
        right.addWidget(self.picker)
        self.detected = QLabel("")
        self.detected.setWordWrap(True)
        self.detected.setObjectName("Detected")
        right.addWidget(self.detected)
        self.editor = ChannelEditor()
        self.editor.changed.connect(self._edited)
        self.state.unitsChanged.connect(self.editor.apply_units)
        right.addWidget(self.editor)
        self.options = OutputOptionsBox(self.state.settings)
        self.options.changed.connect(self.schedule)
        right.addWidget(self.options)
        right.addStretch(1)
        self.save_here = QPushButton("Save next to texture")
        self.save_here.setObjectName("Primary")
        self.save_here.clicked.connect(lambda: self.save(None))
        self.save_as = QPushButton("Save to folder…")
        self.save_as.clicked.connect(self.save_to)
        for b in (self.save_here, self.save_as):
            b.setEnabled(False)
            right.addWidget(b)
        scroll = QScrollArea()
        scroll.setWidget(right_w)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(480)
        root.addWidget(scroll, 1)

        self._timer = QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self.regenerate)
        self.session = EditSession(self.strip, self.editor, self.legend)

    # ------------------------------------------------------------------
    def load(self, path: Path) -> None:
        try:
            self.rgba = load_rgba(path)
        except OSError as e:
            QMessageBox.warning(self, "AutoVV", f"Could not open {path.name}:\n{e}")
            return
        self.path = path
        self.edits = TextureEdits()
        self.picker.set_assignment({})     # color assignments belong to the previous texture
        h, w = self.rgba.shape[:2]
        self.file_label.setText(f"{path.name}   ·   {w}×{h}")
        self.drop.label.setText(f"{path.name}\n\nDrop another texture or click to browse")
        self.regenerate()
        self._ask_ollama()

    def _ask_ollama(self) -> None:
        """Ask Ollama about the name in the background; regenerate when it answers."""
        client = self.state.ollama()
        if client is None or self.path is None:
            return
        name, folder = self.path.stem, self.path.parent.name
        keyword, _ = classify_name(name, folder)
        if client.has_answer(name, folder) or (keyword and not self.state.settings.ollama_verify_keywords):
            return
        self.detected.setText(self.detected.text() + "<br><i>Asking Ollama about the name…</i>")
        w = OllamaClassifyWorker(client, [(name, folder)])
        w.finished.connect(lambda _res: self.regenerate())
        w.failed.connect(lambda _err: self.regenerate())
        start(w, self)

    def _edited(self) -> None:
        """Channel editor changed: re-apply edits to the cached result (no regeneration)."""
        if self.result is None:
            return
        self.result = self.session.reapply()
        self.summary.setPlainText(self.result.summary())

    def schedule(self) -> None:
        if self.rgba is not None:
            self._timer.start()

    def regenerate(self) -> None:
        if self.rgba is None or self.state.kb is None:
            return
        opts = self.options.options()
        self.options.store(self.state.settings)
        decision = decide_material(self.path.stem, self.path.parent.name, self.picker.selected(),
                                   self.state.ollama(cached_only=True),
                                   self.state.settings.ollama_verify_keywords)
        self.result = self.session.show(self.state.kb, self.rgba, self.path.stem, self.path.parent.name,
                                        decision, opts, self.edits, self.picker.assignment())
        self.picker.set_palette(self.session.swatches)
        self.summary.setPlainText(self.result.summary())
        mats = ", ".join(self.state.label(m) for m in self.result.materials) or "Auto"
        self.detected.setText(f"Using: <b>{mats}</b><br><span style='color:#9aa3b2'>"
                              f"{self.result.material_reason}</span>")
        out = map_name(self.path.stem, opts.mode) + "." + opts.image_format
        self.save_here.setText(f"Save next to texture  ({out})")
        for b in (self.save_here, self.save_as):
            b.setEnabled(True)

    def save_to(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Save MERS into folder",
                                                  str(self.path.parent) if self.path else "")
        if folder:
            self.save(Path(folder))

    def save(self, folder: Path | None) -> None:
        if self.result is None or self.path is None:
            return
        folder = folder or self.path.parent
        opts = self.options.options()
        map_path = folder / f"{map_name(self.path.stem, opts.mode)}.{opts.image_format}"
        if map_path.exists() and QMessageBox.question(
                self, "AutoVV", f"{map_path.name} already exists. Overwrite it?") != QMessageBox.StandardButton.Yes:
            return
        try:
            mp, ts = save_outputs(self.result, self.path.stem, folder, opts)
        except OSError as e:
            QMessageBox.warning(self, "AutoVV", f"Saving failed:\n{e}")
            return
        msg = f"Saved {mp.name}" + (f" and {ts.name}" if ts else "") + f" to {folder}"
        pbr = ensure_pbr_for(folder)          # inside a pack: make sure it declares "pbr"
        self.state.status(msg + (f" · manifest.json: {pbr}" if pbr else ""))
