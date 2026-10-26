"""Settings tab: optional Ollama connection."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QVBoxLayout, QWidget)

from ..ollama_client import OllamaClient
from .widgets import set_value_units
from .workers import OllamaClassifyWorker, OllamaModelsWorker, start


class SettingsTab(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        s = self.state.settings
        root = QVBoxLayout(self)

        prog = QGroupBox("Program")
        pf = QFormLayout(prog)
        self.units = QComboBox()
        self.units.addItem("0 – 255 (like the MERS file)", "255")
        self.units.addItem("Percent (0 – 100%)", "%")
        self.units.setCurrentIndex(max(0, self.units.findData(s.slider_units)))
        self.units.setToolTip("How channel values are shown and typed in the editor sliders. "
                              "Files are always written as 0-255.")
        self.units.currentIndexChanged.connect(self._units_changed)
        pf.addRow("Slider values", self.units)
        root.addWidget(prog)

        box = QGroupBox("Ollama (optional)")
        form = QFormLayout(box)
        about = QLabel("MERS maps are always made by AutoVV's learned generator, never by a language model. "
                       "If you run <a href='https://ollama.com'>Ollama</a> locally, AutoVV can ask it one thing: "
                       "what material a texture's <i>name</i> refers to (for example \"nickel_block\" is a metal). "
                       "It is only asked when the built-in keyword list doesn't recognize a name, unless you tick "
                       "\"verify\" below. Recommended model: <b>qwen2.5:7b</b> (or <b>llama3.2:3b</b> on low-end PCs).")
        about.setWordWrap(True)
        about.setOpenExternalLinks(True)
        about.setObjectName("Hint")
        form.addRow(about)
        self.enabled = QCheckBox("Use Ollama to identify materials from texture names")
        self.enabled.setChecked(s.ollama_enabled)
        form.addRow(self.enabled)
        self.host = QLineEdit(s.ollama_host)
        form.addRow("Host", self.host)
        mr = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setMinimumWidth(260)
        if s.ollama_model:
            self.model.addItem(s.ollama_model)
        refresh = QPushButton("Load models")
        refresh.clicked.connect(self.load_models)
        mr.addWidget(self.model, 1)
        mr.addWidget(refresh)
        form.addRow("Model", mr)
        self.verify = QCheckBox("Also verify names the keyword list already recognized")
        self.verify.setChecked(s.ollama_verify_keywords)
        form.addRow(self.verify)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        form.addRow("Status", self.status)
        tr = QHBoxLayout()
        self.try_name = QLineEdit()
        self.try_name.setPlaceholderText("Try a texture name, e.g. nickel_block")
        test = QPushButton("Ask")
        test.clicked.connect(self.try_classify)
        tr.addWidget(self.try_name, 1)
        tr.addWidget(test)
        form.addRow("Test", tr)
        self.try_result = QLabel("")
        self.try_result.setWordWrap(True)
        form.addRow("", self.try_result)
        root.addWidget(box)

        save = QPushButton("Save settings")
        save.setObjectName("Primary")
        save.clicked.connect(self.save)
        root.addWidget(save)
        root.addStretch(1)
        for w in (self.enabled, self.verify):
            w.toggled.connect(self.save)

    def _units_changed(self) -> None:
        self.state.settings.slider_units = self.units.currentData()
        self.state.settings.save()
        set_value_units(self.state.settings.slider_units)
        self.state.unitsChanged.emit()
        self.state.status("Slider values: " + self.units.currentText())

    def save(self) -> None:
        s = self.state.settings
        s.ollama_enabled = self.enabled.isChecked()
        s.ollama_host = self.host.text().strip() or "http://localhost:11434"
        s.ollama_model = self.model.currentText().strip()
        s.ollama_verify_keywords = self.verify.isChecked()
        s.save()
        self.state.reset_ollama()
        self.state.status("Settings saved")

    def load_models(self) -> None:
        self.status.setText("Contacting Ollama…")
        w = OllamaModelsWorker(self.host.text().strip() or "http://localhost:11434")
        w.finished.connect(self._models)
        w.failed.connect(lambda err: self.status.setText(
            f"<span style='color:#e8795c'>Not reachable: {err.strip().splitlines()[-1]}</span><br>"
            "Install Ollama from ollama.com, run <code>ollama pull qwen2.5:7b</code>, and keep it running."))
        start(w, self)

    def _models(self, models: list[str]) -> None:
        current = self.model.currentText()
        self.model.clear()
        self.model.addItems(models)
        if current in models:
            self.model.setCurrentText(current)
        self.status.setText(f"<span style='color:#5fd38d'>Connected</span> — {len(models)} model(s) installed"
                            if models else "Connected, but no models are installed (try `ollama pull qwen2.5:7b`).")

    def try_classify(self) -> None:
        name = self.try_name.text().strip()
        if not name:
            return
        self.save()
        client = OllamaClient(self.state.settings.ollama_host, self.state.settings.ollama_model)
        self.try_result.setText("Thinking…")
        w = OllamaClassifyWorker(client, [(name, "")])
        w.finished.connect(lambda res: self.try_result.setText(self._fmt(res[(name, "")])))
        w.failed.connect(lambda err: self.try_result.setText(err.strip().splitlines()[-1]))
        start(w, self)

    def _fmt(self, result) -> str:
        key, why = result
        return f"<b>{self.state.label(key)}</b> — {why}" if key else why
