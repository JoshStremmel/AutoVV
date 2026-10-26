"""Main window and shared application state."""
from __future__ import annotations

import sys

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QTabWidget

from .. import __version__
from ..knowledge import DEFAULT_KB_PATH, KnowledgeBase
from ..materials import BY_KEY
from ..ollama_client import OllamaClient
from ..settings import Settings
from .learn_tab import LearnTab
from .pack_tab import PackTab
from .settings_tab import SettingsTab
from .single_tab import SingleFileTab
from .theme import STYLESHEET
from .widgets import set_value_units


class AppState(QObject):
    """Things every tab needs: knowledge base, settings, Ollama client, status bar."""

    kbChanged = Signal()
    unitsChanged = Signal()
    statusMessage = Signal(str)

    def __init__(self):
        super().__init__()
        self.settings = Settings.load()
        set_value_units(self.settings.slider_units)
        self.kb: KnowledgeBase | None = None
        self._ollama: OllamaClient | None = None
        try:
            self.kb = KnowledgeBase.load(DEFAULT_KB_PATH)
        except (OSError, ValueError, KeyError):
            self.kb = None

    def set_kb(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.kbChanged.emit()

    def ollama(self, cached_only: bool = False):
        """The Ollama client if enabled. ``cached_only`` never blocks on a network call."""
        s = self.settings
        if not (s.ollama_enabled and s.ollama_model):
            return None
        if self._ollama is None:
            self._ollama = OllamaClient(s.ollama_host, s.ollama_model)
        return self._ollama.cached_view() if cached_only else self._ollama

    def reset_ollama(self) -> None:
        self._ollama = None

    def label(self, key: str | None) -> str:
        return BY_KEY[key].label if key in BY_KEY else (key or "auto")

    def status(self, text: str) -> None:
        self.statusMessage.emit(text)


def make_icon() -> QIcon:
    """A small four-quadrant block: metal, emissive, roughness, subsurface."""
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    cols = ("#c9d3e0", "#ffcf5a", "#7a8699", "#6fdc8c")
    for i, c in enumerate(cols):
        p.fillRect(QRectF(4 + (i % 2) * 28, 4 + (i // 2) * 28, 28, 28), QColor(c))
    p.setPen(QColor("#1b1d22"))
    p.drawRect(QRectF(4, 4, 55, 55))
    p.end()
    return QIcon(pix)


class MainWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle(f"AutoVV {__version__} — Vibrant Visuals MER generator")
        self.setWindowIcon(make_icon())
        self.resize(1320, 860)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(SingleFileTab(state), "Single File")
        tabs.addTab(PackTab(state), "Load Pack")
        tabs.addTab(LearnTab(state), "Learn")
        tabs.addTab(SettingsTab(state), "Settings")
        self.setCentralWidget(tabs)

        bar = self.statusBar()
        credit = QLabel("  Created by Josh Stremmel")
        bar.addWidget(credit)
        # Messages get their own label: showMessage() would hide the credit while shown.
        self.message = QLabel()
        self.message.setObjectName("Hint")
        bar.addWidget(self.message, 1)
        self._message_timer = QTimer(self, singleShot=True, interval=8000)
        self._message_timer.timeout.connect(self.message.clear)
        self.kb_label = QLabel()
        bar.addPermanentWidget(self.kb_label)
        state.statusMessage.connect(self._show_message)
        state.kbChanged.connect(self._kb_status)
        self._kb_status()
        if state.kb is None:
            tabs.setCurrentIndex(2)
            QMessageBox.information(self, "AutoVV", "No knowledge base was found.\n"
                                    "Open the Learn tab and run a learning session on the vanilla textures.")

    def _show_message(self, text: str) -> None:
        self.message.setText("   ·   " + text)
        self._message_timer.start()

    def _kb_status(self) -> None:
        kb = self.state.kb
        if kb is None:
            self.kb_label.setText("No knowledge base  ")
            return
        m = kb.meta
        self.kb_label.setText(f"Knowledge base: {m.get('regions', '?')} regions from "
                              f"{m.get('textures_used', '?')} vanilla textures · trained {m.get('trained_at', '?')}  ")

    def closeEvent(self, e):
        self.state.settings.save()
        super().closeEvent(e)


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AutoVV")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(make_icon())
    win = MainWindow(AppState())
    win.show()
    return app.exec()
