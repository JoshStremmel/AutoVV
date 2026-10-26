"""Learning session tab: (re)train the knowledge base from vanilla MERS files."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
                               QTextBrowser, QVBoxLayout, QWidget)

from ..knowledge import DEFAULT_KB_PATH
from ..trainer import DEFAULT_VANILLA, source_label
from .workers import TrainWorker, start


class LearnTab(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self.worker = None
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        src = QGroupBox("Vanilla texture folders to study")
        sl = QVBoxLayout(src)
        intro = QLabel("AutoVV studies every texture that has a vanilla texture_set.json and MER/MERS map: "
                       "how each channel follows the color's brightness, where material regions split, "
                       "how borders and crevices are treated and what each material's preset band is. "
                       "It then checks itself by regenerating textures it has not seen.")
        intro.setWordWrap(True)
        intro.setObjectName("Hint")
        sl.addWidget(intro)
        self.sources = QListWidget()
        for s in self.state.settings.training_sources or [str(DEFAULT_VANILLA)]:
            self._add_source(s)
        self.sources.setMaximumHeight(110)
        sl.addWidget(self.sources)
        row = QHBoxLayout()
        add = QPushButton("Add folder…")
        add.clicked.connect(self.add_source)
        rem = QPushButton("Remove")
        rem.clicked.connect(lambda: [self.sources.takeItem(self.sources.row(i))
                                     for i in self.sources.selectedItems()])
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        sl.addLayout(row)
        tip = QLabel("Tip: the vanilla pack's textures/items and textures/entity folders can be added too.")
        tip.setObjectName("Hint")
        tip.setWordWrap(True)
        sl.addWidget(tip)
        self.evaluate = QCheckBox("Run hold-out evaluation afterwards (recommended)")
        self.evaluate.setChecked(True)
        sl.addWidget(self.evaluate)
        left.addWidget(src)

        self.run_btn = QPushButton("Start learning session")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self.run)
        left.addWidget(self.run_btn)
        self.progress = QProgressBar()
        left.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        left.addWidget(self.log, 1)

        lw = QWidget()
        lw.setLayout(left)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(lw)
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(True)
        split.addWidget(self.report)
        split.setSizes([420, 780])
        root.addWidget(split)
        self.show_report()

    def _add_source(self, path: str) -> None:
        """List a folder; the built-in vanilla folder shows as "Vanilla Files" without its location."""
        item = QListWidgetItem(source_label(path))
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        self.sources.addItem(item)

    def add_source(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder with vanilla textures + texture sets")
        if folder:
            self._add_source(folder)

    def show_report(self) -> None:
        report = DEFAULT_KB_PATH.with_name("training_report.md")
        if report.exists():
            self.report.setMarkdown(report.read_text(encoding="utf-8"))
        else:
            self.report.setMarkdown("# No knowledge base yet\n\nRun a learning session to build one.")

    def run(self) -> None:
        sources = [Path(self.sources.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.sources.count())]
        missing = [str(s) for s in sources if not s.is_dir()]
        if not sources or missing:
            self.log.setPlainText("Missing folder(s):\n" + "\n".join(missing) if missing else "Add a folder first.")
            return
        self.state.settings.training_sources = [str(s) for s in sources]
        self.state.settings.save()
        self.log.clear()
        self.run_btn.setEnabled(False)
        self.worker = TrainWorker(sources, self.evaluate.isChecked(), DEFAULT_KB_PATH)
        self.worker.progress.connect(self._progress)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._failed)
        start(self.worker, self)

    def _progress(self, i: int, n: int, text: str) -> None:
        self.progress.setMaximum(max(n, 1))
        self.progress.setValue(i)
        self.progress.setFormat(f"{i}/{n}  {text}")

    def _done(self, kb) -> None:
        self.worker = None
        self.run_btn.setEnabled(True)
        self.state.set_kb(kb)
        self.show_report()
        self.log.appendPlainText("Knowledge base reloaded.")

    def _failed(self, err: str) -> None:
        self.worker = None
        self.run_btn.setEnabled(True)
        self.log.appendPlainText(err)
