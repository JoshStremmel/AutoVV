"""Load pack mode: browse a resource pack, multi-select textures/folders, batch generate."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QRadioButton, QScrollArea, QSplitter,
                               QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator, QVBoxLayout,
                               QWidget, QWidgetAction)

from ..imaging import load_rgba
from ..materials import MATERIALS, classify_name
from ..pack import PackScan, TextureEntry, discover_packs, read_pack_info
from ..edits import TextureEdits
from ..pack import MapFile, apply_existing_map, ensure_pbr_capability, find_maps
from ..pipeline import decide_material
from .editing import EditSession
from .widgets import THUMB_SIZE, ChannelEditor, ChannelStrip, MaterialPicker, OutputOptionsBox
from .workers import GenerateJob, GenerateWorker, ScanWorker, start

ROLE_ENTRY = Qt.ItemDataRole.UserRole
ROLE_OVERRIDE = Qt.ItemDataRole.UserRole + 1
ROLE_EDITS = Qt.ItemDataRole.UserRole + 2
ROLE_ASSIGN = Qt.ItemDataRole.UserRole + 3   # {material: [swatch, ...]} for this texture
COL_NAME, COL_MATERIAL, COL_STATUS = 0, 1, 2
GREEN, AMBER, GRAY = QColor("#5fd38d"), QColor("#e8b85c"), QColor("#8a93a3")


class PackTab(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self.scan: PackScan | None = None
        self.pack_root: Path | None = None
        self.worker: GenerateWorker | None = None
        self.maps: list[MapFile] = []

        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ---- pack chooser
        top = QHBoxLayout()
        top.addWidget(QLabel("Resource pack"))
        self.pack_combo = QComboBox()
        self.pack_combo.setMinimumWidth(360)
        self.pack_combo.activated.connect(self._combo_chosen)
        top.addWidget(self.pack_combo, 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_packs)
        browse = QPushButton("Open folder…")
        browse.clicked.connect(self.browse)
        self.system = QCheckBox("Show UI/system folders")
        self.system.toggled.connect(lambda: self.pack_root and self.open_pack(self.pack_root))
        for w in (refresh, browse, self.system):
            top.addWidget(w)
        root.addLayout(top)
        self.pack_label = QLabel("Pick one of your installed packs or open any pack folder.")
        self.pack_label.setObjectName("Hint")
        root.addWidget(self.pack_label)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        # ---- tree + selection tools
        tree_box = QWidget()
        tl = QVBoxLayout(tree_box)
        tl.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter textures…")
        self.filter.textChanged.connect(self.apply_filter)
        tools.addWidget(self.filter, 1)
        for text, fn in (("All", lambda: self.check_all(True)), ("None", lambda: self.check_all(False)),
                         ("Missing MER only", self.check_missing), ("Expand", lambda: self.tree.expandAll()),
                         ("Collapse", lambda: self.tree.collapseAll())):
            b = QPushButton(text)
            b.clicked.connect(fn)
            tools.addWidget(b)
        tl.addLayout(tools)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Texture", "Material", "MER"])
        self.tree.setColumnWidth(COL_NAME, 330)
        self.tree.setColumnWidth(COL_MATERIAL, 150)
        self.tree.setUniformRowHeights(True)
        self.tree.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.context_menu)
        self.tree.currentItemChanged.connect(self.preview)
        self.tree.itemChanged.connect(self._item_changed)
        tl.addWidget(self.tree, 1)
        self.count_label = QLabel("")
        self.count_label.setObjectName("Hint")
        tl.addWidget(self.count_label)
        split.addWidget(tree_box)

        # ---- right: preview + options + run
        right_w = QWidget()
        rl = QVBoxLayout(right_w)
        rl.setContentsMargins(0, 0, 0, 0)
        self.preview_title = QLabel("Select a texture to preview")
        self.preview_title.setObjectName("Title")
        rl.addWidget(self.preview_title)
        self.strip = ChannelStrip(columns=5, min_size=64, max_size=140)
        rl.addWidget(self.strip)
        self.channel_hint = QLabel("")
        self.channel_hint.setObjectName("Hint")
        rl.addWidget(self.channel_hint)
        self.preview_info = QLabel("")
        self.preview_info.setObjectName("Hint")
        self.preview_info.setWordWrap(True)
        rl.addWidget(self.preview_info)

        self.picker = MaterialPicker("Material override for all ticked textures (optional)", columns=3)
        self.picker.changed.connect(self._picker_changed)
        self._assign_item: QTreeWidgetItem | None = None
        rl.addWidget(self.picker)
        self.editor = ChannelEditor()
        self.editor.changed.connect(self._edited)
        self.state.unitsChanged.connect(self.editor.apply_units)
        rl.addWidget(self.editor)
        self.session = EditSession(self.strip, self.editor, self.channel_hint)

        row = QHBoxLayout()
        self.options = OutputOptionsBox(self.state.settings)
        self.options.changed.connect(lambda: self.preview(self.tree.currentItem()))
        row.addWidget(self.options)
        dest = QGroupBox("Destination")
        dl = QVBoxLayout(dest)
        self.in_place = QRadioButton("Write into the pack (next to each texture)")
        self.export = QRadioButton("Export to a separate folder")
        self.in_place.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.in_place)
        grp.addButton(self.export)
        dl.addWidget(self.in_place)
        dl.addWidget(self.export)
        er = QHBoxLayout()
        self.export_path = QLineEdit()
        self.export_path.setPlaceholderText("Export folder")
        eb = QPushButton("…")
        eb.setFixedWidth(32)
        eb.clicked.connect(self._choose_export)
        er.addWidget(self.export_path)
        er.addWidget(eb)
        dl.addLayout(er)
        self.overwrite = QCheckBox("Overwrite textures that already have MER/MERS")
        self.overwrite.setChecked(self.state.settings.overwrite_existing)
        dl.addWidget(self.overwrite)
        row.addWidget(dest)
        rl.addLayout(row)

        run = QHBoxLayout()
        self.generate_btn = QPushButton("Generate MER files")
        self.generate_btn.setObjectName("Primary")
        self.generate_btn.clicked.connect(self.generate)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        run.addWidget(self.generate_btn, 1)
        run.addWidget(self.cancel_btn)
        rl.addLayout(run)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        rl.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Generation log")
        rl.addWidget(self.log, 1)

        scroll = QScrollArea()
        scroll.setWidget(right_w)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        split.addWidget(scroll)
        split.setSizes([560, 640])

        self.refresh_packs()

    # ------------------------------------------------------------- packs
    def refresh_packs(self) -> None:
        self.pack_combo.clear()
        self.pack_combo.addItem("— choose a resource pack —", None)
        for p in discover_packs():
            icon = QIcon(str(p.icon)) if p.icon else QIcon()
            self.pack_combo.addItem(icon, p.display, str(p.path))

    def _combo_chosen(self, idx: int) -> None:
        path = self.pack_combo.itemData(idx)
        if path:
            self.open_pack(Path(path))

    def browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open resource pack folder")
        if folder:
            self.open_pack(Path(folder))

    def open_pack(self, root: Path) -> None:
        self.pack_root = root
        info = read_pack_info(root)
        name = info.name if info else root.name
        self.pack_label.setText(f"<b>{name}</b> — {root}" + ("" if info else
                                "  <span style='color:#e8b85c'>(no manifest.json found)</span>"))
        self.tree.clear()
        self.count_label.setText("Scanning…")
        w = ScanWorker(root, self.system.isChecked())
        w.finished.connect(self._scanned)
        w.failed.connect(lambda err: self.count_label.setText(f"Scan failed: {err.splitlines()[-1]}"))
        start(w, self)

    def _scanned(self, result) -> None:
        scan, thumbs = result
        self.scan = scan
        self.maps = find_maps(scan.root)
        self.tree.blockSignals(True)
        self.tree.clear()
        folder_items: dict[str, QTreeWidgetItem] = {}

        def folder_item(rel: str) -> QTreeWidgetItem:
            if rel in folder_items:
                return folder_items[rel]
            parent_rel, _, leaf = rel.rpartition("/")
            parent = folder_item(parent_rel) if parent_rel else self.tree.invisibleRootItem()
            it = QTreeWidgetItem(parent, [leaf or rel])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
            it.setCheckState(COL_NAME, Qt.CheckState.Unchecked)
            f = it.font(0)
            f.setBold(True)
            it.setFont(0, f)
            folder_items[rel] = it
            return it

        for rel, entries in scan.folders().items():
            parent = folder_item(rel) if rel else self.tree.invisibleRootItem()
            for e in entries:
                it = QTreeWidgetItem(parent, [e.name])
                if thumbs.get(e.key) is not None:
                    it.setIcon(COL_NAME, QIcon(QPixmap.fromImage(thumbs[e.key])))
                it.setData(COL_NAME, ROLE_ENTRY, e)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                self._decorate(it, e)
                it.setCheckState(COL_NAME, Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self.tree.expandToDepth(3)
        self._refresh_folder_states()

    def _decorate(self, it: QTreeWidgetItem, e: TextureEntry) -> None:
        override = it.data(COL_NAME, ROLE_OVERRIDE)
        if override:
            it.setText(COL_MATERIAL, self.state.label(override) + " (set)")
            it.setForeground(COL_MATERIAL, QBrush(AMBER))
        else:
            key, why = classify_name(e.name, e.rel_folder)
            it.setText(COL_MATERIAL, self.state.label(key) if key else "auto")
            it.setForeground(COL_MATERIAL, QBrush(QColor("#c9d1dd") if key else GRAY))
            it.setToolTip(COL_MATERIAL, why)
        assign = it.data(COL_NAME, ROLE_ASSIGN)
        if assign:
            it.setText(COL_MATERIAL, it.text(COL_MATERIAL) + "  + " + ", ".join(
                self.state.label(m) for m in assign) + " by color")
        edits = it.data(COL_NAME, ROLE_EDITS)
        if edits is not None and edits.active:
            it.setText(COL_MATERIAL, it.text(COL_MATERIAL) + "  ✎ edited")
            it.setToolTip(COL_MATERIAL, "\n".join(edits.describe()))
        status = {"image": ("✓ MERS/MER", GREEN), "uniform": ("✓ uniform", GREEN),
                  "set_without_mer": ("set, no MER", AMBER), "none": ("—", GRAY)}[e.mer_status]
        it.setText(COL_STATUS, status[0])
        it.setForeground(COL_STATUS, QBrush(status[1]))
        it.setToolTip(COL_NAME, str(e.color_path))

    # ---------------------------------------------------------- selection
    def _entries(self, checked_only: bool = False):
        it = QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            e = item.data(COL_NAME, ROLE_ENTRY)
            if e is not None and (not checked_only or item.checkState(COL_NAME) == Qt.CheckState.Checked):
                yield item, e
            it += 1

    def check_all(self, on: bool) -> None:
        self.tree.blockSignals(True)
        for item, _ in self._entries():
            if not item.isHidden():
                item.setCheckState(COL_NAME, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self._refresh_folder_states()

    def check_missing(self) -> None:
        self.tree.blockSignals(True)
        for item, e in self._entries():
            item.setCheckState(COL_NAME, Qt.CheckState.Unchecked if e.has_mer else Qt.CheckState.Checked)
        self.tree.blockSignals(False)
        self._refresh_folder_states()

    def _refresh_folder_states(self) -> None:
        """Recompute folder tick states bottom-up after changes made with signals blocked."""
        def sync(item: QTreeWidgetItem) -> Qt.CheckState:
            if item.data(COL_NAME, ROLE_ENTRY) is not None:
                return item.checkState(COL_NAME)
            states = {sync(item.child(i)) for i in range(item.childCount())}
            state = states.pop() if len(states) == 1 else Qt.CheckState.PartiallyChecked
            item.setData(COL_NAME, Qt.ItemDataRole.CheckStateRole, state)
            return state

        self.tree.blockSignals(True)
        # Clear auto-tristate while syncing so setting a folder doesn't re-propagate to children.
        folders = []
        it = QTreeWidgetItemIterator(self.tree)
        while it.value():
            if it.value().data(COL_NAME, ROLE_ENTRY) is None:
                folders.append(it.value())
                it.value().setFlags(it.value().flags() & ~Qt.ItemFlag.ItemIsAutoTristate)
            it += 1
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            if root.child(i).childCount():
                sync(root.child(i))
        for f in folders:
            f.setFlags(f.flags() | Qt.ItemFlag.ItemIsAutoTristate)
        self.tree.blockSignals(False)
        self.tree.viewport().update()
        self._update_counts()

    def _item_changed(self, _item, col) -> None:
        if col == COL_NAME:
            self._update_counts()

    def _update_counts(self) -> None:
        total = sum(1 for _ in self._entries())
        have = sum(1 for _, e in self._entries() if e.has_mer)
        ticked = sum(1 for _ in self._entries(True))
        self.count_label.setText(f"{total} textures · {have} already have MER/MERS · {ticked} ticked")
        self.generate_btn.setText(f"Generate MER files for {ticked} texture{'s' if ticked != 1 else ''}")
        self.generate_btn.setEnabled(ticked > 0 and self.worker is None)

    def apply_filter(self, text: str) -> None:
        text = text.lower().strip()

        def walk(item: QTreeWidgetItem) -> bool:
            e = item.data(COL_NAME, ROLE_ENTRY)
            if e is not None:
                show = not text or text in e.key.lower()
                item.setHidden(not show)
                return show
            any_child = False
            for i in range(item.childCount()):
                any_child |= walk(item.child(i))
            item.setHidden(not any_child)
            return any_child

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            walk(root.child(i))
        if text:
            self.tree.expandAll()

    def context_menu(self, pos) -> None:
        items = [i for i in self.tree.selectedItems() if i.data(COL_NAME, ROLE_ENTRY) is not None]
        if not items:
            return
        menu = QMenu(self)
        sub = menu.addMenu(f"Set material for {len(items)} selected")
        auto = QAction("Auto-detect", sub)
        auto.triggered.connect(lambda: self._set_override(items, None))
        sub.addAction(auto)
        sub.addSeparator()
        for m in MATERIALS:
            a = QAction(m.label, sub)
            a.triggered.connect(lambda _=False, k=m.key: self._set_override(items, k))
            sub.addAction(a)
        self._existing_map_menu(menu, items)
        tick = menu.addAction("Tick selected")
        tick.triggered.connect(lambda: [i.setCheckState(COL_NAME, Qt.CheckState.Checked) for i in items])
        untick = menu.addAction("Untick selected")
        untick.triggered.connect(lambda: [i.setCheckState(COL_NAME, Qt.CheckState.Unchecked) for i in items])
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _existing_map_menu(self, menu: QMenu, items) -> None:
        """Submenu with a search box listing every MER/MERS map already in the pack."""
        sub = menu.addMenu(f"Use existing MER/MERS for {len(items)} selected")
        if not self.maps:
            sub.addAction("No MER/MERS files in this pack").setEnabled(False)
            return
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8, 8, 8, 8)
        search = QLineEdit()
        search.setPlaceholderText(f"Search {len(self.maps)} MER/MERS files in this pack…")
        search.setClearButtonEnabled(True)
        bl.addWidget(search)
        results = QListWidget()
        results.setMinimumSize(440, 280)
        bl.addWidget(results)
        hint = QLabel("Click a file to use it. It is copied next to each texture, and resized if needed.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        bl.addWidget(hint)

        def fill(text: str = "") -> None:
            results.clear()
            words = text.lower().split()
            for m in self.maps:
                if all(w in m.rel.lower() for w in words):
                    li = QListWidgetItem(f"{m.rel}    [{m.kind.upper()}]")
                    li.setData(Qt.ItemDataRole.UserRole, m)
                    li.setToolTip(str(m.path))
                    results.addItem(li)
            if results.count():
                results.setCurrentRow(0)

        def choose(li: QListWidgetItem | None = None) -> None:
            li = li or results.currentItem()
            if li is not None:
                menu.close()
                self._use_existing_map(items, li.data(Qt.ItemDataRole.UserRole))

        search.textChanged.connect(fill)
        search.returnPressed.connect(choose)
        results.itemClicked.connect(choose)
        fill()
        act = QWidgetAction(sub)
        act.setDefaultWidget(box)
        sub.addAction(act)
        sub.aboutToShow.connect(lambda: QTimer.singleShot(0, search.setFocus))

    def _use_existing_map(self, items, source: MapFile) -> None:
        export_dir = Path(self.export_path.text().strip()) if (
            self.export.isChecked() and self.export_path.text().strip()) else None
        replacing = [it for it in items if it.data(COL_NAME, ROLE_ENTRY).has_mer]
        if replacing and QMessageBox.question(
                self, "AutoVV", f"{len(replacing)} of the selected textures already have a MER/MERS map.\n"
                f"Point them at {source.rel} instead?") != QMessageBox.StandardButton.Yes:
            return
        self.log.appendPlainText(f"Using {source.rel} for {len(items)} texture(s):")
        for it in items:
            e = it.data(COL_NAME, ROLE_ENTRY)
            out_dir = (export_dir / e.rel_folder) if export_dir else e.color_path.parent
            try:
                target, _ts, note = apply_existing_map(e.color_path, source, out_dir)
                self.log.appendPlainText(f"✓ {e.key}  →  {target.name} ({note})")
            except (OSError, ValueError) as ex:
                self.log.appendPlainText(f"✗ {e.key}: {ex}")
        target = export_dir or self.pack_root
        if target and (target / "manifest.json").exists():
            self.log.appendPlainText(f"manifest.json: {ensure_pbr_capability(target)}")
        self.state.status(f"Applied {source.path.name} to {len(items)} texture(s)")
        if self.pack_root and not export_dir:
            self.open_pack(self.pack_root)   # refresh the MER column

    def _set_override(self, items, key: str | None) -> None:
        for it in items:
            it.setData(COL_NAME, ROLE_OVERRIDE, key)
            self._decorate(it, it.data(COL_NAME, ROLE_ENTRY))
        self.preview(self.tree.currentItem())

    # ------------------------------------------------------------ preview
    def _materials_for(self, item: QTreeWidgetItem) -> list[str] | None:
        override = item.data(COL_NAME, ROLE_OVERRIDE)
        if override:
            return [override]
        return self.picker.selected() or None

    def preview(self, item: QTreeWidgetItem | None, *_):
        e = item.data(COL_NAME, ROLE_ENTRY) if item else None
        if e is None or self.state.kb is None:
            return
        try:
            rgba = load_rgba(e.color_path)
        except OSError as ex:
            self.preview_info.setText(f"Cannot read image: {ex}")
            return
        opts = self.options.options()
        ol = self.state.ollama(cached_only=True)
        decision = decide_material(e.name, e.rel_folder, self._materials_for(item), ol,
                                   self.state.settings.ollama_verify_keywords)
        edits = item.data(COL_NAME, ROLE_EDITS)
        if edits is None:
            edits = TextureEdits()           # each texture keeps its own edits
            item.setData(COL_NAME, ROLE_EDITS, edits)
        if item is not self._assign_item:
            # Color assignments are per texture; the ticked preset stays for all ticked textures.
            self._assign_item = item
            self.picker.set_assignment(item.data(COL_NAME, ROLE_ASSIGN))
        res = self.session.show(self.state.kb, rgba, e.name, e.rel_folder, decision, opts, edits,
                                self.picker.assignment())
        self.picker.set_palette(self.session.swatches)
        h, w = rgba.shape[:2]
        self.preview_title.setText(f"{e.name}   ·   {w}×{h}")
        mats = ", ".join(self.state.label(m) for m in res.materials) or "Auto"
        extra = "  ·  already has a MER/MERS map" if e.has_mer else ""
        self.preview_info.setText(f"Preview: {mats} ({res.material_reason}) · {len(res.regions)} region(s){extra}")

    def _picker_changed(self) -> None:
        item = self.tree.currentItem()
        e = item.data(COL_NAME, ROLE_ENTRY) if item else None
        if e is not None:
            item.setData(COL_NAME, ROLE_ASSIGN, self.picker.assignment() or None)
            self._decorate(item, e)
        self.preview(item)

    def _edited(self) -> None:
        """Channel editor changed: re-apply to the cached preview and mark the texture."""
        item = self.tree.currentItem()
        e = item.data(COL_NAME, ROLE_ENTRY) if item else None
        if e is None or self.session.auto is None:
            return
        self.session.reapply()
        self._decorate(item, e)

    # ----------------------------------------------------------- generate
    def _choose_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export folder")
        if folder:
            self.export_path.setText(folder)
            self.export.setChecked(True)

    def generate(self) -> None:
        if self.scan is None or self.state.kb is None:
            return
        jobs = []
        skipped = 0
        for item, e in self._entries(True):
            if e.has_mer and not self.overwrite.isChecked():
                skipped += 1
                continue
            edits = item.data(COL_NAME, ROLE_EDITS)
            jobs.append(GenerateJob(e, self._materials_for(item), edits if edits and edits.active else None,
                                    item.data(COL_NAME, ROLE_ASSIGN)))
        if not jobs:
            QMessageBox.information(self, "AutoVV", "Nothing to do: every ticked texture already has a MER map.\n"
                                    "Tick \"Overwrite textures that already have MER/MERS\" to replace them.")
            return
        export_dir = None
        if self.export.isChecked():
            if not self.export_path.text().strip():
                QMessageBox.warning(self, "AutoVV", "Choose an export folder first.")
                return
            export_dir = Path(self.export_path.text().strip())
        s = self.state.settings
        self.options.store(s)
        s.overwrite_existing = self.overwrite.isChecked()
        s.save()
        self.log.clear()
        if skipped:
            self.log.appendPlainText(f"Skipping {skipped} texture(s) that already have MER/MERS maps.")
        self.worker = GenerateWorker(self.state.kb, jobs, self.options.options(), self.pack_root, export_dir,
                                     self.state.ollama(), s.ollama_verify_keywords)
        self.worker.progress.connect(self._progress)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        start(self.worker, self)

    def cancel(self) -> None:
        if self.worker:
            self.worker.canceled = True

    def _progress(self, i: int, n: int, text: str) -> None:
        self.progress.setMaximum(max(n, 1))
        self.progress.setValue(i)
        self.progress.setFormat(f"{i}/{n}  {text}")

    def _done(self, result) -> None:
        done, errors = result
        self.worker = None
        self.cancel_btn.setEnabled(False)
        self.log.appendPlainText(f"\nFinished: {done} generated, {errors} failed.")
        self.state.status(f"Generated {done} MER file(s)" + (f", {errors} failed" if errors else ""))
        if self.pack_root and not self.export.isChecked():
            self.open_pack(self.pack_root)  # refresh MER status column
        self._update_counts()

    def _failed(self, err: str) -> None:
        self.worker = None
        self.cancel_btn.setEnabled(False)
        self.log.appendPlainText(err)
        self._update_counts()
