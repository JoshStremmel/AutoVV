"""Reusable Qt widgets: image views, drop zone, material picker, output options."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QIntValidator, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget,
                               QWidgetAction)

from ..edits import CHANNEL_NAMES, CHANNELS
from ..imaging import IMAGE_EXTS, load_rgba
from ..materials import BY_KEY, MATERIALS
from ..palette import sorted_by_hue

CHANNEL_HINTS = {
    "M": "White = metallic\nBlack = matte",
    "E": "White = full brightness\nBlack = no light",
    "R": "White = rough\nBlack = smooth",
    "S": "White = light passes through\nBlack = solid",
}

COLLAPSED_SWATCHES = 15   # 3 rows of 5 before "See all"
SWATCH_GAP = 6


def share_text(share: float) -> str:
    """Share of the texture as a percentage; tiny colors don't show as 0%."""
    return f"{share:.0%}" if share >= 0.01 else "<1%"


IMAGE_FILTER = "Textures (" + " ".join(f"*{e}" for e in IMAGE_EXTS) + ")"


def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = np.ascontiguousarray(np.dstack([arr, arr, arr, np.full_like(arr, 255)]))
    elif arr.shape[2] == 3:
        arr = np.ascontiguousarray(np.dstack([arr, np.full((h, w), 255, np.uint8)]))
    return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def first_frame(arr: np.ndarray) -> np.ndarray:
    """Vertical flipbooks (e.g. 16x512) preview as their first square frame."""
    h, w = arr.shape[:2]
    return arr[:w] if h > w and h % w == 0 else arr


THUMB_SIZE = 28


def thumbnail(path, size: int = THUMB_SIZE) -> QImage | None:
    """Small pixel-sharp preview of a texture (first frame of a flipbook). Safe off the GUI thread."""
    try:
        arr = first_frame(load_rgba(path))
    except (OSError, ValueError):
        return None
    return to_qimage(arr).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.FastTransformation)


def channel_image(mers: np.ndarray, channel: int, opaque: np.ndarray | None) -> np.ndarray:
    c = mers[..., channel]
    alpha = np.where(opaque, 255, 0).astype(np.uint8) if opaque is not None else np.full_like(c, 255)
    return np.dstack([c, c, c, alpha])


class ImageView(QWidget):
    """Pixel-art friendly image display: nearest-neighbor scaling over a checkerboard.

    Optionally clickable (channel previews) with a selected state.
    """

    clicked = Signal()

    def __init__(self, caption: str = "", min_size: int = 96, clickable: bool = False, parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        self._caption = caption
        self._placeholder = ""
        self._clickable = clickable
        self._selected = False
        self.setMinimumSize(min_size, min_size + (18 if caption else 0))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip(f"Click to edit {caption.lower()}")

    def set_array(self, arr: np.ndarray | None) -> None:
        self._pix = QPixmap.fromImage(to_qimage(first_frame(arr))) if arr is not None else None
        if arr is not None:
            self._placeholder = ""
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def set_selected(self, on: bool) -> None:
        self._selected = on
        self.update()

    def mousePressEvent(self, e):
        if self._clickable and e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        cap_h = 18 if self._caption else 0
        side = max(8, min(self.width(), self.height() - cap_h))
        x0 = (self.width() - side) / 2
        rect = QRectF(x0, 0, side, side)
        # checkerboard
        cell = max(4, side / 16)
        p.save()
        p.setClipRect(rect)
        p.fillRect(rect, QColor(52, 54, 60))
        y = 0.0
        row = 0
        while y < side:
            x = 0.0
            col = row % 2
            while x < side:
                if col % 2:
                    p.fillRect(QRectF(x0 + x, y, cell, cell), QColor(64, 66, 73))
                x += cell
                col += 1
            y += cell
            row += 1
        p.restore()
        if self._pix is not None:
            pw, ph = self._pix.width(), self._pix.height()
            s = side / max(pw, ph)
            tw, th = pw * s, ph * s
            target = QRectF(x0 + (side - tw) / 2, (side - th) / 2, tw, th)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawPixmap(target, self._pix, QRectF(0, 0, pw, ph))
        elif self._placeholder:
            p.setPen(QColor(150, 155, 165))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._placeholder)
        if self._selected:
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawRect(rect.adjusted(0, 0, -1, -1))
        else:
            p.setPen(QColor(90, 94, 104))
            p.drawRect(rect.adjusted(0, 0, -1, -1))
        if cap_h:
            f = p.font()
            f.setBold(self._selected)
            p.setFont(f)
            p.setPen(QColor("#ffffff") if self._selected else QColor(190, 195, 205))
            p.drawText(QRectF(0, side + 2, self.width(), cap_h - 2), Qt.AlignmentFlag.AlignCenter, self._caption)
        p.end()


class ChannelStrip(QWidget):
    """Color texture plus the four generated channels. Channel previews are clickable."""

    CAPTIONS = ("Color", "Metalness", "Emissive", "Roughness", "Subsurface")
    channelSelected = Signal(int)          # 0..3 = M, E, R, S

    def __init__(self, columns: int = 5, min_size: int = 96, max_size: int = 260, parent=None):
        super().__init__(parent)
        self.columns, self.max_size = columns, max_size
        self.rows = (len(self.CAPTIONS) + columns - 1) // columns
        self.mode = "mers"
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.views = [ImageView(c, min_size, clickable=i > 0) for i, c in enumerate(self.CAPTIONS)]
        for i, v in enumerate(self.views):
            grid.addWidget(v, i // columns, i % columns)
            if i > 0:
                v.clicked.connect(lambda ch=i - 1: self._clicked(ch))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _clicked(self, ch: int) -> None:
        if ch == 3 and self.mode == "mer":
            return  # subsurface is not written in MER mode
        self.set_selected(ch)
        self.channelSelected.emit(ch)

    def set_selected(self, ch: int | None) -> None:
        for i, v in enumerate(self.views[1:]):
            v.set_selected(i == ch)

    def resizeEvent(self, e):
        # Keep each preview square: height follows the available width.
        side = min(self.max_size, (self.width() - 10 * (self.columns - 1)) // self.columns)
        h = self.rows * (side + 18) + 10 * (self.rows - 1)
        if self.height() != h:
            self.setFixedHeight(max(h, 60))
        super().resizeEvent(e)

    def show_result(self, rgba: np.ndarray | None, mers: np.ndarray | None, mode: str = "mers") -> None:
        self.mode = mode
        self.views[0].set_array(rgba)
        opaque = rgba[..., 3] > 0 if rgba is not None else None
        for i in range(4):
            v = self.views[i + 1]
            if mers is None:
                v.set_array(None)
            elif i == 3 and mode == "mer":
                v.set_array(None)
                v.set_placeholder("not used\nin MER mode")
            else:
                v.set_array(channel_image(mers, i, opaque))


class DropZone(QFrame):
    """Drag-and-drop target that also opens a file dialog when clicked."""

    fileSelected = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(120)
        lay = QVBoxLayout(self)
        self.label = QLabel("Drop a texture here\nor click to browse\n\n.png  .tga  .jpg")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("DropLabel")
        lay.addWidget(self.label)

    def _set_hover(self, on: bool) -> None:
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, _):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a texture", "", IMAGE_FILTER)
        if path:
            self.fileSelected.emit(Path(path))

    def dragEnterEvent(self, e):
        if self._first_image(e.mimeData()):
            e.acceptProposedAction()
            self._set_hover(True)

    def dragLeaveEvent(self, _):
        self._set_hover(False)

    def dropEvent(self, e):
        self._set_hover(False)
        path = self._first_image(e.mimeData())
        if path:
            self.fileSelected.emit(path)

    @staticmethod
    def _first_image(mime) -> Path | None:
        for url in mime.urls() if mime.hasUrls() else []:
            p = Path(url.toLocalFile())
            if p.suffix.lower() in IMAGE_EXTS:
                return p
        return None


class MaterialPicker(QGroupBox):
    """Material presets: one at a time, optionally given to specific colors only.

    * Ticking a material replaces the previous one (unticking = auto-detect).
    * "Assign to colors" gives the ticked material to chosen colors only. It then
      stays ticked and highlighted (tooltip lists its colors), and another material
      can be ticked for the colors that are left. When every color has a material,
      the remaining presets are disabled.
    * Clicking a highlighted material selects it for editing; it never drops its colors.
      Colors are only released with "Remove" in its dropdown. Colors owned by one
      preset are grayed out in every other preset's dropdown.
    """

    changed = Signal()

    def __init__(self, title: str = "Material (optional)", columns: int = 2, parent=None):
        super().__init__(title, parent)
        lay = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        self.boxes: dict[str, QCheckBox] = {}
        for i, m in enumerate(MATERIALS):
            cb = QCheckBox(m.label)
            cb.setToolTip(m.description)
            cb.clicked.connect(lambda on, k=m.key: self._clicked(k, on))
            self.boxes[m.key] = cb
            grid.addWidget(cb, i // columns, i % columns)
        lay.addLayout(grid)
        row = QHBoxLayout()
        self.assign_btn = QToolButton()
        self.assign_btn.setObjectName("AssignButton")
        self.assign_btn.setText("Assign to colors")
        self.assign_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.assign_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.assign_menu = QMenu(self.assign_btn)
        self.assign_menu.aboutToShow.connect(self._fill_menu)
        self.assign_btn.setMenu(self.assign_menu)
        row.addWidget(self.assign_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.hint = QLabel("Tick one material, or leave all unticked to auto-detect. "
                           "Use “Assign to colors” to give it only some colors, then tick another for the rest.")
        self.hint.setWordWrap(True)
        self.hint.setObjectName("Hint")
        lay.addWidget(self.hint)

        self.base: str | None = None               # ticked, covers every unassigned color
        self.active: str | None = None             # the material "Assign to colors" edits
        self.assigned: dict[str, list[int]] = {}   # material -> swatch indices
        self.swatches = []
        self._refresh()

    # ------------------------------------------------------------ state
    def selected(self) -> list[str]:
        """The material for colors without an assignment (empty = auto-detect)."""
        return [self.base] if self.base else []

    def assignment(self) -> dict[str, list[int]]:
        return {m: list(c) for m, c in self.assigned.items() if c}

    def set_palette(self, swatches) -> None:
        """Colors of the current texture (assignments refer to these swatch indices)."""
        self.swatches = list(swatches)
        valid = {s.index for s in self.swatches}
        self.assigned = {m: [c for c in cs if c in valid] for m, cs in self.assigned.items()}
        self.assigned = {m: cs for m, cs in self.assigned.items() if cs}
        self._refresh()

    def set_assignment(self, assignment: dict[str, list[int]] | None) -> None:
        """Load a texture's color assignments without emitting ``changed``."""
        self.assigned = {m: list(c) for m, c in (assignment or {}).items() if c}
        if self.base in self.assigned:
            self.base = None
        self.active = self.base
        self._refresh()

    def clear(self) -> None:
        self.base, self.assigned, self.active = None, {}, None
        self._refresh()
        self.changed.emit()

    def _free_colors(self) -> list:
        used = {c for cs in self.assigned.values() for c in cs}
        return [s for s in self.swatches if s.index not in used]

    # ---------------------------------------------------------- handlers
    def _clicked(self, key: str, on: bool) -> None:
        if key in self.assigned:
            # A highlighted preset keeps its colors: clicking it only makes it the one
            # "Assign to colors" edits. Colors are removed from inside that dropdown.
            self.active = key
            self._refresh()
            return
        if on:
            self.base = self.active = key   # one preset at a time: replaces the previous tick
        elif self.base == key:
            self.base = None
            if self.active == key:
                self.active = None
        self._refresh()
        self.changed.emit()

    def _fill_menu(self) -> None:
        self.assign_menu.clear()
        mat = self.active
        if mat is None:
            return
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8, 6, 8, 6)
        grouped = any(sw.grouped for sw in self.swatches)
        bl.addWidget(QLabel(f"Colors that are <b>{BY_KEY[mat].label}</b> "
                            f"({len(self.swatches)}{' groups of similar colors' if grouped else ''}):"))
        owner = {c: m for m, cs in self.assigned.items() for c in cs if m != mat}
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 6, 0)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        scroll.setFixedHeight(min(len(self.swatches), 12) * 30 + 6)   # scrolls past 12 colors
        bl.addWidget(scroll)
        for sw in sorted_by_hue(self.swatches):
            cb = QCheckBox(f"{sw.hex}   ·   {share_text(sw.share)}")
            pix = QPixmap(14, 14)
            pix.fill(QColor(sw.hex))
            cb.setIcon(QIcon(pix))
            cb.setChecked(sw.index in self.assigned.get(mat, []))
            if sw.index in owner:
                # Already another preset's: shown grayed out and unticked so it can't be taken over.
                other = BY_KEY[owner[sw.index]].label
                dim = QColor(sw.hex)
                dim.setAlpha(70)
                pix.fill(QColor("#1e2026"))
                p = QPainter(pix)
                p.fillRect(0, 0, 14, 14, dim)
                p.end()
                cb.setIcon(QIcon(pix))
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.setText(f"{sw.hex}   ·   {share_text(sw.share)}   — {other}")
                cb.setToolTip(f"Already {other}. Select {other} and remove it there to free this color.")
            cb.toggled.connect(lambda on, s=sw.index: self._toggle_color(mat, s, on))
            il.addWidget(cb)
        il.addStretch(1)
        row = QHBoxLayout()
        if mat in self.assigned:
            remove = QPushButton(f"Remove {BY_KEY[mat].label}")
            remove.setToolTip(f"Take {BY_KEY[mat].label} off all its colors")
            remove.clicked.connect(lambda: self._remove(mat))
            row.addWidget(remove)
        done = QPushButton("Done")
        done.clicked.connect(self.assign_menu.close)
        row.addWidget(done)
        bl.addLayout(row)
        act = QWidgetAction(self.assign_menu)
        act.setDefaultWidget(box)
        self.assign_menu.addAction(act)

    def _remove(self, material: str) -> None:
        """Explicitly take a preset off all of its colors (the only way colors are released)."""
        self.assigned.pop(material, None)
        if self.active == material:
            self.active = self.base
        self.assign_menu.close()
        self._refresh()
        self.changed.emit()

    def _toggle_color(self, material: str, swatch: int, on: bool) -> None:
        cols = self.assigned.setdefault(material, [])
        if on and swatch not in cols:
            cols.append(swatch)
        elif not on and swatch in cols:
            cols.remove(swatch)
        if cols:
            if self.base == material:
                self.base = None            # it now covers only its colors
        else:
            del self.assigned[material]
            if self.base is None:
                self.base = material        # no colors left: back to covering everything
        self._refresh()
        self.changed.emit()

    # ------------------------------------------------------------- look
    def _refresh(self) -> None:
        full = bool(self.swatches) and not self._free_colors()
        names = {s.index: s.hex for s in self.swatches}
        for key, cb in self.boxes.items():
            cb.blockSignals(True)
            if key in self.assigned:
                cb.setChecked(True)
                cb.setEnabled(True)
                editing = key == self.active
                cb.setStyleSheet("QCheckBox { background: %s; border: 1px solid %s; border-radius: 4px;"
                                 " padding: 1px 4px; }" % (("#2b5a45", "#ffffff") if editing else
                                                          ("#24473a", "#24473a")))
                cols = ", ".join(names.get(c, f"#{c}") for c in self.assigned[key])
                cb.setToolTip(f"{BY_KEY[key].label} is assigned to: {cols}\n"
                              + ("“Assign to colors” is editing this preset." if editing else
                                 "Click to edit its colors with “Assign to colors”."))
            else:
                cb.setChecked(key == self.base)
                cb.setEnabled(not full)
                cb.setStyleSheet("")
                cb.setToolTip(BY_KEY[key].description if not full else
                              "Every color already has a material. Click a highlighted preset and use Remove in its dropdown to free colors.")
            cb.blockSignals(False)
        can_assign = self.active is not None and bool(self.swatches) and (
            bool(self._free_colors()) or self.active in self.assigned)
        self.assign_btn.setEnabled(can_assign)
        self.assign_btn.setToolTip(f"Choose which colors are {BY_KEY[self.active].label}"
                                   if can_assign else "Tick a material first (with a texture loaded)")


class OutputOptionsBox(QGroupBox):
    changed = Signal()

    def __init__(self, settings, parent=None):
        super().__init__("Output", parent)
        form = QFormLayout(self)
        self.mode = QComboBox()
        self.mode.addItem("MERS (Vibrant Visuals)", "mers")
        self.mode.addItem("MER (RTX compatible)", "mer")
        self.mode.setToolTip("MERS adds the subsurface channel used by Vibrant Visuals.\n"
                             "MER is the 3-channel format that also works with RTX ray tracing.")
        self.fmt = QComboBox()
        self.fmt.addItem("TGA (like vanilla)", "tga")
        self.fmt.addItem("PNG", "png")
        self.noise = QCheckBox("Roughness grain where vanilla uses it")
        self.edges = QCheckBox("Learned edge/crevice adjustments")
        form.addRow("Format", self.mode)
        form.addRow("File type", self.fmt)
        form.addRow(self.noise)
        form.addRow(self.edges)
        self.mode.setCurrentIndex(max(0, self.mode.findData(settings.mode)))
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(settings.image_format)))
        self.noise.setChecked(settings.noise)
        self.edges.setChecked(settings.edge_effects)
        for w in (self.mode, self.fmt):
            w.currentIndexChanged.connect(self.changed)
        for w in (self.noise, self.edges):
            w.toggled.connect(self.changed)

    def options(self):
        from ..pipeline import OutputOptions
        return OutputOptions(self.mode.currentData(), self.fmt.currentData(),
                             self.noise.isChecked(), self.edges.isChecked())

    def store(self, settings) -> None:
        o = self.options()
        settings.mode, settings.image_format = o.mode, o.image_format
        settings.noise, settings.edge_effects = o.noise, o.edge_effects


_UNITS = "255"   # "255" (0-255) or "%" (0-100); set from the program settings


def set_value_units(units: str) -> None:
    global _UNITS
    _UNITS = "%" if units == "%" else "255"


def to_display(v: float) -> int:
    """Stored 0-255 value -> what the user sees (0-255 or 0-100)."""
    return int(round(v * 100 / 255)) if _UNITS == "%" else int(round(v))


def from_display(d: float) -> int:
    return int(round(d * 255 / 100)) if _UNITS == "%" else int(round(d))


def fmt_value(v: float) -> str:
    return f"{to_display(v)}%" if _UNITS == "%" else f"{to_display(v)}"


def channel_tip(channel: str) -> str:
    f = fmt_value
    return {"M": f"{f(0)} = not metal, {f(255)} = fully metal (vanilla metals use about {f(128)}-{f(191)}).",
            "E": f"{f(0)} = no glow. Vanilla: glowstone ~{f(40)}, torch flames ~{f(40)}-{f(60)}, "
                 f"redstone lamp {f(255)}.",
            "R": f"{f(0)} = mirror-smooth, {f(255)} = fully matte (vanilla stone ~{f(150)}-{f(230)}).",
            "S": f"{f(0)} = none, {f(255)} = strong (vanilla leaves ~{f(100)}-{f(250)}, gems/wool {f(255)})."}[channel]


class _ValueRow(QWidget):
    """Slider + editable number field + reset button.

    Values are always emitted on the 0-255 scale; the slider and field show either
    0-255 or 0-100 % depending on the program setting.
    """

    valueChanged = Signal(int)
    resetClicked = Signal()

    def __init__(self, label: str, reset_text: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        row.addWidget(self.label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip(tooltip)
        self.slider.setMinimumWidth(80)
        self.slider.valueChanged.connect(self._moved)
        row.addWidget(self.slider, 1)
        self.field = QLineEdit()
        self.field.setFixedWidth(46)
        self.field.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.field.editingFinished.connect(self._typed)
        row.addWidget(self.field)
        self.unit = QLabel("")
        self.unit.setObjectName("Hint")
        row.addWidget(self.unit)
        self.reset = QPushButton(reset_text)
        self.reset.setMinimumWidth(64)
        self.reset.clicked.connect(self.resetClicked)
        row.addWidget(self.reset)
        self._manual = False
        self.apply_units()

    def apply_units(self) -> None:
        top = 100 if _UNITS == "%" else 255
        self.slider.blockSignals(True)
        self.slider.setRange(0, top)
        self.slider.blockSignals(False)
        self.field.setValidator(QIntValidator(0, top, self.field))   # numbers only

    def _set_unit_text(self) -> None:
        unit = "%" if _UNITS == "%" else "/ 255"
        self.unit.setText(unit if self._manual else f"{unit} (learned)")

    def show(self, value: float, manual: bool) -> None:  # noqa: A003 - mirrors QWidget naming
        self._manual = manual
        d = to_display(value)
        self.slider.blockSignals(True)
        self.slider.setValue(d)
        self.slider.blockSignals(False)
        self.field.setText(str(d))
        self._set_unit_text()

    def _moved(self, d: int) -> None:
        self._manual = True
        self.field.setText(str(d))
        self._set_unit_text()
        self.valueChanged.emit(from_display(d))

    def _typed(self) -> None:
        text = self.field.text().strip()
        if not text:
            return
        d = max(0, min(self.slider.maximum(), int(text)))
        self.field.setText(str(d))
        if d != self.slider.value():
            self.slider.setValue(d)       # emits through _moved
        elif not self._manual:
            self._moved(d)                # typing the learned value makes it manual
        # else: focus left the field without a change


class ChannelEditor(QGroupBox):
    """Edit one channel of the MERS map: per detected color, or overall.

    Click a color to select it; a slider for that color (with an Auto reset)
    appears under the swatches. Colors with a manual value are not changed by
    the overall slider.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("", parent)
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        self.colors_label = QLabel("Colors: click one to set its value")
        head.addWidget(self.colors_label, 1)
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setToolTip("Reset this channel: learned values for every color and overall")
        self.auto_btn.clicked.connect(self._reset_channel)
        head.addWidget(self.auto_btn)
        lay.addLayout(head)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(SWATCH_GAP)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidget(self.grid_host)
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay.addWidget(self.grid_scroll)
        self.see_all = QPushButton("")
        self.see_all.clicked.connect(self._toggle_all)
        lay.addWidget(self.see_all)
        self._expanded = False
        # Slider for the selected color: shown under the swatches, above Overall.
        self.color_row = _ValueRow("Color", "Auto")
        self.color_row.valueChanged.connect(self._color_moved)
        self.color_row.resetClicked.connect(self._color_reset)
        self.color_row.setVisible(False)
        lay.addWidget(self.color_row)
        self.mode_label = QLabel("")
        self.mode_label.setObjectName("Hint")
        self.mode_label.setWordWrap(True)
        lay.addWidget(self.mode_label)
        self.overall = _ValueRow("Overall", "Learned")
        self.overall.valueChanged.connect(self._overall_moved)
        self.overall.resetClicked.connect(self._overall_reset)
        lay.addWidget(self.overall)

        self.buttons: list[QToolButton] = []
        self.edits = None                     # TextureEdits being edited
        self.channel: str | None = None        # nothing selected until a preview is clicked
        self.swatches = []
        self.labels: np.ndarray | None = None
        self.auto_mers: np.ndarray | None = None
        self.opaque: np.ndarray | None = None
        self._selected: int | None = None     # swatch index being edited
        self.setVisible(False)

    # ------------------------------------------------------------ context
    def set_context(self, edits, swatches, labels, auto_mers, opaque) -> None:
        """Point the editor at a texture: its edits, palette and automatic (learned) result."""
        if edits is not self.edits:
            self._selected = None
            self._expanded = False          # a new texture starts with the first 15 colors
        self.edits, self.swatches, self.labels = edits, swatches, labels
        self.auto_mers, self.opaque = auto_mers, opaque
        self._rebuild()

    def set_channel(self, channel: str | None) -> None:
        if channel != self.channel:
            self._selected = None
        self.channel = channel
        self._rebuild()

    def apply_units(self) -> None:
        """The 0-255 / % setting changed."""
        self.overall.apply_units()
        self.color_row.apply_units()
        self._rebuild()

    @property
    def _edit(self):
        return self.edits.channels[self.channel]

    def _ci(self) -> int:
        return CHANNELS.index(self.channel)

    def _learned_color(self, swatch: int) -> float:
        m = (self.labels == swatch) & self.opaque
        return float(self.auto_mers[..., self._ci()][m].mean()) if m.any() else 0.0

    def _learned_overall(self) -> float:
        manual = np.isin(self.labels, list(self._edit.colors)) if self._edit.colors else False
        rest = self.opaque & ~manual
        return float(self.auto_mers[..., self._ci()][rest].max()) if rest.any() else 0.0

    # -------------------------------------------------------------- build
    def _rebuild(self) -> None:
        self.setVisible(self.edits is not None and self.channel is not None)
        if self.edits is None or self.channel is None:
            return
        self.setTitle(f"{CHANNEL_NAMES[self.channel]} for this texture")
        for b in self.buttons:
            b.deleteLater()
        self.buttons = []
        edit = self._edit
        ordered = sorted_by_hue(self.swatches)
        shown = ordered if self._expanded else ordered[:COLLAPSED_SWATCHES]
        if not self._expanded and self._selected is not None and all(s.index != self._selected for s in shown):
            shown = shown + [s for s in ordered if s.index == self._selected]
        for i, sw in enumerate(shown):
            b = QToolButton()
            b.setProperty("swatch", sw.index)
            b.setFixedSize(40, 40)
            manual = sw.index in edit.colors
            value = edit.colors[sw.index] if manual else self._learned_color(sw.index)
            b.setText(fmt_value(edit.colors[sw.index]) if manual else "")
            text_col = "#000000" if sum(sw.rgb) > 380 else "#ffffff"
            border = "1px solid #ffffff" if sw.index == self._selected else "1px solid #3c414c"
            b.setStyleSheet(
                f"QToolButton {{ background: {sw.hex}; color: {text_col}; font-weight: 700;"
                f" border: {border}; border-radius: 6px; }}"
                f"QToolButton:hover {{ border: 1px solid #9aa3b2; }}")
            b.setToolTip(f"{sw.hex}{' (similar colors)' if sw.grouped else ''} · {share_text(sw.share)} of the texture\n"
                         f"{CHANNEL_NAMES[self.channel]}: {fmt_value(value)}"
                         f"{' (set)' if manual else ' (learned)'}")
            b.clicked.connect(lambda _on=False, s=sw.index: self._select(s))
            self.grid.addWidget(b, i // 5, i % 5)
            self.buttons.append(b)
        rows = (len(shown) + 4) // 5
        visible_rows = min(rows, 5) if self._expanded else rows   # expanded: scroll past 5 rows
        self.grid_scroll.setFixedHeight(max(visible_rows, 1) * (40 + SWATCH_GAP))
        self.grid_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded if self._expanded else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        extra = len(self.swatches) - COLLAPSED_SWATCHES
        self.see_all.setVisible(extra > 0)
        self.see_all.setText("Show fewer" if self._expanded else f"See all {len(self.swatches)} colors")
        self._show_color_row()
        self.overall.slider.setToolTip(channel_tip(self.channel))
        if edit.overall is not None:
            self.overall.show(edit.overall, True)
        else:
            self.overall.show(self._learned_overall(), False)
        self._update_mode_label()

    def _update_mode_label(self) -> None:
        edit = self._edit
        n = len(edit.colors)
        parts = [f"{n} color{'s' if n != 1 else ''} set manually (not affected by Overall)" if n
                 else "All colors automatic"]
        parts.append(f"overall {fmt_value(edit.overall)}" if edit.overall is not None else "overall learned")
        self.mode_label.setText(" · ".join(parts))

    def _show_color_row(self) -> None:
        sw = next((s for s in self.swatches if s.index == self._selected), None)
        self.color_row.setVisible(sw is not None)
        if sw is None:
            return
        edit = self._edit
        manual = sw.index in edit.colors
        self.color_row.label.setText(f"<span style='color:{sw.hex}'>■</span> Color")
        self.color_row.slider.setToolTip(channel_tip(self.channel))
        self.color_row.show(edit.colors[sw.index] if manual else self._learned_color(sw.index), manual)

    # ----------------------------------------------------------- handlers
    def _toggle_all(self) -> None:
        self._expanded = not self._expanded
        self._rebuild()

    def _select(self, swatch: int) -> None:
        self._selected = None if self._selected == swatch else swatch
        self._rebuild()

    def _color_moved(self, v: int) -> None:
        if self._selected is None:
            return
        self._edit.colors[self._selected] = v
        # Update just the swatch label and summary; rebuilding would restart the slider drag.
        for b in self.buttons:
            if b.property("swatch") == self._selected:
                b.setText(fmt_value(v))
        self._update_mode_label()
        self.changed.emit()

    def _color_reset(self) -> None:
        if self._selected is not None:
            self._edit.colors.pop(self._selected, None)
            self._rebuild()
            self.changed.emit()

    def _overall_moved(self, v: int) -> None:
        self._edit.overall = v
        self._rebuild()
        self.changed.emit()

    def _overall_reset(self) -> None:
        self._edit.overall = None
        self._rebuild()
        self.changed.emit()

    def _reset_channel(self) -> None:
        self._edit.overall = None
        self._edit.colors.clear()
        self._rebuild()
        self.changed.emit()
