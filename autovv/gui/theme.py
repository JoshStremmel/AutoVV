"""Dark theme stylesheet."""
from pathlib import Path

_ASSETS = Path(__file__).with_name("assets").as_posix()

STYLESHEET = """
QCheckBox::indicator, QTreeView::indicator, QRadioButton::indicator {
    width: 14px; height: 14px; border: 1px solid #5a6070; border-radius: 3px; background: #16181d; }
QCheckBox::indicator:hover, QTreeView::indicator:hover, QRadioButton::indicator:hover { border-color: #5fd38d; }
QCheckBox::indicator:checked, QTreeView::indicator:checked {
    background: #2f7d52; border-color: #3c9a66; image: url(ASSETS/check.svg); }
QTreeView::indicator:indeterminate { background: #2a4a3a; border-color: #3c9a66; image: url(ASSETS/partial.svg); }
QRadioButton::indicator { border-radius: 8px; }
QRadioButton::indicator:checked { background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
    stop:0 #5fd38d, stop:0.45 #5fd38d, stop:0.55 #16181d, stop:1 #16181d); border-color: #3c9a66; }
QWidget { background: #1e2026; color: #d7dce5; font-size: 10pt; }
QMainWindow, QTabWidget::pane { background: #1e2026; }
QTabBar::tab { background: #262930; color: #9aa3b2; padding: 9px 20px; border: none;
               border-bottom: 2px solid transparent; font-weight: 600; }
QTabBar::tab:selected { color: #ffffff; border-bottom: 2px solid #5fd38d; background: #1e2026; }
QTabBar::tab:hover { color: #d7dce5; }
QGroupBox { border: 1px solid #343842; border-radius: 6px; margin-top: 14px; padding: 10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #aeb6c4; font-weight: 600; }
QPushButton { background: #2d313a; border: 1px solid #3c414c; border-radius: 5px; padding: 6px 12px; }
QPushButton:hover { background: #363b46; }
QPushButton:disabled { color: #6b7280; background: #25282e; }
QToolButton#AssignButton { background: #2d313a; border: 1px solid #3c414c; border-radius: 5px; padding: 5px 12px; }
QToolButton#AssignButton:hover { background: #363b46; }
QToolButton#AssignButton:disabled { color: #6b7280; background: #25282e; }
QToolButton#AssignButton::menu-indicator { subcontrol-position: right center; right: 4px; }
QPushButton#Primary { background: #2f7d52; border-color: #3c9a66; color: white; font-weight: 600; padding: 8px 14px; }
QPushButton#Primary:hover { background: #379160; }
QPushButton#Primary:disabled { background: #2a3b31; color: #7d8f84; border-color: #33463a; }
QLineEdit, QComboBox, QPlainTextEdit, QTextBrowser, QListWidget, QTreeWidget {
    background: #16181d; border: 1px solid #343842; border-radius: 5px; padding: 4px;
    selection-background-color: #2f5d8a; }
QComboBox QAbstractItemView { background: #16181d; selection-background-color: #2f5d8a; }
QTreeWidget::item { padding: 2px 0; }
QHeaderView::section { background: #262930; color: #aeb6c4; border: none; padding: 5px; font-weight: 600; }
QProgressBar { background: #16181d; border: 1px solid #343842; border-radius: 5px; text-align: center; height: 18px; }
QProgressBar::chunk { background: #2f7d52; border-radius: 4px; }
QCheckBox, QRadioButton { spacing: 6px; background: transparent; }
QCheckBox:disabled, QRadioButton:disabled { color: #6b7280; }
QCheckBox::indicator:disabled { background: #22252b; border-color: #3a3e47; }
QLabel { background: transparent; }
QLabel#Title { font-size: 12pt; font-weight: 600; color: #ffffff; }
QLabel#Hint { color: #8f98a8; }
QLabel#Detected { padding: 4px 2px; }
QFrame#DropZone { border: 2px dashed #4a5060; border-radius: 10px; background: #1a1c21; }
QFrame#DropZone[hover="true"] { border-color: #5fd38d; background: #1d2a23; }
QLabel#DropLabel { color: #9aa3b2; font-size: 11pt; }
QScrollArea { background: transparent; }
QSplitter::handle { background: #2a2d34; width: 3px; }
QStatusBar { background: #181a1f; color: #9aa3b2; }
QMenu { background: #262930; border: 1px solid #3c414c; }
QMenu::item:selected { background: #2f5d8a; }
QToolTip { background: #262930; color: #d7dce5; border: 1px solid #3c414c; }
QSlider::groove:horizontal { height: 6px; background: #16181d; border: 1px solid #343842; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #2f7d52; border-radius: 3px; }
QSlider::handle:horizontal { background: #5fd38d; width: 14px; margin: -5px 0; border-radius: 7px; }
""".replace("ASSETS", _ASSETS)
