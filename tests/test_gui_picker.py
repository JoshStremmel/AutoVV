"""Material picker behavior (needs Qt; runs headless)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from autovv.gui.widgets import MaterialPicker  # noqa: E402
from autovv.palette import detect_palette, sorted_by_hue  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _picker():
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[:4, :4, :3] = (200, 200, 205)       # gray  -> metal
    rgba[:4, 4:, :3] = (150, 100, 50)        # brown
    rgba[4:, :, :3] = (60, 140, 40)          # green
    _, swatches = detect_palette(rgba, "t")
    pk = MaterialPicker()
    pk.set_palette(swatches)
    gray = next(s.index for s in swatches if s.rgb == (200, 200, 205))
    return pk, swatches, gray


def _rows(pk):
    pk._fill_menu()
    return pk.assign_menu.actions()[0].defaultWidget().findChildren(QtWidgets.QCheckBox)


def test_new_preset_never_takes_over_assigned_colors(app):
    pk, swatches, gray = _picker()
    pk.boxes["metal"].click()
    order = [s.index for s in sorted_by_hue(swatches)]
    _rows(pk)[order.index(gray)].setChecked(True)
    assert pk.assignment() == {"metal": [gray]}

    for key in ("wood", "stone"):                      # tick other presets one after another
        pk.boxes[key].click()
        assert pk.assignment() == {"metal": [gray]}   # metal keeps its color
        assert pk.boxes["metal"].isChecked()

    row = _rows(pk)[order.index(gray)]                 # stone's dropdown
    assert not row.isChecked() and not row.isEnabled() and "Metal" in row.text()


def test_clicking_highlighted_preset_keeps_its_colors(app):
    pk, swatches, gray = _picker()
    pk.boxes["metal"].click()
    order = [s.index for s in sorted_by_hue(swatches)]
    _rows(pk)[order.index(gray)].setChecked(True)
    pk.boxes["wood"].click()
    pk.boxes["metal"].click()                          # select Metal again to edit it
    assert pk.assignment() == {"metal": [gray]} and pk.boxes["metal"].isChecked()
    assert pk.active == "metal" and pk.base == "wood"
    pk._remove("metal")                                # the explicit way to free colors
    assert pk.assignment() == {} and not pk.boxes["metal"].isChecked()
