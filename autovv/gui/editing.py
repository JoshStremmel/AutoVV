"""Preview + channel editing shared by the Single File and Load Pack tabs."""
from __future__ import annotations

import copy

from ..edits import CHANNELS, TextureEdits, apply_edits
from ..palette import detect_palette
from ..pipeline import generate_for
from .widgets import CHANNEL_HINTS


class EditSession:
    """Keeps the automatic (learned) result of the current texture and applies edits on top.

    Regenerating is only needed when the texture, material or output options change;
    edits from the channel editor are re-applied to the cached result instantly.
    """

    def __init__(self, strip, editor, hint_label=None):
        self.strip, self.editor, self.hint = strip, editor, hint_label
        self.rgba = self.auto = self.labels = self.edits = None
        self.swatches = []
        self.mode = "mers"
        strip.channelSelected.connect(lambda i: self.select_channel(CHANNELS[i]))
        self.select_channel(editor.channel)

    def select_channel(self, channel: str | None) -> None:
        self.editor.set_channel(channel)
        self.strip.set_selected(CHANNELS.index(channel) if channel else None)
        if self.hint is not None:
            self.hint.setText(CHANNEL_HINTS[channel] if channel else
                              "Click Metalness, Emissive, Roughness or Subsurface to edit it.")

    def show(self, kb, rgba, name, folder, decision, options, edits: TextureEdits,
             assignment: dict[str, list[int]] | None = None):
        self.rgba, self.edits, self.mode = rgba, edits, options.mode
        self.auto = generate_for(kb, rgba, name, folder, decision, options, assignment=assignment)
        self.labels, swatches = detect_palette(rgba, name)
        self.swatches = swatches
        channel = self.editor.channel
        if self.mode == "mer" and channel == "S":
            channel = None                 # subsurface isn't written in MER mode
        self.select_channel(channel)
        self.editor.set_context(edits, swatches, self.labels, self.auto.mers, rgba[..., 3] > 0)
        return self.reapply()

    def reapply(self):
        """Result with the current edits applied (also refreshes the previews)."""
        res = self.auto
        if self.edits is not None and self.edits.active:
            res = copy.copy(self.auto)
            res.mers = apply_edits(self.auto.mers, self.rgba[..., 3] > 0, self.labels, self.edits)
            res.notes = list(self.auto.notes) + self.edits.describe()
        self.strip.show_result(self.rgba, res.mers, self.mode)
        return res
