"""Tab 3 -- which engine codes what.

Two decisions live here. Which engines to run, with the hardware they need made
explicit so a choice that cannot run on this cluster is visibly greyed out
rather than failing forty minutes into a queue. And, when more than one engine
runs, which of them is authoritative for each behaviour -- the hybrid
arrangement where the audio engine owns the vocalisations, the pose engine owns
the proxemics, and the video-LLM owns everything that is left.
"""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..widgets import Banner, ButtonRow, Table

#: The engine an unowned behaviour falls through to, shown in the ownership table.
ANY_ENGINE = "(any engine that finds it)"


class EnginePanel(QWidget):
    """Pick engines, edit their options, and assign behaviour ownership."""

    state_changed = Signal()

    OWNERSHIP_COLUMNS = ("Behaviour", "Type", "Coded by")

    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.catalogue: dict = {}
        self._loading = False
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self.banner = Banner("Connect to the server to see which engines it can run.")
        layout.addWidget(self.banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -- engine list --------------------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("<b>Engines</b>"))
        self.engine_list = QListWidget()
        self.engine_list.currentItemChanged.connect(self._on_selection_changed)
        self.engine_list.itemChanged.connect(self._on_engine_toggled)
        left_layout.addWidget(self.engine_list, 1)
        splitter.addWidget(left)

        # -- details ------------------------------------------------------
        right = QTabWidget()

        about = QWidget()
        about_layout = QVBoxLayout(about)
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        about_layout.addWidget(self.description_label)
        about_layout.addStretch(1)
        right.addTab(about, "About")

        options = QWidget()
        options_layout = QVBoxLayout(options)
        options_layout.addWidget(QLabel(
            "Engine options, as JSON. These are handed to the engine unchanged."))
        self.options_edit = QPlainTextEdit()
        self.options_edit.setPlaceholderText("{}")
        options_layout.addWidget(self.options_edit, 1)
        option_buttons = ButtonRow()
        option_buttons.add("Reset to defaults", self._reset_options)
        option_buttons.add("Apply", self._apply_options, primary=True)
        options_layout.addWidget(option_buttons)
        right.addTab(options, "Options")

        resources = QWidget()
        resources_layout = QVBoxLayout(resources)
        resources_layout.addWidget(QLabel(
            "SLURM request for this engine's array job, as JSON. One array task "
            "codes one video by default."))
        self.slurm_edit = QPlainTextEdit()
        resources_layout.addWidget(self.slurm_edit, 1)
        slurm_buttons = ButtonRow()
        slurm_buttons.add("Reset to defaults", self._reset_slurm)
        slurm_buttons.add("Apply", self._apply_slurm, primary=True)
        resources_layout.addWidget(slurm_buttons)
        right.addTab(resources, "Resources")

        splitter.addWidget(right)
        splitter.setSizes([260, 560])
        layout.addWidget(splitter, 1)

        # -- ownership ----------------------------------------------------
        ownership_box = QGroupBox(
            "Which engine is authoritative for each behaviour")
        ownership_layout = QVBoxLayout(ownership_box)
        self.ownership_table = Table(self.OWNERSHIP_COLUMNS, stretch_column=0)
        ownership_layout.addWidget(self.ownership_table)
        ownership_buttons = ButtonRow()
        ownership_buttons.add_left("Suggest", self._suggest_ownership,
                                   tooltip="Assign vocalisations to the audio engine and "
                                           "proxemics to the pose engine, from the wording "
                                           "of your ethogram.")
        ownership_buttons.add_left("Clear", self._clear_ownership)
        ownership_layout.addWidget(ownership_buttons)
        layout.addWidget(ownership_box)

    # -- catalogue ---------------------------------------------------------
    def set_catalogue(self, catalogue: dict) -> None:
        self.catalogue = catalogue
        self.controller.set_engine_defaults(catalogue)
        self._loading = True
        try:
            self.engine_list.clear()
            for name, entry in catalogue.items():
                item = QListWidgetItem(entry.get("label", name))
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                spec = self.controller.state.engines.get(name)
                available = bool(entry.get("available"))
                item.setCheckState(
                    Qt.CheckState.Checked if (spec and spec.enabled) else Qt.CheckState.Unchecked)
                if not available:
                    item.setForeground(Qt.GlobalColor.gray)
                    reason = entry.get("unavailable_reason") or "not installed on the server"
                    item.setToolTip(f"Not available: {reason}")
                self.engine_list.addItem(item)
        finally:
            self._loading = False
        if self.engine_list.count():
            self.engine_list.setCurrentRow(0)
        self._update_banner()

    def _current_engine(self) -> str:
        item = self.engine_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def _on_selection_changed(self, current, _previous) -> None:
        name = self._current_engine()
        if not name:
            return
        entry = self.catalogue.get(name, {})
        spec = self.controller.state.engines.get(name)
        requires = entry.get("requires", {})
        requirement_text = ""
        if requires:
            bits = []
            if requires.get("min_sm"):
                bits.append(f"compute capability {requires['min_sm'] / 10:.1f}+")
            if requires.get("min_vram_gb"):
                bits.append(f"{requires['min_vram_gb']} GB VRAM")
            requirement_text = f"<p><b>Needs:</b> {', '.join(bits)}</p>"

        status = ("<p style='color:#12502a'><b>Available on the server.</b></p>"
                  if entry.get("available") else
                  f"<p style='color:#7a1c1c'><b>Not available:</b> "
                  f"{entry.get('unavailable_reason') or 'not installed on the server'}</p>")

        self.description_label.setText(
            f"<h3>{entry.get('label', name)}</h3>{status}"
            f"<p>{entry.get('description', '')}</p>{requirement_text}")
        self.options_edit.setPlainText(
            json.dumps(spec.options if spec else entry.get("options", {}),
                       indent=2, ensure_ascii=False))
        self.slurm_edit.setPlainText(
            json.dumps(spec.slurm.to_dict() if spec else entry.get("slurm", {}),
                       indent=2, ensure_ascii=False))

    def _on_engine_toggled(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        spec = self.controller.state.engines.get(name)
        if spec is None:
            return
        wanted = item.checkState() == Qt.CheckState.Checked
        entry = self.catalogue.get(name, {})
        if wanted and not entry.get("available"):
            reply = QMessageBox.question(
                self, "Engine not available",
                f"{entry.get('label', name)} reports as unavailable on the server:\n\n"
                f"{entry.get('unavailable_reason') or 'not installed'}\n\n"
                f"Enable it anyway? The job will queue and then fail on the node.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                self._loading = True
                item.setCheckState(Qt.CheckState.Unchecked)
                self._loading = False
                return
        spec.enabled = wanted
        self._update_banner()
        self.refresh_ownership()
        self.state_changed.emit()

    # -- options -----------------------------------------------------------
    def _apply_options(self) -> None:
        name = self._current_engine()
        spec = self.controller.state.engines.get(name)
        if spec is None:
            return
        try:
            spec.options = json.loads(self.options_edit.toPlainText() or "{}")
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "Not valid JSON", str(exc))
            return
        self.state_changed.emit()

    def _reset_options(self) -> None:
        name = self._current_engine()
        spec = self.controller.state.engines.get(name)
        if spec is None:
            return
        spec.options = dict(self.catalogue.get(name, {}).get("options", {}))
        self.options_edit.setPlainText(json.dumps(spec.options, indent=2, ensure_ascii=False))
        self.state_changed.emit()

    def _apply_slurm(self) -> None:
        from ...common.jobspec import SlurmSpec

        name = self._current_engine()
        spec = self.controller.state.engines.get(name)
        if spec is None:
            return
        try:
            spec.slurm = SlurmSpec.from_dict(json.loads(self.slurm_edit.toPlainText() or "{}"))
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "Not valid JSON", str(exc))
            return
        self.state_changed.emit()

    def _reset_slurm(self) -> None:
        from ...common.jobspec import SlurmSpec

        name = self._current_engine()
        spec = self.controller.state.engines.get(name)
        if spec is None:
            return
        spec.slurm = SlurmSpec.from_dict(self.catalogue.get(name, {}).get("slurm", {}))
        self.slurm_edit.setPlainText(json.dumps(spec.slurm.to_dict(), indent=2))
        self.state_changed.emit()

    # -- ownership ---------------------------------------------------------
    def _suggest_ownership(self) -> None:
        self.controller.suggest_ownership()
        self.refresh_ownership()
        self.state_changed.emit()

    def _clear_ownership(self) -> None:
        self.controller.state.ownership = {}
        self.refresh_ownership()
        self.state_changed.emit()

    def refresh_ownership(self) -> None:
        etho = self.controller.state.ethogram
        enabled = [e.name for e in self.controller.state.enabled_engines]
        ownership = self.controller.state.ownership

        self._loading = True
        try:
            self.ownership_table.clear_rows()
            for b in etho.behaviors:
                row = self.ownership_table.append_row(
                    [b.code, "state" if b.is_state else "point", ""])
                combo = QComboBox()
                combo.addItem(ANY_ENGINE)
                combo.addItems(enabled)
                current = ownership.get(b.code)
                combo.setCurrentText(current if current in enabled else ANY_ENGINE)
                combo.currentTextChanged.connect(
                    lambda text, code=b.code: self._set_owner(code, text))
                self.ownership_table.setCellWidget(row, 2, combo)
        finally:
            self._loading = False

    def _set_owner(self, code: str, engine: str) -> None:
        if self._loading:
            return
        if engine == ANY_ENGINE:
            self.controller.state.ownership.pop(code, None)
        else:
            self.controller.state.ownership[code] = engine
        self.state_changed.emit()

    # -- display -----------------------------------------------------------
    def refresh(self) -> None:
        self.refresh_ownership()
        self._update_banner()

    def _update_banner(self) -> None:
        enabled = [e.name for e in self.controller.state.enabled_engines]
        if not self.catalogue:
            self.banner.show_message(
                "Connect to the server to see which engines it can run.", "info")
        elif not enabled:
            self.banner.show_message(
                "No engine selected. Tick at least one -- start with 'mock' to "
                "check the whole round trip before spending GPU time.", "warn")
        elif len(enabled) == 1:
            self.banner.show_message(f"Running {enabled[0]}.", "good")
        else:
            self.banner.show_message(
                f"Running {len(enabled)} engines: {', '.join(enabled)}. Use the "
                f"ownership table below to say which one wins for each behaviour; "
                f"behaviours left on '{ANY_ENGINE}' keep every engine's events, "
                f"deduplicated.", "good")
