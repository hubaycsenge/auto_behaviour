"""Tab 2 -- subjects and the ethogram.

Laid out the way BORIS lays them out, because the people using this already
know that layout. The one addition is the warning strip: ABC checks for the
subject-as-trial-phase idiom and offers to convert it, since a project built
that way cannot use BORIS's own per-subject time budgets and cannot tell an
engine which actor to watch.
"""

from __future__ import annotations

import pathlib

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QInputDialog,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...common.ethogram import (  # noqa: F401 - names re-exported for the panel
    BEHAVIOR_TYPES,
    EPISODE_CATEGORY,
    POINT,
    STATE,
    Behavior,
    Subject,
)
from ..widgets import Banner, ButtonRow, Table


class EthogramPanel(QWidget):
    """Edit the coding scheme: who acts, and what counts as a behaviour."""

    state_changed = Signal()

    BEHAVIOR_COLUMNS = ("Code", "Type", "Category", "Key", "Colour", "Excluded", "Description")
    SUBJECT_COLUMNS = ("Name", "Key", "Description")

    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._loading = False
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        self.banner = Banner(
            "Import your ethogram from an existing BORIS project or spreadsheet, "
            "or build one here.")
        layout.addWidget(self.banner)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # -- subjects -----------------------------------------------------
        subject_box = QGroupBox("Subjects  ·  who the behaviour is attributed to")
        subject_layout = QVBoxLayout(subject_box)
        self.subject_table = Table(self.SUBJECT_COLUMNS, stretch_column=2)
        for col in range(len(self.SUBJECT_COLUMNS)):
            self.subject_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Stretch if col == 2
                else QHeaderView.ResizeMode.ResizeToContents)
        self.subject_table.itemChanged.connect(self._on_subject_edited)
        subject_layout.addWidget(self.subject_table)

        subject_buttons = ButtonRow()
        subject_buttons.add_left("Add subject", self._add_subject)
        subject_buttons.add_left("Remove", self._remove_subject)
        subject_layout.addWidget(subject_buttons)
        splitter.addWidget(subject_box)

        # -- behaviours ---------------------------------------------------
        behavior_box = QGroupBox("Ethogram  ·  what the subjects do")
        behavior_layout = QVBoxLayout(behavior_box)
        self.behavior_table = Table(self.BEHAVIOR_COLUMNS, stretch_column=6)
        self.behavior_table.itemChanged.connect(self._on_behavior_edited)
        behavior_layout.addWidget(self.behavior_table)

        behavior_buttons = ButtonRow()
        behavior_buttons.add_left("Import...", self._import)
        behavior_buttons.add_left("Add behaviour", self._add_behavior)
        behavior_buttons.add_left("Remove", self._remove_behavior)
        behavior_buttons.add("Mark as trial phase", self._mark_episode,
                             tooltip="Move the selected behaviours into the "
                                     f"'{EPISODE_CATEGORY}' category as state events. "
                                     "Use this for the phases of a test session.")
        behavior_layout.addWidget(behavior_buttons)
        splitter.addWidget(behavior_box)

        splitter.setSizes([180, 420])
        layout.addWidget(splitter, 1)

    # -- import -----------------------------------------------------------
    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import an ethogram", "",
            "Ethogram (*.boris *.xlsx *.csv *.tsv);;BORIS project (*.boris);;"
            "Spreadsheet (*.xlsx *.csv *.tsv);;All files (*)")
        if not path:
            return
        try:
            self.controller.load_ethogram(path)
        except Exception as exc:  # noqa: BLE001 - a bad file is the user's problem to see
            QMessageBox.critical(self, "Import failed",
                                 f"Could not read {pathlib.Path(path).name}:\n\n{exc}")
            return

        # BORIS exports subjects and behaviours as two sheets; offer the second.
        source = pathlib.Path(path)
        spreadsheet = source.suffix.lower() in (".xlsx", ".csv", ".tsv")
        if spreadsheet and not self.controller.state.ethogram.subjects:
            reply = QMessageBox.question(
                self, "Subjects",
                "That file held behaviours only. Do you have a separate subjects "
                "sheet to import as well?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                subject_path, _ = QFileDialog.getOpenFileName(
                    self, "Import subjects", str(source.parent),
                    "Spreadsheet (*.xlsx *.csv *.tsv);;All files (*)")
                if subject_path:
                    try:
                        self.controller.load_ethogram(subject_path)
                    except Exception as exc:  # noqa: BLE001
                        QMessageBox.critical(self, "Import failed", str(exc))
        self.refresh()
        self.state_changed.emit()

    # -- subject editing ---------------------------------------------------
    def _add_subject(self) -> None:
        name, ok = QInputDialog.getText(self, "Add subject", "Subject name")
        if not ok or not name.strip():
            return
        try:
            self.controller.state.ethogram.add_subject(Subject(name.strip()))
        except ValueError as exc:
            QMessageBox.warning(self, "Duplicate subject", str(exc))
            return
        self.refresh()
        self.state_changed.emit()

    def _remove_subject(self) -> None:
        rows = self.subject_table.selected_rows()
        names = [self.subject_table.column_text(r, 0) for r in rows]
        for name in names:
            self.controller.state.ethogram.remove_subject(name)
        self.refresh()
        self.state_changed.emit()

    def _on_subject_edited(self, item) -> None:
        if self._loading:
            return
        subjects = self.controller.state.ethogram.subjects
        row, col = item.row(), item.column()
        if not 0 <= row < len(subjects):
            return
        value = item.text()
        if col == 0:
            subjects[row].name = value.strip()
        elif col == 1:
            subjects[row].key = value[:1]
        else:
            subjects[row].description = value
        self._check_problems()
        self.state_changed.emit()

    # -- behaviour editing -------------------------------------------------
    def _add_behavior(self) -> None:
        code, ok = QInputDialog.getText(self, "Add behaviour", "Behaviour code")
        if not ok or not code.strip():
            return
        try:
            self.controller.state.ethogram.add(Behavior(code.strip()))
        except ValueError as exc:
            QMessageBox.warning(self, "Duplicate behaviour", str(exc))
            return
        self.refresh()
        self.state_changed.emit()

    def _remove_behavior(self) -> None:
        rows = self.behavior_table.selected_rows()
        codes = [self.behavior_table.column_text(r, 0) for r in rows]
        for code in codes:
            self.controller.state.ethogram.remove(code)
        self.refresh()
        self.state_changed.emit()

    def _mark_episode(self) -> None:
        rows = self.behavior_table.selected_rows()
        if not rows:
            QMessageBox.information(
                self, "Nothing selected",
                "Select the behaviours that mark the phases of a session first.")
            return
        codes = [self.behavior_table.column_text(r, 0) for r in rows]
        self.controller.mark_episode_behaviors(codes)
        self.refresh()
        self.state_changed.emit()

    def _on_behavior_edited(self, item) -> None:
        if self._loading:
            return
        behaviors = self.controller.state.ethogram.behaviors
        row, col = item.row(), item.column()
        if not 0 <= row < len(behaviors):
            return
        b = behaviors[row]
        value = item.text()
        if col == 0:
            b.code = value.strip()
        elif col == 1:
            b.type = value if value in BEHAVIOR_TYPES else POINT
        elif col == 2:
            b.category = value.strip()
        elif col == 3:
            b.key = value[:1]
        elif col == 4:
            b.color = value.strip()
        elif col == 5:
            b.excluded = value.strip()
        else:
            b.description = value
        self._check_problems()
        self.state_changed.emit()

    # -- display -----------------------------------------------------------
    def refresh(self) -> None:
        etho = self.controller.state.ethogram
        self._loading = True
        try:
            self.subject_table.clear_rows()
            for s in etho.subjects:
                self.subject_table.append_row([s.name, s.key, s.description],
                                              editable=(0, 1, 2))

            self.behavior_table.clear_rows()
            for i, b in enumerate(etho.behaviors):
                row = self.behavior_table.append_row(
                    [b.code, b.type, b.category, b.key, b.color, b.excluded, b.description],
                    editable=(0, 2, 3, 4, 5, 6))
                combo = QComboBox()
                combo.addItems(list(BEHAVIOR_TYPES))
                combo.setCurrentText(b.type)
                combo.currentTextChanged.connect(
                    lambda text, index=i: self._set_type(index, text))
                self.behavior_table.setCellWidget(row, 1, combo)
        finally:
            self._loading = False
        self._check_problems()

    def _set_type(self, index: int, text: str) -> None:
        if self._loading:
            return
        behaviors = self.controller.state.ethogram.behaviors
        if 0 <= index < len(behaviors):
            behaviors[index].type = text
            self.state_changed.emit()

    def _check_problems(self) -> None:
        etho = self.controller.state.ethogram
        problems = etho.validate()
        episodes = [b.code for b in etho.episode_behaviors()]

        if problems:
            self.banner.show_message(" ".join(problems[:3]), "warn")
        elif not etho.behaviors:
            self.banner.show_message(
                "Import your ethogram from an existing BORIS project or spreadsheet, "
                "or build one here.", "info")
        elif not etho.subjects:
            self.banner.show_message(
                f"{len(etho)} behaviour(s), no subjects. With no subjects declared "
                f"every event is coded against BORIS's 'No focal subject', which is "
                f"right for single-animal work. Add subjects if you need to tell "
                f"actors apart.", "info")
        else:
            detail = (f"  ·  {len(episodes)} trial phase(s) in the "
                      f"'{EPISODE_CATEGORY}' category" if episodes else "")
            self.banner.show_message(
                f"{len(etho)} behaviour(s), {len(etho.subjects)} subject(s)"
                f"{detail}.", "good")
