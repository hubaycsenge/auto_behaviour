"""Tab 1 -- the project and its videos.

The observation-ID rule is the thing this tab exists to make visible: the file
name without its extension *is* the observation ID, and two files that reduce
to the same ID would silently overwrite each other. Rather than resolve that
quietly, the table flags the collision and excludes both until the user renames
one.
"""

from __future__ import annotations

import pathlib

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets import Banner, Table, human_bytes, human_duration


class SourcePanel(QWidget):
    """Project metadata, the source folder, and the resulting observation list."""

    scan_requested = Signal(str, bool)
    state_changed = Signal()

    COLUMNS = ("Use", "Observation ID", "File", "Duration", "Size", "Note")

    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        # -- project ------------------------------------------------------
        project_box = QGroupBox("Project")
        form = QFormLayout(project_box)
        self.name_edit = QLineEdit(self.controller.state.project_name)
        self.name_edit.textChanged.connect(self._on_name_changed)
        form.addRow("Project name", self.name_edit)

        self.description_edit = QPlainTextEdit()
        self.description_edit.setPlaceholderText(
            "Optional. Written into the BORIS project description, above the "
            "note recording which engines produced the coding.")
        self.description_edit.setMaximumHeight(60)
        self.description_edit.textChanged.connect(self._on_description_changed)
        form.addRow("Description", self.description_edit)

        out_row = QWidget()
        out_layout = QHBoxLayout(out_row)
        out_layout.setContentsMargins(0, 0, 0, 0)
        self.project_path_edit = QLineEdit()
        self.project_path_edit.setPlaceholderText(
            "Where to save the .boris file -- keep it beside the videos so the "
            "paths inside stay relative")
        self.project_path_edit.textChanged.connect(self._on_project_path_changed)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._pick_project_path)
        out_layout.addWidget(self.project_path_edit, 1)
        out_layout.addWidget(browse)
        form.addRow("Save project as", out_row)
        layout.addWidget(project_box)

        # -- source folder ------------------------------------------------
        source_box = QGroupBox("Source folder")
        source_layout = QVBoxLayout(source_box)

        picker = QWidget()
        picker_layout = QHBoxLayout(picker)
        picker_layout.setContentsMargins(0, 0, 0, 0)
        self.folder_edit = QLineEdit(self.controller.state.source_dir)
        self.folder_edit.setPlaceholderText("Folder containing the videos to code")
        browse_folder = QPushButton("Choose folder...")
        browse_folder.clicked.connect(self._pick_folder)
        self.rescan_button = QPushButton("Rescan")
        self.rescan_button.clicked.connect(self._rescan)
        picker_layout.addWidget(self.folder_edit, 1)
        picker_layout.addWidget(browse_folder)
        picker_layout.addWidget(self.rescan_button)
        source_layout.addWidget(picker)

        self.recursive_box = QCheckBox("Include sub-folders")
        self.recursive_box.setToolTip(
            "Off by default: a recursive scan can find two files with the same "
            "name in different folders, and they would want the same observation ID.")
        self.recursive_box.toggled.connect(self._on_recursive_toggled)
        source_layout.addWidget(self.recursive_box)

        self.banner = Banner(
            "Choose the folder holding your videos. Each file becomes one "
            "observation, named after the file without its extension.")
        source_layout.addWidget(self.banner)
        layout.addWidget(source_box)

        # -- video table --------------------------------------------------
        self.table = Table(self.COLUMNS, stretch_column=2)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all(True))
        select_none = QPushButton("Select none")
        select_none.clicked.connect(lambda: self._set_all(False))
        controls_layout.addWidget(select_all)
        controls_layout.addWidget(select_none)
        controls_layout.addStretch(1)
        self.summary_label = QLabel("")
        controls_layout.addWidget(self.summary_label)
        layout.addWidget(controls)

    # -- state ------------------------------------------------------------
    def _on_name_changed(self, text: str) -> None:
        self.controller.state.project_name = text
        self.state_changed.emit()

    def _on_description_changed(self) -> None:
        self.controller.state.project_description = self.description_edit.toPlainText()

    def _on_project_path_changed(self, text: str) -> None:
        self.controller.state.project_path = text
        self.state_changed.emit()

    def _on_recursive_toggled(self, checked: bool) -> None:
        self.controller.state.recursive = checked

    def _pick_folder(self) -> None:
        start = self.folder_edit.text() or self.controller.config["client"].get(
            "last_source_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Choose the video folder", start)
        if folder:
            self.folder_edit.setText(folder)
            self._rescan()

    def _pick_project_path(self) -> None:
        # Default beside the videos: that is what keeps the stored media paths
        # relative, so the project and the folder can be moved together.
        default_dir = self.folder_edit.text() or str(pathlib.Path.home())
        suggestion = str(pathlib.Path(default_dir) / f"{self.name_edit.text() or 'abc'}.boris")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the BORIS project as", suggestion, "BORIS project (*.boris)")
        if path:
            if not path.endswith(".boris"):
                path += ".boris"
            self.project_path_edit.setText(path)

    def _rescan(self) -> None:
        folder = self.folder_edit.text().strip()
        if folder:
            self.scan_requested.emit(folder, self.recursive_box.isChecked())

    def _set_all(self, value: bool) -> None:
        for entry in self.controller.state.videos:
            entry.include = value and not entry.problem
        self.refresh()

    def _on_item_changed(self, item) -> None:
        if item.column() != 0:
            return
        row = item.row()
        videos = self.controller.state.videos
        if 0 <= row < len(videos):
            videos[row].include = item.checkState() == Qt.CheckState.Checked
            self._update_summary()
            self.state_changed.emit()

    # -- display ----------------------------------------------------------
    def refresh(self) -> None:
        videos = self.controller.state.videos
        self.folder_edit.setText(self.controller.state.source_dir)
        self.table.blockSignals(True)
        self.table.clear_rows()
        for entry in videos:
            row = self.table.append_row([
                "", entry.observation_id, entry.path.name,
                human_duration(entry.duration), human_bytes(entry.size_bytes),
                entry.problem,
            ])
            check = self.table.item(row, 0)
            check.setFlags((check.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                           & ~Qt.ItemFlag.ItemIsEditable)
            check.setCheckState(Qt.CheckState.Checked if entry.include
                                else Qt.CheckState.Unchecked)
            if entry.problem:
                for col in range(self.table.columnCount()):
                    self.table.item(row, col).setForeground(Qt.GlobalColor.darkRed)
        self.table.blockSignals(False)
        self._update_summary()

    def _update_summary(self) -> None:
        videos = self.controller.state.videos
        selected = self.controller.state.selected_videos
        total_bytes = sum(v.size_bytes for v in selected)
        total_seconds = sum(v.duration for v in selected)
        self.summary_label.setText(
            f"{len(selected)} of {len(videos)} selected  ·  "
            f"{human_duration(total_seconds)}  ·  {human_bytes(total_bytes)}")

        problems = [v for v in videos if v.problem]
        if not videos:
            self.banner.show_message(
                "Choose the folder holding your videos. Each file becomes one "
                "observation, named after the file without its extension.", "info")
        elif problems:
            self.banner.show_message(
                f"{len(problems)} file(s) cannot be used yet -- see the Note column. "
                f"Two files that share a name (even with different extensions) would "
                f"claim the same observation ID.", "warn")
        else:
            self.banner.show_message(
                f"{len(videos)} video(s) found. Observation IDs come from the file "
                f"names: {', '.join(v.observation_id for v in videos[:3])}"
                + (", ..." if len(videos) > 3 else ""), "good")
