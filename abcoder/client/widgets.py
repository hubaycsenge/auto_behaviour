"""Small shared widgets and helpers for the ABC client."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def human_duration(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "-"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class Table(QTableWidget):
    """A QTableWidget with the settings every table in this app wants."""

    def __init__(self, headers: Sequence[str], parent: QWidget | None = None,
                 stretch_column: int | None = None) -> None:
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(list(headers))
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if stretch_column is not None and 0 <= stretch_column < len(headers):
            header.setSectionResizeMode(stretch_column, QHeaderView.ResizeMode.Stretch)

    def set_row_text(self, row: int, values: Sequence[str], editable: Iterable[int] = ()) -> None:
        editable_set = set(editable)
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if col not in editable_set:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, col, item)

    def append_row(self, values: Sequence[str], editable: Iterable[int] = ()) -> int:
        row = self.rowCount()
        self.insertRow(row)
        self.set_row_text(row, values, editable)
        return row

    def clear_rows(self) -> None:
        self.setRowCount(0)

    def selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.selectedIndexes()})

    def column_text(self, row: int, col: int) -> str:
        item = self.item(row, col)
        return item.text() if item else ""


class Banner(QLabel):
    """A one-line status strip that colours itself by severity."""

    LEVELS = {
        "info": ("#e8eef7", "#1c3d5a"),
        "good": ("#e6f4ea", "#12502a"),
        "warn": ("#fdf3e0", "#6b4a06"),
        "error": ("#fdeaea", "#7a1c1c"),
    }

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.set_level("info")

    def set_level(self, level: str) -> None:
        background, foreground = self.LEVELS.get(level, self.LEVELS["info"])
        self.setStyleSheet(
            f"QLabel {{ background: {background}; color: {foreground};"
            f" padding: 7px 10px; border-radius: 4px; }}"
        )

    def show_message(self, text: str, level: str = "info") -> None:
        self.set_level(level)
        self.setText(text)


class ButtonRow(QWidget):
    """A right-aligned row of buttons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addStretch(1)

    def add(self, text: str, slot=None, *, primary: bool = False,
            enabled: bool = True, tooltip: str = "") -> QPushButton:
        button = QPushButton(text, self)
        if slot is not None:
            button.clicked.connect(slot)
        button.setEnabled(enabled)
        if tooltip:
            button.setToolTip(tooltip)
        if primary:
            button.setDefault(True)
        self._layout.addWidget(button)
        return button

    def add_left(self, text: str, slot=None, *, tooltip: str = "") -> QPushButton:
        button = QPushButton(text, self)
        if slot is not None:
            button.clicked.connect(slot)
        if tooltip:
            button.setToolTip(tooltip)
        self._layout.insertWidget(0, button)
        return button


def section(title: str, *widgets: QWidget) -> QWidget:
    """A titled vertical stack."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel(f"<b>{title}</b>")
    layout.addWidget(label)
    for w in widgets:
        layout.addWidget(w)
    return container
