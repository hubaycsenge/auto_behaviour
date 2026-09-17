"""Tab 4 -- submit, watch, collect.

This is also where the staging decision becomes visible. ABC has already
established, by writing a probe file and asking the server to read it back,
whether the cluster can see the video folder. The strip at the top says which
answer it got, and the "Delete videos from server" button is live only in the
case where something was actually copied -- on a shared filesystem the path on
the server *is* the researcher's original folder, and offering to delete it
would be offering to destroy their data.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..widgets import Banner, ButtonRow, Table, human_bytes


class RunPanel(QWidget):
    """Submission, progress and collection."""

    submit_requested = Signal(bool)         # dry_run
    collect_requested = Signal()
    purge_requested = Signal()
    cancel_requested = Signal()
    refresh_requested = Signal()

    OBSERVATION_COLUMNS = ("Observation", "State", "Events", "Detail")

    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._poll = QTimer(self)
        self._poll.timeout.connect(self.refresh_requested.emit)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        self.staging_banner = Banner("Not connected to the server yet.")
        layout.addWidget(self.staging_banner)

        # -- options ------------------------------------------------------
        options_box = QGroupBox("Run options")
        form = QFormLayout(options_box)

        self.per_task_spin = QSpinBox()
        self.per_task_spin.setRange(1, 200)
        self.per_task_spin.setValue(self.controller.state.per_task)
        self.per_task_spin.setToolTip(
            "Videos per SLURM array task. Raise it when the model takes longer to "
            "load than to run, or when the corpus is larger than the cluster's "
            "array limit.")
        self.per_task_spin.valueChanged.connect(
            lambda v: setattr(self.controller.state, "per_task", v))
        form.addRow("Videos per task", self.per_task_spin)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(self.controller.state.min_confidence)
        self.confidence_spin.setToolTip(
            "Events below this confidence are left out of the BORIS project. "
            "Keep it at 0 for a first pass: a low-confidence event a human can "
            "reject costs less than a missed one they never see.")
        self.confidence_spin.valueChanged.connect(
            lambda v: setattr(self.controller.state, "min_confidence", v))
        form.addRow("Minimum confidence", self.confidence_spin)

        self.evidence_box = QCheckBox(
            "Write the engine, its confidence and its reasoning into the BORIS comment")
        self.evidence_box.setChecked(self.controller.state.include_evidence)
        self.evidence_box.toggled.connect(
            lambda v: setattr(self.controller.state, "include_evidence", v))
        form.addRow("Provenance", self.evidence_box)
        layout.addWidget(options_box)

        # -- actions ------------------------------------------------------
        actions = ButtonRow()
        self.dry_run_button = actions.add_left(
            "Check without submitting", lambda: self.submit_requested.emit(True),
            tooltip="Write the sbatch scripts and validate the job, but queue nothing.")
        self.submit_button = actions.add("Submit to the cluster",
                                         lambda: self.submit_requested.emit(False),
                                         primary=True)
        self.cancel_button = actions.add("Cancel job", self.cancel_requested.emit,
                                         enabled=False)
        layout.addWidget(actions)

        # -- progress -----------------------------------------------------
        progress_box = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%  ·  %v of %m")
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("No job submitted yet.")
        progress_layout.addWidget(self.progress_label)
        layout.addWidget(progress_box)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.observation_table = Table(self.OBSERVATION_COLUMNS, stretch_column=3)
        splitter.addWidget(self.observation_table)

        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("<b>Messages</b>"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log, 1)
        splitter.addWidget(log_container)
        splitter.setSizes([520, 380])
        layout.addWidget(splitter, 1)

        # -- collect / cleanup --------------------------------------------
        finish = ButtonRow()
        self.purge_button = finish.add_left(
            "Delete videos from server", self._confirm_purge, tooltip="")
        self.purge_button.setEnabled(False)
        self.collect_button = finish.add("Save BORIS project",
                                         self.collect_requested.emit, primary=True,
                                         enabled=False)
        layout.addWidget(finish)

    # -- staging -----------------------------------------------------------
    def set_staging(self, uploads: bool, description: str) -> None:
        """Explain where the videos are and enable the delete button accordingly."""
        if uploads:
            self.staging_banner.show_message(
                f"{description}  ·  The cluster cannot see your video folder, so the "
                f"selected files will be uploaded when you submit. Use "
                f"'Delete videos from server' once you have saved the project.", "warn")
            self.purge_button.setEnabled(True)
            self.purge_button.setToolTip(
                "Removes the uploaded copies from the job directory on the server. "
                "Your originals are untouched.")
        else:
            self.staging_banner.show_message(
                f"{description}  ·  The cluster can read your video folder directly, "
                f"so nothing will be copied.", "good")
            self.purge_button.setEnabled(False)
            self.purge_button.setToolTip(
                "Disabled: nothing was uploaded. The server reads your videos where "
                "they already are, and deleting them would delete your originals.")

    # -- progress ----------------------------------------------------------
    def set_progress(self, payload: dict) -> None:
        expected = int(payload.get("expected", 0))
        done = int(payload.get("done", 0))
        failed = int(payload.get("failed", 0))
        running = int(payload.get("running", 0))
        pending = int(payload.get("pending", 0))

        self.progress_bar.setMaximum(max(1, expected))
        self.progress_bar.setValue(done + failed)

        bits = [f"{done} done"]
        if failed:
            bits.append(f"{failed} failed")
        if running:
            bits.append(f"{running} running")
        if pending:
            bits.append(f"{pending} waiting")
        self.progress_label.setText("  ·  ".join(bits))

        for message in payload.get("messages", []):
            self.append_log(message)

        rows = payload.get("observations") or []
        if rows:
            self._fill_observations(rows)

        finished = bool(payload.get("finished")) and pending == 0
        self.collect_button.setEnabled(done > 0)
        self.cancel_button.setEnabled(not finished and expected > 0)
        if finished:
            self.stop_polling()
            if done:
                self.append_log(
                    f"All tasks finished: {done} observation-engine result(s) ready. "
                    f"Save the BORIS project when you are ready.")

    def _fill_observations(self, rows: list[dict]) -> None:
        self.observation_table.clear_rows()
        for row in rows:
            engines = row.get("engines", {})
            states = [f"{name}: {info.get('state', '?')}" for name, info in engines.items()]
            events = sum(int(info.get("events", 0)) for info in engines.values())
            errors = [info.get("error", "") for info in engines.values() if info.get("error")]
            overall = ("failed" if all(i.get("state") == "failed" for i in engines.values())
                       else "done" if all(i.get("state") == "done" for i in engines.values())
                       else "running")
            index = self.observation_table.append_row([
                row.get("observation_id", ""), overall, str(events),
                errors[0] if errors else ", ".join(states),
            ])
            if overall == "failed":
                for col in range(self.observation_table.columnCount()):
                    self.observation_table.item(index, col).setForeground(
                        Qt.GlobalColor.darkRed)

    # -- polling -----------------------------------------------------------
    def start_polling(self, seconds: int) -> None:
        self._poll.start(max(3, seconds) * 1000)

    def stop_polling(self) -> None:
        self._poll.stop()

    # -- logging -----------------------------------------------------------
    def append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    # -- purge -------------------------------------------------------------
    def _confirm_purge(self) -> None:
        reply = QMessageBox.question(
            self, "Delete videos from server",
            "Delete the uploaded copies of your videos from this job's directory "
            "on the server?\n\nYour originals on this machine are not touched. "
            "The coding results stay on the server, so you can still save the "
            "BORIS project afterwards -- but re-running an engine would need the "
            "videos uploaded again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Yes:
            self.purge_requested.emit()

    def report_purge(self, payload: dict) -> None:
        note = payload.get("note")
        if note:
            self.append_log(note)
            return
        deleted = payload.get("deleted", [])
        freed = int(payload.get("bytes_freed", 0))
        self.append_log(
            f"Deleted {len(deleted)} file(s) from the server, freeing {human_bytes(freed)}.")
        self.purge_button.setEnabled(False)
