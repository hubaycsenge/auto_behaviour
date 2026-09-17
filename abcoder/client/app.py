"""The ABC client: a Qt front end over :class:`~abcoder.client.controller.Controller`.

Four tabs in the order the work happens -- videos, ethogram, engines, run --
and a status bar that always says where the server is and whether the videos
have to travel. Every call that touches the network or the disk goes through
:mod:`abcoder.client.workers`, so the window never freezes while 3 GB uploads.
"""

from __future__ import annotations

import pathlib
import sys

from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
)

from ..common.boris import BorisProject, ethogram_to_boris_file, migrate_subjects_to_episodes
from ..common.config import load_config
from ..common.ethogram import Subject
from ..version import __version__
from .controller import Controller
from .panels import EnginePanel, EthogramPanel, RunPanel, SourcePanel
from .workers import run_async


class MainWindow(QMainWindow):
    def __init__(self, controller: Controller) -> None:
        super().__init__()
        self.controller = controller
        self.pool = QThreadPool.globalInstance()
        self._busy = False

        self.setWindowTitle(f"ABC {__version__} - Automated Behaviour Coder")
        self.resize(1180, 820)

        self.source_panel = SourcePanel(controller)
        self.ethogram_panel = EthogramPanel(controller)
        self.engine_panel = EnginePanel(controller)
        self.run_panel = RunPanel(controller)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.source_panel, "1 · Videos")
        self.tabs.addTab(self.ethogram_panel, "2 · Ethogram")
        self.tabs.addTab(self.engine_panel, "3 · Engines")
        self.tabs.addTab(self.run_panel, "4 · Run")
        self.setCentralWidget(self.tabs)

        self._build_status_bar()
        self._build_menu()
        self._wire()

        self.ethogram_panel.refresh()
        self._restore_session()
        self.connect_to_server()

    # ------------------------------------------------------------------ chrome
    def _build_status_bar(self) -> None:
        self.server_label = QLabel("Connecting...")
        self.busy_bar = QProgressBar()
        self.busy_bar.setMaximumWidth(220)
        self.busy_bar.setVisible(False)
        self.statusBar().addWidget(self.server_label, 1)
        self.statusBar().addPermanentWidget(self.busy_bar)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "&Open a BORIS project as a template...",
                     self.open_template)
        self._action(file_menu, "&Export ethogram as .boris...", self.export_ethogram)
        file_menu.addSeparator()
        self._action(file_menu, "&Convert a subjects-as-phases project...",
                     self.migrate_project)
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, QKeySequence.StandardKey.Quit)

        server_menu = self.menuBar().addMenu("&Server")
        self._action(server_menu, "&Reconnect", self.connect_to_server,
                     QKeySequence("Ctrl+R"))
        self._action(server_menu, "Re-open a &previous job...", self.attach_job)
        self._action(server_menu, "Server &details...", self.show_server_details)

        help_menu = self.menuBar().addMenu("&Help")
        self._action(help_menu, "&About ABC", self.show_about)

    def _action(self, menu, text: str, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    def _wire(self) -> None:
        self.source_panel.scan_requested.connect(self.scan_folder)
        self.source_panel.state_changed.connect(self._sync_panels)
        self.ethogram_panel.state_changed.connect(self._sync_panels)
        self.engine_panel.state_changed.connect(self._sync_panels)

        self.run_panel.submit_requested.connect(self.submit)
        self.run_panel.collect_requested.connect(self.collect)
        self.run_panel.purge_requested.connect(self.purge)
        self.run_panel.cancel_requested.connect(self.cancel)
        self.run_panel.refresh_requested.connect(self.refresh_status)

    def _restore_session(self) -> None:
        client_cfg = self.controller.config.get("client", {})
        self.controller.state.recursive = bool(client_cfg.get("recursive_scan", False))
        self.source_panel.recursive_box.setChecked(self.controller.state.recursive)
        last = client_cfg.get("last_source_dir", "")
        if last and pathlib.Path(last).is_dir():
            self.source_panel.folder_edit.setText(last)

    # ----------------------------------------------------------------- busy UI
    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        self.busy_bar.setVisible(busy)
        if busy:
            self.busy_bar.setRange(0, 0)
        if message:
            self.statusBar().showMessage(message, 0 if busy else 6000)
        elif not busy:
            self.statusBar().clearMessage()

    def _on_progress(self, message: str, fraction: float) -> None:
        self.busy_bar.setVisible(True)
        if fraction < 0:
            self.busy_bar.setRange(0, 0)
        else:
            self.busy_bar.setRange(0, 100)
            self.busy_bar.setValue(int(fraction * 100))
        self.statusBar().showMessage(message)

    def _on_error(self, message: str, detail: str) -> None:
        self._set_busy(False)
        self.run_panel.append_log(f"ERROR: {message}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Something went wrong")
        box.setText(message)
        box.setDetailedText(detail)
        box.exec()

    # ---------------------------------------------------------------- server
    def connect_to_server(self) -> None:
        self._set_busy(True, "Connecting to the server...")

        def work():
            description, info = self.controller.connect(
                self.source_panel.folder_edit.text().strip() or None)
            catalogue = self.controller.available_engines()
            return description, info, catalogue

        run_async(self.pool, work, on_done=self._server_connected,
                  on_error=self._server_failed)

    def _server_connected(self, payload) -> None:
        description, info, catalogue = payload
        self._set_busy(False)
        gpu = info.get("gpu") or {}
        gpu_text = (f"  ·  {gpu.get('name')} (cc {gpu.get('compute_capability')})"
                    if gpu else "")
        self.server_label.setText(
            f"{description}  ·  ABC {info.get('abc_version', '?')} on Python "
            f"{info.get('python', '?')}{gpu_text}")
        self.engine_panel.set_catalogue(catalogue)
        self.run_panel.set_staging(self.controller.uploads_required, description)
        self.run_panel.append_log(f"Connected: {description}")

        missing = [name for name, entry in catalogue.items() if not entry.get("available")]
        if missing:
            self.run_panel.append_log(
                "Engines not installed on the server: " + ", ".join(missing))

    def _server_failed(self, message: str, detail: str) -> None:
        self._set_busy(False)
        self.server_label.setText("Not connected")
        self.run_panel.append_log(f"Could not reach the server: {message}")
        QMessageBox.warning(
            self, "Cannot reach the server",
            f"{message}\n\nCheck the SSH settings in your ABC config, or run the "
            f"client on a machine that can submit to SLURM. You can still build "
            f"the ethogram and choose videos offline.")

    def show_server_details(self) -> None:
        import json
        box = QMessageBox(self)
        box.setWindowTitle("Server details")
        box.setText(self.controller.transport.description
                    if self.controller.transport else "Not connected")
        box.setDetailedText(json.dumps(self.controller.server_info, indent=2))
        box.exec()

    # ------------------------------------------------------------------ scan
    def scan_folder(self, folder: str, recursive: bool) -> None:
        self._set_busy(True, "Scanning...")

        def work(progress=None):
            return self.controller.scan_source(folder, recursive, progress=progress)

        run_async(self.pool, work, wants_progress=True,
                  on_progress=self._on_progress,
                  on_done=self._scan_done, on_error=self._on_error)

    def _scan_done(self, entries) -> None:
        self._set_busy(False, f"Found {len(entries)} video(s).")
        self.source_panel.refresh()
        # A folder change can flip the shared-filesystem answer, so re-ask.
        if self.controller.transport is not None:
            self.connect_to_server()
        if not self.source_panel.project_path_edit.text() and entries:
            suggested = pathlib.Path(self.controller.state.source_dir) / (
                f"{self.controller.state.project_name or 'abc'}.boris")
            self.source_panel.project_path_edit.setText(str(suggested))

    # ---------------------------------------------------------------- submit
    def submit(self, dry_run: bool) -> None:
        problems = self._preflight()
        if problems:
            QMessageBox.warning(self, "Not ready to submit",
                                "Fix these first:\n\n· " + "\n· ".join(problems))
            return

        self._set_busy(True, "Preparing the job...")

        def work(progress=None):
            job = self.controller.build_job()
            issues = job.validate()
            if issues:
                raise ValueError("; ".join(issues))
            if progress:
                progress("Writing the job manifest", 0.05)
            self.controller.write_manifest()
            uploaded = self.controller.upload(progress=progress)
            if progress:
                progress("Submitting to SLURM", 0.95)
            result = self.controller.submit(dry_run=dry_run)
            return job, uploaded, result

        run_async(self.pool, work, wants_progress=True,
                  on_progress=self._on_progress,
                  on_done=lambda payload: self._submitted(payload, dry_run),
                  on_error=self._on_error)

    def _preflight(self) -> list[str]:
        state = self.controller.state
        problems: list[str] = []
        if self.controller.transport is None:
            problems.append("Not connected to the server.")
        if not state.selected_videos:
            problems.append("No videos selected on the Videos tab.")
        if not state.ethogram.behaviors:
            problems.append("The ethogram is empty.")
        if not state.enabled_engines:
            problems.append("No engine selected on the Engines tab.")
        if not state.project_path:
            problems.append("No output path for the BORIS project.")
        problems.extend(state.ethogram.validate())
        return problems

    def _submitted(self, payload, dry_run: bool) -> None:
        job, uploaded, result = payload
        self._set_busy(False)
        if uploaded:
            from .widgets import human_bytes
            self.run_panel.append_log(f"Uploaded {human_bytes(uploaded)} to the server.")
        self.run_panel.append_log(
            f"Job {job.job_id} written to {job.job_dir}")
        for array in result.get("submitted", []):
            self.run_panel.append_log(
                f"  {array['engine']}: SLURM job {array['job_id']}, "
                f"{array['n_tasks']} array task(s)")
        if dry_run:
            self.run_panel.append_log(
                "Dry run: the sbatch scripts are written but nothing was queued. "
                f"They are in {job.job_dir}/slurm.")
            return
        self.tabs.setCurrentWidget(self.run_panel)
        self.run_panel.start_polling(
            int(self.controller.config["client"].get("poll_seconds", 15)))
        self.refresh_status()

    def cancel(self) -> None:
        if self.controller.job is None:
            return
        reply = QMessageBox.question(
            self, "Cancel job",
            "Cancel this job's queued and running SLURM tasks?\n\n"
            "Results already written stay on the server.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if reply != QMessageBox.StandardButton.Yes:
            return
        run_async(self.pool, self.controller.cancel,
                  on_done=lambda p: self.run_panel.append_log(
                      f"Cancelled SLURM job(s): {', '.join(p.get('cancelled', [])) or 'none'}"),
                  on_error=self._on_error)
        self.run_panel.stop_polling()

    # ---------------------------------------------------------------- status
    def refresh_status(self) -> None:
        if self.controller.job is None:
            return
        run_async(self.pool, self.controller.status, True,
                  on_done=self.run_panel.set_progress,
                  on_error=lambda m, d: self.run_panel.append_log(f"Status check failed: {m}"))

    # --------------------------------------------------------------- collect
    def collect(self) -> None:
        target = self.controller.state.project_path
        if not target:
            self._pick_project_target()
            target = self.controller.state.project_path
            if not target:
                return
        if pathlib.Path(target).exists():
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"{pathlib.Path(target).name} already exists. Replace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._set_busy(True, "Building the BORIS project...")
        run_async(self.pool, self.controller.collect, target, wants_progress=True,
                  on_progress=self._on_progress, on_done=self._collected,
                  on_error=self._on_error)

    def _pick_project_target(self) -> None:
        start = self.controller.state.source_dir or str(pathlib.Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the BORIS project as",
            str(pathlib.Path(start) / f"{self.controller.state.project_name}.boris"),
            "BORIS project (*.boris)")
        if path:
            if not path.endswith(".boris"):
                path += ".boris"
            self.source_panel.project_path_edit.setText(path)

    def _collected(self, report) -> None:
        self._set_busy(False)
        self.run_panel.append_log(f"Saved: {report.summary()}")
        for warning in report.warnings[:40]:
            self.run_panel.append_log(f"  {warning}")

        lines = [report.summary(), "", f"Saved to {self.controller.state.project_path}"]
        if report.missing_media:
            lines += ["",
                      f"{len(report.missing_media)} video(s) referenced by the project "
                      f"were not found at the expected path. BORIS will ask you to "
                      f"locate them when you open it."]
        if report.observations_failed:
            lines += ["", "Failed observations:"] + [
                f"  {k}: {v}" for k, v in list(report.observations_failed.items())[:10]]
        if report.observations_skipped:
            lines += ["",
                      f"{len(report.observations_skipped)} observation(s) had no result "
                      f"yet and were left out."]

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information if report.ok else QMessageBox.Icon.Warning)
        box.setWindowTitle("BORIS project saved" if report.ok else "Nothing to save")
        box.setText("\n".join(lines))
        if report.warnings:
            box.setDetailedText("\n".join(report.warnings))
        box.exec()
        self.controller.save_settings()

    # ----------------------------------------------------------------- purge
    def purge(self) -> None:
        run_async(self.pool, self.controller.purge_videos,
                  on_done=self.run_panel.report_purge, on_error=self._on_error)

    # -------------------------------------------------------------- job reuse
    def attach_job(self) -> None:
        if self.controller.transport is None:
            QMessageBox.information(self, "Not connected",
                                    "Connect to the server first.")
            return
        self._set_busy(True, "Listing jobs...")
        run_async(self.pool, self.controller.server_jobs,
                  on_done=self._choose_job, on_error=self._on_error)

    def _choose_job(self, payload: dict) -> None:
        self._set_busy(False)
        jobs = payload.get("jobs", [])
        if not jobs:
            QMessageBox.information(
                self, "No jobs",
                f"No jobs found under {payload.get('root')}.")
            return
        labels = [
            f"{j['job_id']} · {j['project_name']} · {j['observations']} video(s) · "
            f"{', '.join(j['engines'])}"
            for j in jobs
        ]
        choice, ok = QInputDialog.getItem(self, "Re-open a job", "Job", labels, 0, False)
        if not ok:
            return
        job_dir = jobs[labels.index(choice)]["job_dir"]
        run_async(self.pool, self.controller.attach, job_dir,
                  on_done=self._job_attached, on_error=self._on_error)

    def _job_attached(self, job) -> None:
        self.run_panel.append_log(f"Re-opened job {job.job_id} ({job.job_dir})")
        self.ethogram_panel.refresh()
        self.engine_panel.refresh()
        self.source_panel.refresh()
        if job.client_source_dir and pathlib.Path(job.client_source_dir).is_dir():
            self.scan_folder(job.client_source_dir, self.controller.state.recursive)
        self.tabs.setCurrentWidget(self.run_panel)
        self.refresh_status()

    # ------------------------------------------------------------- templates
    def open_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a BORIS project as a template", "",
            "BORIS project (*.boris);;All files (*)")
        if not path:
            return
        try:
            project = BorisProject.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot open", str(exc))
            return
        self.controller.state.ethogram = project.ethogram
        if project.name:
            self.source_panel.name_edit.setText(project.name)
        self.ethogram_panel.refresh()
        self.engine_panel.refresh()

        problems = project.ethogram.validate()
        if any("both a behaviour code and a subject name" in p for p in problems):
            self._offer_migration()

    def _offer_migration(self) -> None:
        QMessageBox.information(
            self, "Subjects look like trial phases",
            "In this project the subject names match behaviour codes, which is "
            "the pattern you get when the subject field is used to record which "
            "phase of a session an event falls in.\n\n"
            "ABC codes the BORIS way: subjects are actors, and phases are state "
            "behaviours in an 'Episode' category. Use File > Convert a "
            "subjects-as-phases project to rewrite an existing project, or fix "
            "the ethogram here before submitting.")

    def export_ethogram(self) -> None:
        if not self.controller.state.ethogram.behaviors:
            QMessageBox.information(self, "Nothing to export", "The ethogram is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ethogram", "ethogram.boris", "BORIS project (*.boris)")
        if path:
            ethogram_to_boris_file(self.controller.state.ethogram, path,
                                   name=self.controller.state.project_name)
            self.statusBar().showMessage(f"Ethogram written to {path}", 6000)

    def migrate_project(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self, "Project to convert", "", "BORIS project (*.boris)")
        if not source:
            return
        names, ok = QInputDialog.getText(
            self, "Real subjects",
            "The actual subjects in this project, comma separated.\n"
            "Every behaviour event is attributed to the first one; the former "
            "subjects become state behaviours in an 'Episode' category.",
            text="Dog")
        if not ok or not names.strip():
            return
        target, _ = QFileDialog.getSaveFileName(
            self, "Save the converted project as",
            str(pathlib.Path(source).with_name(pathlib.Path(source).stem + "_converted.boris")),
            "BORIS project (*.boris)")
        if not target:
            return
        try:
            project = BorisProject.load(source)
            subjects = [Subject(n.strip()) for n in names.split(",") if n.strip()]
            migrated, report = migrate_subjects_to_episodes(project, subjects)
            migrated.save(target)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Conversion failed", str(exc))
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Converted")
        box.setText(f"Written to {target}\n\n{len(report)} change(s).")
        box.setDetailedText("\n".join(report))
        box.exec()

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About ABC",
            f"<h3>ABC {__version__}</h3>"
            "<p>Automated Behaviour Coder - an automated event recorder that "
            "writes BORIS projects.</p>"
            "<p>Videos are coded on a SLURM cluster by one or more engines; the "
            "project is assembled here, next to your videos, with relative paths "
            "so it stays portable.</p>"
            "<p><b>Every event it produces is a suggestion.</b> Published "
            "agreement between zero-shot video models and expert human coders is "
            "fair at best, and predicted timestamps are coarse. Review the "
            "coding in BORIS before you analyse it.</p>")

    # ----------------------------------------------------------------- close
    def _sync_panels(self) -> None:
        self.engine_panel.refresh_ownership()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        try:
            self.controller.save_settings()
            self.controller.close()
        except Exception:  # noqa: BLE001 - never block a close
            pass
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("ABC")
    app.setOrganizationName("ABC")

    controller = Controller(load_config("client"))
    window = MainWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
