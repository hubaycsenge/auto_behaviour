"""Client logic with no Qt in it.

Everything the GUI does -- scanning a folder, deciding whether to upload,
submitting, polling, collecting, purging -- lives here so it can be tested and
scripted without a display. :mod:`abcoder.client.app` is a thin Qt shell over
this class.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..common import media as media_mod
from ..common.boris import BorisProject, ethogram_from_table
from ..common.config import load_config, save_config, suggest_ownership
from ..common.ethogram import EPISODE_CATEGORY, Ethogram
from ..common.events import MediaInfo
from ..common.jobspec import (
    STAGING_SHARED,
    STAGING_UPLOAD,
    EngineSpec,
    JobLayout,
    JobSpec,
    ObservationSpec,
    SlurmSpec,
)
from ..common.project_builder import BuildReport, write_project
from .transport import Transport, TransportError, detect_transport

ProgressFn = Callable[[str, float], None]


@dataclass
class VideoEntry:
    """One file found in the source folder, as the client sees it."""

    path: pathlib.Path
    observation_id: str
    size_bytes: int = 0
    info: MediaInfo | None = None
    include: bool = True
    problem: str = ""

    @property
    def duration(self) -> float:
        return self.info.duration if self.info else 0.0


@dataclass
class ClientState:
    """Everything the user has configured in the GUI."""

    project_name: str = "ABC project"
    project_description: str = ""
    source_dir: str = ""
    project_path: str = ""
    recursive: bool = False
    ethogram: Ethogram = field(default_factory=Ethogram)
    videos: list[VideoEntry] = field(default_factory=list)
    engines: dict[str, EngineSpec] = field(default_factory=dict)
    ownership: dict[str, str] = field(default_factory=dict)
    include_evidence: bool = True
    min_confidence: float = 0.0
    per_task: int = 1

    @property
    def selected_videos(self) -> list[VideoEntry]:
        return [v for v in self.videos if v.include and not v.problem]

    @property
    def enabled_engines(self) -> list[EngineSpec]:
        return [e for e in self.engines.values() if e.enabled]


class Controller:
    """Drives a job from folder to .boris file."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config("client")
        self.state = ClientState()
        self.transport: Transport | None = None
        self.uploads_required = False
        self.server_info: dict[str, Any] = {}
        self.job: JobSpec | None = None
        self.job_dir: str = ""

    # ------------------------------------------------------------------ setup
    def connect(self, source_dir: str | None = None) -> tuple[str, dict[str, Any]]:
        """Pick a transport and ask the server what it can do.

        Returns ``(description, check_payload)``. Raises
        :class:`~abcoder.client.transport.base.TransportError` if the server is
        unreachable, which the GUI shows without wedging the window.
        """
        if self.transport is not None:
            self.transport.close()
        self.transport, self.uploads_required = detect_transport(
            self.config, source_dir or self.state.source_dir or None)
        self.server_info = self.transport.check()
        return self.transport.description, self.server_info

    def available_engines(self) -> dict[str, Any]:
        """The engine catalogue from the server, annotated with availability."""
        if self.transport is None:
            raise TransportError("not connected")
        catalogue = self.transport.engines()
        checks = self.server_info.get("engines", {})
        for name, entry in catalogue.items():
            entry["available"] = bool(checks.get(name, {}).get("available", False))
            entry["unavailable_reason"] = checks.get(name, {}).get("reason", "")
        return catalogue

    def set_engine_defaults(self, catalogue: dict[str, Any]) -> None:
        """Seed the engine table from the catalogue, keeping any user edits."""
        for name, entry in catalogue.items():
            if name in self.state.engines:
                continue
            self.state.engines[name] = EngineSpec(
                name=name,
                options=dict(entry.get("options", {})),
                slurm=SlurmSpec.from_dict(entry.get("slurm", {})),
                enabled=False,
            )

    # ------------------------------------------------------------------ media
    def scan_source(self, folder: str | pathlib.Path, recursive: bool | None = None,
                    progress: ProgressFn | None = None) -> list[VideoEntry]:
        """Find the videos, derive observation IDs, probe what we can.

        Probing happens locally when this machine has codecs, and on the server
        otherwise -- a laptop with no ffmpeg still gets durations, because the
        BORIS project needs them to draw its timeline.
        """
        folder = pathlib.Path(folder).expanduser()
        recursive = self.state.recursive if recursive is None else recursive
        paths = media_mod.discover_videos(folder, recursive=recursive)

        duplicates = media_mod.duplicate_observation_ids(paths)
        entries: list[VideoEntry] = []
        for p in paths:
            obs_id = media_mod.observation_id_for(p)
            problem = ""
            if obs_id in duplicates:
                others = [q.name for q in duplicates[obs_id] if q != p]
                problem = (f"observation ID {obs_id!r} is also used by "
                           f"{', '.join(others)}; rename one of them")
            entries.append(VideoEntry(path=p, observation_id=obs_id,
                                      size_bytes=p.stat().st_size, problem=problem,
                                      include=not problem))

        self._probe(entries, progress)
        self.state.source_dir = str(folder)
        self.state.videos = entries
        return entries

    def _probe(self, entries: Sequence[VideoEntry], progress: ProgressFn | None) -> None:
        if media_mod.BACKEND is not None:
            for i, entry in enumerate(entries):
                if progress:
                    progress(f"Probing {entry.path.name}", (i + 1) / max(1, len(entries)))
                try:
                    entry.info = media_mod.probe(entry.path)
                except Exception as exc:  # noqa: BLE001
                    entry.problem = entry.problem or f"cannot read this file: {exc}"
                    entry.include = False
            return

        # No local codecs: ask the server, but only when it can see the files.
        if self.transport is None or self.uploads_required:
            for entry in entries:
                entry.info = MediaInfo(path=str(entry.path))
            return
        try:
            if progress:
                progress("Probing media on the server", 0.5)
            payload = self.transport.abc(
                ["probe", *[str(e.path) for e in entries]], timeout=600)
            by_path = {row.get("path"): row for row in payload if isinstance(row, dict)}
            for entry in entries:
                row = by_path.get(str(entry.path), {})
                entry.info = (MediaInfo.from_dict(row) if "duration" in row
                              else MediaInfo(path=str(entry.path)))
        except TransportError:
            for entry in entries:
                entry.info = MediaInfo(path=str(entry.path))

    # -------------------------------------------------------------- ethogram
    def load_ethogram(self, path: str | pathlib.Path) -> Ethogram:
        """Import an ethogram from a .boris project or a BORIS spreadsheet."""
        path = pathlib.Path(path).expanduser()
        if path.suffix.lower() == ".boris":
            project = BorisProject.load(path)
            etho = project.ethogram
        else:
            etho = ethogram_from_table(path)
        if etho.behaviors:
            self.state.ethogram.behaviors = etho.behaviors
            self.state.ethogram.categories = etho.categories
        if etho.subjects:
            self.state.ethogram.subjects = etho.subjects
        return self.state.ethogram

    def suggest_ownership(self) -> dict[str, str]:
        names = [e.name for e in self.state.enabled_engines]
        self.state.ownership = suggest_ownership(self.state.ethogram.codes, names)
        return self.state.ownership

    def mark_episode_behaviors(self, codes: Iterable[str]) -> None:
        """Move the named behaviours into the Episode category as state events.

        The client offers this because the alternative -- a subject per trial
        phase -- is the idiom this tool exists to replace, and a project that
        arrives that way should be fixable in one click.
        """
        wanted = set(codes)
        etho = self.state.ethogram
        for b in etho.behaviors:
            if b.code in wanted:
                b.category = EPISODE_CATEGORY
                if not b.is_state:
                    b.type = "State event"
        if wanted and EPISODE_CATEGORY not in etho.categories:
            etho.categories.append(EPISODE_CATEGORY)

    # ------------------------------------------------------------------- job
    def build_job(self) -> JobSpec:
        """Turn the current state into a manifest, without touching the server."""
        if self.transport is None:
            raise TransportError("not connected")

        staging = STAGING_UPLOAD if self.uploads_required else STAGING_SHARED
        job = JobSpec(
            project_name=self.state.project_name,
            project_description=self.state.project_description,
            ethogram=self.state.ethogram,
            engines=list(self.state.enabled_engines),
            ownership=dict(self.state.ownership),
            staging=staging,
            client_source_dir=self.state.source_dir,
        )
        job_dir = self.transport.remote_join(self.transport.jobs_root, job.job_id)
        job.job_dir = job_dir

        for entry in self.state.selected_videos:
            if staging == STAGING_UPLOAD:
                server_path = self.transport.remote_join(job_dir, "videos", entry.path.name)
            else:
                server_path = str(entry.path.resolve())
            info = entry.info or MediaInfo(path=server_path)
            job.observations.append(ObservationSpec(
                observation_id=entry.observation_id,
                server_media=[server_path],
                client_media=[str(entry.path.resolve())],
                info=[MediaInfo(**{**info.to_dict(), "path": server_path})],
            ))

        self.job = job
        self.job_dir = job_dir
        return job

    def upload(self, progress: ProgressFn | None = None) -> int:
        """Copy the selected videos to the server. Returns bytes transferred.

        A no-op on a shared filesystem -- that is the whole point of detecting
        it -- so the GUI can call this unconditionally.
        """
        if self.job is None or self.transport is None:
            raise TransportError("build the job first")
        if self.job.staging != STAGING_UPLOAD:
            return 0

        total = sum(v.size_bytes for v in self.state.selected_videos) or 1
        sent = 0
        for entry in self.state.selected_videos:
            obs = self.job.observation(entry.observation_id)
            if obs is None:
                continue
            remote = obs.server_media[0]
            if self.transport.exists(remote):
                sent += entry.size_bytes
                if progress:
                    progress(f"{entry.path.name} already on the server", sent / total)
                continue

            base = sent

            def report(done: int, _total: int, name=entry.path.name, base=base) -> None:
                if progress:
                    progress(f"Uploading {name}", min(1.0, (base + done) / total))

            self.transport.put_file(entry.path, remote, progress=report)
            sent += entry.size_bytes
            if progress:
                progress(f"Uploaded {entry.path.name}", min(1.0, sent / total))
        return sent

    def write_manifest(self) -> str:
        """Write job.json on the server and return the job directory."""
        if self.job is None or self.transport is None:
            raise TransportError("build the job first")
        self.transport.makedirs(self.job_dir)
        for sub in ("videos", "results", "merged", "logs", "slurm"):
            self.transport.makedirs(self.transport.remote_join(self.job_dir, sub))
        self.transport.write_text(
            self.transport.remote_join(self.job_dir, "job.json"),
            json.dumps(self.job.to_dict(), indent=2, ensure_ascii=False),
        )
        return self.job_dir

    def submit(self, dry_run: bool = False) -> dict[str, Any]:
        if self.transport is None or not self.job_dir:
            raise TransportError("write the manifest first")
        args = ["submit", self.job_dir, "--per-task", str(self.state.per_task)]
        if dry_run:
            args.append("--dry-run")
        return self.transport.abc(args, timeout=300)

    def status(self, per_observation: bool = False) -> dict[str, Any]:
        if self.transport is None or not self.job_dir:
            raise TransportError("no job submitted")
        args = ["status", self.job_dir]
        if per_observation:
            args.append("--per-observation")
        return self.transport.abc(args, timeout=180)

    def cancel(self) -> dict[str, Any]:
        if self.transport is None or not self.job_dir:
            raise TransportError("no job submitted")
        return self.transport.abc(["cancel", self.job_dir], timeout=120)

    # --------------------------------------------------------------- collect
    def collect(self, project_path: str | pathlib.Path | None = None,
                progress: ProgressFn | None = None) -> BuildReport:
        """Build the BORIS project on this machine.

        Results are fetched to a local staging directory first when the server
        is remote, so the project is assembled against the client's own copy of
        the videos and its paths come out relative and portable.
        """
        if self.transport is None or self.job is None:
            raise TransportError("no job to collect")
        target = pathlib.Path(project_path or self.state.project_path).expanduser()
        if not target.suffix:
            target = target.with_suffix(".boris")

        if self.transport.kind == "shared":
            layout = JobLayout(self.job_dir)
            job = JobSpec.load(layout.manifest)
            local_layout = layout
        else:
            staging = pathlib.Path(tempfile.mkdtemp(prefix="abc-collect-"))
            local_layout = JobLayout(staging).ensure()
            manifest_text = self.transport.read_text(
                self.transport.remote_join(self.job_dir, "job.json"))
            local_layout.manifest.write_text(manifest_text, encoding="utf-8")
            job = JobSpec.load(local_layout.manifest)

            names = self.transport.listdir(self.transport.remote_join(self.job_dir, "results"))
            for i, name in enumerate(names):
                if not name.endswith(".json"):
                    continue
                if progress:
                    progress(f"Fetching {name}", (i + 1) / max(1, len(names)))
                self.transport.get_file(
                    self.transport.remote_join(self.job_dir, "results", name),
                    local_layout.results / name,
                )

        # The manifest on the server records server-side media paths; the
        # project must describe the client's copies instead.
        client_paths = {v.observation_id: str(v.path.resolve())
                        for v in self.state.videos}
        for obs in job.observations:
            local = client_paths.get(obs.observation_id)
            if local:
                obs.client_media = [local]

        report = write_project(
            job, local_layout, target,
            include_evidence=self.state.include_evidence,
            min_confidence=self.state.min_confidence,
        )
        self.state.project_path = str(target)
        return report

    # ----------------------------------------------------------------- purge
    def purge_videos(self) -> dict[str, Any]:
        """Delete the uploaded videos from the server.

        Refuses on a shared filesystem, where the "uploaded" path is the
        researcher's own source folder. The server enforces this too; the
        client checks so the button can explain itself before anything happens.
        """
        if self.transport is None or not self.job_dir:
            raise TransportError("no job on the server")
        if self.job is not None and self.job.staging != STAGING_UPLOAD:
            return {"deleted": [], "bytes_freed": 0,
                    "note": "This job reads your videos where they already are; "
                            "nothing was copied to the server."}
        return self.transport.abc(["purge", self.job_dir, "--videos"], timeout=300)

    def server_jobs(self) -> dict[str, Any]:
        if self.transport is None:
            raise TransportError("not connected")
        return self.transport.abc(["jobs", "--root", self.transport.jobs_root], timeout=180)

    def purge_job(self, job_dir: str, force: bool = False) -> dict[str, Any]:
        if self.transport is None:
            raise TransportError("not connected")
        args = ["purge", job_dir, "--all"]
        if force:
            args.append("--force")
        return self.transport.abc(args, timeout=300)

    def attach(self, job_dir: str) -> JobSpec:
        """Re-open a job submitted earlier, so the client can resume polling."""
        if self.transport is None:
            raise TransportError("not connected")
        text = self.transport.read_text(self.transport.remote_join(job_dir, "job.json"))
        self.job = JobSpec.from_dict(json.loads(text))
        self.job_dir = job_dir
        self.state.project_name = self.job.project_name
        self.state.ethogram = self.job.ethogram
        self.state.ownership = dict(self.job.ownership)
        self.state.engines = {e.name: e for e in self.job.engines}
        if self.job.client_source_dir:
            self.state.source_dir = self.job.client_source_dir
        return self.job

    # --------------------------------------------------------------- session
    def save_settings(self) -> None:
        self.config.setdefault("client", {})
        self.config["client"]["last_source_dir"] = self.state.source_dir
        self.config["client"]["last_project_dir"] = str(
            pathlib.Path(self.state.project_path).parent) if self.state.project_path else ""
        self.config["client"]["recursive_scan"] = self.state.recursive
        save_config(self.config, "client")

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
