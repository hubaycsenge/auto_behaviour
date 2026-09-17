"""The client <-> server contract.

A job is a directory on the server. The client writes ``job.json`` into it and
never touches anything else; the server reads that file, runs the engines, and
writes results back into the same directory. Everything the server needs is in
the manifest, so a job can be re-run, inspected or debugged long after the
client that made it has gone away::

    <job_dir>/
        job.json                    the manifest below
        status.json                 written by the server as it progresses
        videos/                     uploaded media (only in "upload" staging)
        results/<obs>.<engine>.json  one ObservationResult per engine per video
        merged/<obs>.json            fused result, when several engines ran
        logs/                        SLURM stdout/stderr, one file per array task
        slurm/                       generated sbatch scripts
"""

from __future__ import annotations

import datetime as _dt
import getpass
import json
import os
import pathlib
import socket
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from ..version import PROTOCOL_VERSION, __version__
from .ethogram import Ethogram
from .events import MediaInfo

#: Media lives on a filesystem both sides can see; the server reads it in place.
STAGING_SHARED = "shared"
#: Media was copied to the server and may be deleted from there afterwards.
STAGING_UPLOAD = "upload"


@dataclass
class ObservationSpec:
    """One video, one observation."""

    observation_id: str
    #: Path the *server* uses to read the media.
    server_media: list[str] = field(default_factory=list)
    #: Path the *client* knows the media by; used to write the BORIS project.
    client_media: list[str] = field(default_factory=list)
    info: list[MediaInfo] = field(default_factory=list)
    description: str = ""
    independent_variables: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max((m.duration for m in self.info), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["info"] = [m.to_dict() for m in self.info]
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ObservationSpec:
        return cls(
            observation_id=raw["observation_id"],
            server_media=list(raw.get("server_media", [])),
            client_media=list(raw.get("client_media", [])),
            info=[MediaInfo.from_dict(m) for m in raw.get("info", [])],
            description=raw.get("description", ""),
            independent_variables=raw.get("independent_variables", {}),
        )


@dataclass
class SlurmSpec:
    """How to ask SLURM for the resources an engine needs."""

    partition: str = "small"
    gres: str = ""                  # e.g. "gpu:a100:1"; empty for a CPU-only engine
    nodelist: str = ""              # pin to specific nodes, e.g. "nipg38"
    exclude: str = ""
    cpus_per_task: int = 4
    mem: str = "32G"
    time: str = "08:00:00"
    account: str = ""
    qos: str = ""
    #: Maximum array tasks running at once -- keeps one job from filling the cluster.
    array_throttle: int = 4
    extra_sbatch: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SlurmSpec:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class EngineSpec:
    """One engine to run, and the settings it should run with."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)
    slurm: SlurmSpec = field(default_factory=SlurmSpec)
    #: Behaviour codes this engine should look for. Empty means "all of them".
    behaviors: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "options": self.options,
            "slurm": self.slurm.to_dict(),
            "behaviors": self.behaviors,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EngineSpec:
        return cls(
            name=raw["name"],
            options=raw.get("options", {}),
            slurm=SlurmSpec.from_dict(raw.get("slurm", {})),
            behaviors=list(raw.get("behaviors", [])),
            enabled=bool(raw.get("enabled", True)),
        )


@dataclass
class JobSpec:
    """Everything the server needs in order to code a folder of videos."""

    job_id: str = field(default_factory=lambda: _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                        + "-" + uuid.uuid4().hex[:6])
    project_name: str = "ABC project"
    project_description: str = ""
    ethogram: Ethogram = field(default_factory=Ethogram)
    observations: list[ObservationSpec] = field(default_factory=list)
    engines: list[EngineSpec] = field(default_factory=list)
    #: behaviour code -> engine name that is authoritative for it.
    ownership: dict[str, str] = field(default_factory=dict)
    staging: str = STAGING_SHARED
    #: Absolute path of the job directory on the server.
    job_dir: str = ""
    #: Where the client keeps its videos, for writing relative BORIS paths.
    client_source_dir: str = ""
    created: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    created_by: str = field(default_factory=lambda: f"{_safe_user()}@{_safe_host()}")
    abc_version: str = __version__
    protocol_version: int = PROTOCOL_VERSION
    notes: str = ""

    # -- helpers ------------------------------------------------------------
    @property
    def enabled_engines(self) -> list[EngineSpec]:
        return [e for e in self.engines if e.enabled]

    def observation(self, obs_id: str) -> ObservationSpec | None:
        return next((o for o in self.observations if o.observation_id == obs_id), None)

    def validate(self) -> list[str]:
        problems = list(self.ethogram.validate())
        if not self.observations:
            problems.append("No videos selected.")
        if not self.enabled_engines:
            problems.append("No engine enabled.")
        seen: set[str] = set()
        for o in self.observations:
            if o.observation_id in seen:
                problems.append(f"Duplicate observation ID: {o.observation_id!r}.")
            seen.add(o.observation_id)
            if not o.server_media:
                problems.append(f"{o.observation_id!r} has no media path for the server.")
        for name, owner in self.ownership.items():
            if owner not in {e.name for e in self.enabled_engines}:
                problems.append(
                    f"Behaviour {name!r} is assigned to engine {owner!r}, which is not enabled."
                )
        return problems

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "project_name": self.project_name,
            "project_description": self.project_description,
            "ethogram": self.ethogram.to_dict(),
            "observations": [o.to_dict() for o in self.observations],
            "engines": [e.to_dict() for e in self.engines],
            "ownership": self.ownership,
            "staging": self.staging,
            "job_dir": self.job_dir,
            "client_source_dir": self.client_source_dir,
            "created": self.created,
            "created_by": self.created_by,
            "abc_version": self.abc_version,
            "protocol_version": self.protocol_version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> JobSpec:
        got = int(raw.get("protocol_version", 0))
        if got > PROTOCOL_VERSION:
            raise ValueError(
                f"job.json uses protocol version {got}, but this ABC understands "
                f"at most {PROTOCOL_VERSION}. Upgrade the server-side install."
            )
        return cls(
            job_id=raw.get("job_id", ""),
            project_name=raw.get("project_name", "ABC project"),
            project_description=raw.get("project_description", ""),
            ethogram=Ethogram.from_dict(raw.get("ethogram", {})),
            observations=[ObservationSpec.from_dict(o) for o in raw.get("observations", [])],
            engines=[EngineSpec.from_dict(e) for e in raw.get("engines", [])],
            ownership=raw.get("ownership", {}),
            staging=raw.get("staging", STAGING_SHARED),
            job_dir=raw.get("job_dir", ""),
            client_source_dir=raw.get("client_source_dir", ""),
            created=raw.get("created", ""),
            created_by=raw.get("created_by", ""),
            abc_version=raw.get("abc_version", ""),
            protocol_version=got,
            notes=raw.get("notes", ""),
        )

    def save(self, path: str | os.PathLike) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> JobSpec:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------
# job directory layout
# --------------------------------------------------------------------------

class JobLayout:
    """Paths inside a job directory. One place that knows the layout."""

    def __init__(self, job_dir: str | os.PathLike) -> None:
        self.root = pathlib.Path(job_dir)

    @property
    def manifest(self) -> pathlib.Path:
        return self.root / "job.json"

    @property
    def status(self) -> pathlib.Path:
        return self.root / "status.json"

    @property
    def videos(self) -> pathlib.Path:
        return self.root / "videos"

    @property
    def results(self) -> pathlib.Path:
        return self.root / "results"

    @property
    def merged(self) -> pathlib.Path:
        return self.root / "merged"

    @property
    def logs(self) -> pathlib.Path:
        return self.root / "logs"

    @property
    def slurm(self) -> pathlib.Path:
        return self.root / "slurm"

    def result_file(self, obs_id: str, engine: str) -> pathlib.Path:
        return self.results / f"{_safe_name(obs_id)}.{engine}.json"

    def merged_file(self, obs_id: str) -> pathlib.Path:
        return self.merged / f"{_safe_name(obs_id)}.json"

    def ensure(self) -> JobLayout:
        for d in (self.root, self.videos, self.results, self.merged, self.logs, self.slurm):
            d.mkdir(parents=True, exist_ok=True)
        return self


def _safe_name(name: str) -> str:
    """Make an observation ID safe as a filename without losing identity."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _safe_host() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"
