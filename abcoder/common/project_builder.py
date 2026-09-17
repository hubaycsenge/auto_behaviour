"""Assemble a BORIS project from a job's results.

This runs on the client, because the project has to describe the videos as the
*client* knows them: the ``.boris`` file and the folder of videos must travel
together, and a path like ``/nas/home/.../Merged/clip.mp4`` written into the
project is useless the moment the project is opened on a laptop.

The fusion step is here too, since a project built from several engines has to
decide which one wins where they disagree. The default is the ownership table
from the job -- audio owns the vocalisations, pose owns the proxemics, the VLM
owns the rest -- with unowned behaviours pooled and deduplicated.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .boris import BorisProject, relative_media_path
from .events import Event, MediaInfo, ObservationResult, fuse
from .jobspec import JobLayout, JobSpec


@dataclass
class BuildReport:
    """What happened while building the project, for the client to display."""

    observations_written: int = 0
    observations_skipped: list[str] = field(default_factory=list)
    observations_failed: dict[str, str] = field(default_factory=dict)
    total_events: int = 0
    warnings: list[str] = field(default_factory=list)
    missing_media: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.observations_written > 0

    def summary(self) -> str:
        bits = [f"{self.observations_written} observation(s), {self.total_events} event(s)"]
        if self.observations_failed:
            bits.append(f"{len(self.observations_failed)} failed")
        if self.observations_skipped:
            bits.append(f"{len(self.observations_skipped)} with no result yet")
        if self.missing_media:
            bits.append(f"{len(self.missing_media)} media file(s) not found locally")
        return "; ".join(bits)


def collect_results(
    job: JobSpec, layout: JobLayout, observation_id: str
) -> list[ObservationResult]:
    """Every engine's result for one observation, skipping ones not written yet."""
    out: list[ObservationResult] = []
    for engine in job.enabled_engines:
        path = layout.result_file(observation_id, engine.name)
        if not path.is_file():
            continue
        try:
            out.append(ObservationResult.load(path))
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop the build
            out.append(ObservationResult(
                observation_id=observation_id, failed=True,
                error=f"could not read {path.name}: {exc}",
                engine={"name": engine.name},
            ))
    return out


def build_project(
    job: JobSpec,
    layout: JobLayout,
    project_path: str | pathlib.Path,
    *,
    media_resolver=None,
    include_evidence: bool = True,
    min_confidence: float = 0.0,
    only: Iterable[str] | None = None,
) -> tuple[BorisProject, BuildReport]:
    """Build the BORIS project and return it with a report.

    *media_resolver* maps an :class:`~abcoder.common.jobspec.ObservationSpec` to
    the media paths on *this* machine. The default uses the ``client_media``
    recorded at submission, falling back to the server paths when the job was
    run on a shared filesystem -- which is the same file, seen from the same
    place.

    *include_evidence* copies each engine's one-line justification into the
    BORIS comment field. It makes the coding auditable in BORIS itself, which is
    the whole point of a first-pass machine coding, so it is on by default.
    """
    project_path = pathlib.Path(project_path).expanduser()
    report = BuildReport()
    wanted = set(only) if only is not None else None

    project = BorisProject()
    project.name = job.project_name or "ABC project"
    project.description = _project_description(job)
    project.ethogram = job.ethogram
    project.data["project_date"] = _dt.datetime.now().isoformat(timespec="seconds")

    resolver = media_resolver or _default_media_resolver

    for obs in job.observations:
        if wanted is not None and obs.observation_id not in wanted:
            continue

        results = collect_results(job, layout, obs.observation_id)
        if not results:
            report.observations_skipped.append(obs.observation_id)
            continue

        fatal = [r for r in results if r.failed]
        if len(fatal) == len(results):
            report.observations_failed[obs.observation_id] = fatal[0].error
            continue
        for r in fatal:
            report.warnings.append(
                f"{obs.observation_id}: engine {r.engine.get('name', '?')} failed: {r.error}"
            )

        usable = [r for r in results if not r.failed]
        events, warnings = fuse(usable, job.ethogram, job.ownership, obs.duration)
        if min_confidence > 0:
            events = [e for e in events
                      if e.confidence is None or e.confidence >= min_confidence]
        events = [_with_comment(e, include_evidence) for e in events]
        report.warnings.extend(f"{obs.observation_id}: {w}" for w in warnings)

        client_paths = resolver(obs)
        stored: list[str] = []
        info: list[MediaInfo] = []
        for i, local in enumerate(client_paths):
            local_path = pathlib.Path(local).expanduser()
            if not local_path.exists():
                report.missing_media.append(str(local_path))
            rel = relative_media_path(local_path, project_path)
            stored.append(rel)
            source = obs.info[i] if i < len(obs.info) else MediaInfo(path=rel)
            info.append(MediaInfo(**{**source.to_dict(), "path": rel}))

        project.add_observation(
            obs.observation_id,
            media_paths=stored,
            events=events,
            media_info=info,
            description=_observation_description(obs, usable),
            independent_variables=obs.independent_variables,
        )
        report.observations_written += 1
        report.total_events += len(events)

    return project, report


def _default_media_resolver(obs) -> list[str]:
    return list(obs.client_media or obs.server_media)


def _with_comment(event: Event, include_evidence: bool) -> Event:
    """Fold provenance into the BORIS comment so the coding is auditable there."""
    if not include_evidence:
        return event
    bits = [b for b in (event.comment or "").split(" | ") if b]
    if event.source:
        confidence = f" {event.confidence:.2f}" if event.confidence is not None else ""
        bits.append(f"[{event.source}{confidence}]")
    if event.evidence:
        bits.append(event.evidence)
    return Event(**{**event.to_dict(), "comment": " | ".join(bits)})


def _project_description(job: JobSpec) -> str:
    engines = ", ".join(e.name for e in job.enabled_engines)
    lines = [
        job.project_description.strip(),
        "",
        "Coded automatically by ABC (Automated Behaviour Coder).",
        f"Job {job.job_id}, submitted {job.created} by {job.created_by}.",
        f"Engines: {engines or 'none'}.",
        "",
        "Every event in this project is a machine suggestion. Timestamps from a "
        "video-LLM are approximate, and published zero-shot agreement with expert "
        "human coders is fair at best. Review before analysis.",
    ]
    return "\n".join(x for x in lines if x is not None).strip()


def _observation_description(obs, results: Sequence[ObservationResult]) -> str:
    bits = [obs.description.strip()] if obs.description.strip() else []
    for r in results:
        name = r.engine.get("name", "?")
        model = r.engine.get("model") or r.engine.get("classifier") or ""
        seconds = r.diagnostics.get("seconds")
        detail = f"{name}"
        if model:
            detail += f" ({model})"
        if seconds:
            detail += f", {seconds}s"
        detail += f", {len(r.events)} event(s)"
        bits.append(detail)
    return "\n".join(bits)


def write_project(
    job: JobSpec,
    layout: JobLayout,
    project_path: str | pathlib.Path,
    **kwargs,
) -> BuildReport:
    """Build and save in one call."""
    project, report = build_project(job, layout, project_path, **kwargs)
    project.save(project_path)
    return report
