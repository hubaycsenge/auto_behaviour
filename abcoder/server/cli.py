"""The ``abc`` command line.

The same binary is the server-side worker and the client's remote hands: when
the client talks to the cluster over SSH it does so by running these
subcommands and parsing their JSON. That keeps one protocol instead of two, and
it means everything the GUI can do is also scriptable::

    abc check                        what can this node actually run?
    abc engines                      engine list and defaults, as JSON
    abc submit  <job-dir>            queue the SLURM array jobs
    abc status  <job-dir>            progress, as JSON
    abc collect <job-dir> -o p.boris build the BORIS project
    abc purge   <job-dir> --videos   delete uploaded media from the server
    abc migrate old.boris new.boris  fix a subjects-as-phases project
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

from ..common import media as media_mod
from ..common.boris import (
    BorisProject,
    ethogram_from_table,
    ethogram_to_boris_file,
    migrate_subjects_to_episodes,
)
from ..common.config import ENGINE_PRESETS, engine_defaults, load_config
from ..common.ethogram import Subject
from ..common.jobspec import STAGING_UPLOAD, JobLayout, JobSpec
from ..common.paths import ProbeToken
from ..common.project_builder import write_project
from ..version import PROTOCOL_VERSION, __version__
from . import slurm as slurm_mod


def _emit(data: Any) -> None:
    """Every machine-readable subcommand prints exactly one JSON document."""
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_check(args) -> int:
    """Report what this machine can run, so the client can hide what it cannot.

    This runs on every client connect, so it must be fast and it must not hang.
    That rules out importing the frameworks it reports on: importing torch costs
    seconds on a warm NAS and can stall indefinitely on a half-installed one,
    which would take the GUI's connect with it. Availability is answered with
    :func:`importlib.util.find_spec`, which locates a package without executing
    it, and GPU facts come from ``nvidia-smi`` rather than ``torch.cuda``.
    """
    report: dict[str, Any] = {
        "abc_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "python": sys.version.split()[0],
        "media_backend": media_mod.BACKEND,
        "slurm": bool(shutil.which("sbatch")),
        "engines": {},
        "gpu": _gpu_info(),
    }
    report["torch"] = _installed_version("torch")

    checks: dict[str, Any] = {
        "mock": lambda: (True, ""),
        "vlm_vllm": lambda: _needs_packages("vllm", "torch"),
        "vlm_llamacpp": lambda: (
            (True, "") if shutil.which("llama-server")
            else (False, "llama-server is not on PATH (setup/install_llamacpp.sh)")),
        "audio": lambda: (
            _needs_packages("numpy") if media_mod.BACKEND == "av"
            else (False, "the audio engine needs PyAV (pip install av)")),
        "pose": lambda: _needs_packages("ultralytics"),
    }
    for name, check in checks.items():
        try:
            available, reason = check()
        except Exception as exc:  # noqa: BLE001
            available, reason = False, str(exc)
        entry: dict[str, Any] = {"available": bool(available)}
        if reason:
            entry["reason"] = reason
        report["engines"][name] = entry

    _emit(report)
    return 0


def _module_present(name: str) -> bool:
    """Is *name* importable, without importing it?"""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _needs_packages(*names: str) -> tuple[bool, str]:
    missing = [n for n in names if not _module_present(n)]
    if missing:
        return False, "not installed: " + ", ".join(missing)
    return True, ""


def _installed_version(name: str) -> str | None:
    """A package's version from its metadata -- again without importing it."""
    from importlib import metadata

    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 - absent, or installed without metadata
        return None


def _gpu_info() -> dict[str, Any] | None:
    """GPU facts from nvidia-smi.

    Compute capability is what decides which engines can run here, and
    ``nvidia-smi --query-gpu=compute_cap`` answers in milliseconds where
    ``torch.cuda.get_device_capability`` costs a multi-second CUDA init.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = [r.strip() for r in proc.stdout.splitlines() if r.strip()]
    if proc.returncode != 0 or not rows:
        return None

    first = [c.strip() for c in rows[0].split(",")]
    name = first[0]
    capability = first[1] if len(first) > 1 else ""
    try:
        vram_mb = int(float(first[2])) if len(first) > 2 else 0
    except ValueError:
        vram_mb = 0
    try:
        major = int(float(capability))
    except ValueError:
        major = 0

    return {
        "name": name,
        "count": len(rows),
        "compute_capability": capability,
        "vram_gb": round(vram_mb / 1024.0, 1) if vram_mb else 0,
        # Ampere and later. This is the line that decides vLLM vs llama.cpp.
        "bf16": major >= 8,
    }


def cmd_engines(args) -> int:
    """The engine catalogue: labels, descriptions and defaults, as JSON."""
    cfg = load_config("server")
    out = {}
    for name, preset in ENGINE_PRESETS.items():
        defaults = engine_defaults(name, cfg)
        out[name] = {
            "label": preset["label"],
            "description": preset["description"],
            "requires": preset.get("requires", {}),
            "options": defaults["options"],
            "slurm": defaults["slurm"],
        }
    _emit(out)
    return 0


def cmd_probe(args) -> int:
    """Probe media files and print their duration/fps/streams as JSON."""
    out = []
    for path in args.paths:
        try:
            out.append(media_mod.probe(path).to_dict())
        except Exception as exc:  # noqa: BLE001
            out.append({"path": path, "error": str(exc)})
    _emit(out)
    return 0


def cmd_scan(args) -> int:
    """List the media files in a folder, with their observation IDs."""
    try:
        paths = media_mod.discover_videos(args.folder, recursive=args.recursive)
    except (NotADirectoryError, OSError) as exc:
        _emit({"error": str(exc)})
        return 2
    duplicates = media_mod.duplicate_observation_ids(paths)
    _emit({
        "folder": str(pathlib.Path(args.folder).expanduser()),
        "files": [
            {"path": str(p), "observation_id": media_mod.observation_id_for(p),
             "bytes": p.stat().st_size}
            for p in paths
        ],
        "duplicate_observation_ids": {k: [str(p) for p in v] for k, v in duplicates.items()},
    })
    return 0


def cmd_verify_probe(args) -> int:
    """Confirm a client-written probe file is visible here (shared-filesystem test)."""
    _emit({"shared": ProbeToken(path=args.path, digest=args.digest).verify()})
    return 0


def cmd_submit(args) -> int:
    layout = JobLayout(args.job_dir)
    if not layout.manifest.is_file():
        _emit({"error": f"no job.json in {args.job_dir}"})
        return 2
    job = JobSpec.load(layout.manifest)

    problems = job.validate()
    if problems and not args.force:
        _emit({"error": "job is not valid", "problems": problems})
        return 2

    cfg = load_config("server")
    python = args.python or cfg["server"].get("python") or _venv_python(cfg) or sys.executable
    abc_root = args.abc_root or str(pathlib.Path(__file__).resolve().parents[2])
    hf_home = args.hf_home or str(pathlib.Path(cfg["server"].get("hf_home", "")).expanduser())

    try:
        arrays, preflight_warnings = slurm_mod.submit(
            job, layout, python=python, abc_root=abc_root, hf_home=hf_home,
            per_task=args.per_task, dry_run=args.dry_run, force=args.force)
    except slurm_mod.SlurmError as exc:
        _emit({"error": str(exc)})
        return 1

    _emit({
        "job_id": job.job_id, "job_dir": str(layout.root), "dry_run": args.dry_run,
        "submitted": [a.to_dict() for a in arrays],
        "warnings": problems + preflight_warnings,
    })
    return 0


def cmd_status(args) -> int:
    layout = JobLayout(args.job_dir)
    if not layout.manifest.is_file():
        _emit({"error": f"no job.json in {args.job_dir}"})
        return 2
    job = JobSpec.load(layout.manifest)
    prog = slurm_mod.progress(job, layout)
    payload = prog.to_dict()
    payload["job_id"] = job.job_id
    payload["job_dir"] = str(layout.root)
    payload["staging"] = job.staging
    payload["arrays"] = [a.to_dict() for a in slurm_mod.submitted_arrays(layout)]
    if args.per_observation:
        payload["observations"] = _per_observation(job, layout)
    _emit(payload)
    return 0


def _per_observation(job: JobSpec, layout: JobLayout) -> list[dict[str, Any]]:
    rows = []
    for obs in job.observations:
        row: dict[str, Any] = {"observation_id": obs.observation_id, "engines": {}}
        for engine in job.enabled_engines:
            path = layout.result_file(obs.observation_id, engine.name)
            if not path.is_file():
                row["engines"][engine.name] = {"state": "pending"}
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                row["engines"][engine.name] = {
                    "state": "failed" if data.get("failed") else "done",
                    "events": len(data.get("events", [])),
                    "error": data.get("error", ""),
                    "warnings": len(data.get("warnings", [])),
                }
            except (OSError, json.JSONDecodeError):
                row["engines"][engine.name] = {"state": "pending"}
        rows.append(row)
    return rows


def cmd_collect(args) -> int:
    layout = JobLayout(args.job_dir)
    if not layout.manifest.is_file():
        _emit({"error": f"no job.json in {args.job_dir}"})
        return 2
    job = JobSpec.load(layout.manifest)

    report = write_project(
        job, layout, args.out,
        include_evidence=not args.no_evidence,
        min_confidence=args.min_confidence,
    )
    _emit({
        "project": str(pathlib.Path(args.out).expanduser().resolve()),
        "summary": report.summary(),
        "observations_written": report.observations_written,
        "observations_skipped": report.observations_skipped,
        "observations_failed": report.observations_failed,
        "total_events": report.total_events,
        "missing_media": report.missing_media,
        "warnings": report.warnings[:200],
    })
    return 0 if report.ok else 1


def cmd_run_task(args) -> int:
    from .runner import main as runner_main
    argv = ["--job-dir", args.job_dir, "--engine", args.engine,
            "--shard-index", str(args.shard_index), "--per-task", str(args.per_task)]
    if args.force:
        argv.append("--force")
    for obs in args.observation or []:
        argv += ["--observation", obs]
    return runner_main(argv)


def cmd_purge(args) -> int:
    """Delete a job's uploaded videos, or the whole job directory.

    This is what the client's "Delete videos from server" button calls. It
    refuses to touch media that was never uploaded -- on a shared filesystem the
    path in the manifest is the researcher's own copy, and deleting it would
    destroy the source data rather than a cache.
    """
    layout = JobLayout(args.job_dir)
    if not layout.manifest.is_file():
        _emit({"error": f"no job.json in {args.job_dir}"})
        return 2
    job = JobSpec.load(layout.manifest)

    removed: list[str] = []
    freed = 0

    if args.all:
        if job.staging != STAGING_UPLOAD and not args.force:
            _emit({"error": "refusing to delete a shared-filesystem job directory "
                            "without --force (its videos are your originals)"})
            return 2
        freed = _tree_size(layout.root)
        shutil.rmtree(layout.root, ignore_errors=True)
        _emit({"deleted": "job directory", "path": str(layout.root),
               "bytes_freed": freed})
        return 0

    if job.staging != STAGING_UPLOAD:
        _emit({
            "deleted": [], "bytes_freed": 0,
            "note": "this job reads the videos in place on shared storage; "
                    "there is nothing uploaded to delete.",
        })
        return 0

    if layout.videos.is_dir():
        freed = _tree_size(layout.videos)
        for child in sorted(layout.videos.iterdir()):
            removed.append(child.name)
        shutil.rmtree(layout.videos, ignore_errors=True)
        layout.videos.mkdir(parents=True, exist_ok=True)

    _emit({"deleted": removed, "bytes_freed": freed, "path": str(layout.videos)})
    return 0


def cmd_jobs(args) -> int:
    """List the jobs under the configured jobs root."""
    cfg = load_config("server")
    root = pathlib.Path(args.root or cfg["server"]["jobs_root"]).expanduser()
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for child in sorted(root.iterdir(), reverse=True):
            manifest = child / "job.json"
            if not manifest.is_file():
                continue
            try:
                job = JobSpec.load(manifest)
            except Exception:  # noqa: BLE001
                continue
            layout = JobLayout(child)
            prog = slurm_mod.progress(job, layout) if args.progress else None
            rows.append({
                "job_id": job.job_id, "job_dir": str(child),
                "project_name": job.project_name, "created": job.created,
                "staging": job.staging, "observations": len(job.observations),
                "engines": [e.name for e in job.enabled_engines],
                "video_bytes": _tree_size(layout.videos) if layout.videos.is_dir() else 0,
                "progress": prog.to_dict() if prog else None,
            })
    _emit({"root": str(root), "jobs": rows})
    return 0


def cmd_cancel(args) -> int:
    layout = JobLayout(args.job_dir)
    arrays = slurm_mod.submitted_arrays(layout)
    slurm_mod.cancel([a.job_id for a in arrays])
    _emit({"cancelled": [a.job_id for a in arrays]})
    return 0


def cmd_migrate(args) -> int:
    """Rewrite a project that used the subject field to mark trial phases."""
    project = BorisProject.load(args.source)
    subjects = [Subject(name=n.strip()) for n in (args.subject or []) if n.strip()]
    if not subjects:
        _emit({"error": "give at least one real subject with --subject (e.g. --subject Dog)"})
        return 2
    migrated, report = migrate_subjects_to_episodes(project, subjects,
                                                    category=args.category)
    migrated.save(args.out)
    _emit({"out": str(pathlib.Path(args.out).resolve()),
           "subjects": [s.name for s in subjects],
           "episode_behaviors": [b.code for b in migrated.ethogram.episode_behaviors()],
           "changes": report})
    return 0


def cmd_import_ethogram(args) -> int:
    """Convert a BORIS ethogram spreadsheet into an ethogram-only .boris file."""
    etho = ethogram_from_table(args.source)
    if args.subjects:
        subject_sheet = ethogram_from_table(args.subjects)
        etho.subjects = subject_sheet.subjects
    problems = etho.validate()
    ethogram_to_boris_file(etho, args.out, name=args.name or "")
    _emit({"out": str(pathlib.Path(args.out).resolve()),
           "behaviors": len(etho), "subjects": etho.subject_names,
           "problems": problems})
    return 0 if not problems else 0


def cmd_gui(args) -> int:
    from ..client.app import main as gui_main
    return gui_main([])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _venv_python(cfg: dict[str, Any]) -> str:
    venv = cfg["server"].get("venv", "")
    if not venv:
        return ""
    candidate = pathlib.Path(venv).expanduser() / "bin" / "python"
    return str(candidate) if candidate.is_file() else ""


def _tree_size(path: pathlib.Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abc", description="ABC - Automated Behaviour Coder")
    parser.add_argument("--version", action="version", version=f"ABC {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="report what this machine can run")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("engines", help="engine catalogue and defaults, as JSON")
    p.set_defaults(func=cmd_engines)

    p = sub.add_parser("probe", help="probe media files")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("scan", help="list media in a folder with observation IDs")
    p.add_argument("folder")
    p.add_argument("--recursive", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("verify-probe", help="shared-filesystem check (internal)")
    p.add_argument("path")
    p.add_argument("digest")
    p.set_defaults(func=cmd_verify_probe)

    p = sub.add_parser("submit", help="submit a job's SLURM array jobs")
    p.add_argument("job_dir")
    p.add_argument("--per-task", type=int, default=1,
                   help="observations per array task (default 1)")
    p.add_argument("--python", default="", help="interpreter for the tasks")
    p.add_argument("--abc-root", default="", help="directory holding the abcoder package")
    p.add_argument("--hf-home", default="", help="HF_HOME for model downloads")
    p.add_argument("--dry-run", action="store_true", help="write scripts, do not submit")
    p.add_argument("--force", action="store_true",
                   help="submit despite validation or shared-storage problems")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="job progress, as JSON")
    p.add_argument("job_dir")
    p.add_argument("--per-observation", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("collect", help="build the BORIS project from results")
    p.add_argument("job_dir")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--no-evidence", action="store_true",
                   help="do not write engine provenance into BORIS comments")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("run-task", help="run one engine over one shard (used by sbatch)")
    p.add_argument("job_dir")
    p.add_argument("--engine", required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--per-task", type=int, default=1)
    p.add_argument("--observation", action="append")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run_task)

    p = sub.add_parser("purge", help="delete uploaded videos (or the whole job)")
    p.add_argument("job_dir")
    p.add_argument("--videos", action="store_true", help="delete uploaded media (default)")
    p.add_argument("--all", action="store_true", help="delete the entire job directory")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_purge)

    p = sub.add_parser("jobs", help="list jobs under the jobs root")
    p.add_argument("--root", default="")
    p.add_argument("--progress", action="store_true")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("cancel", help="scancel a job's array jobs")
    p.add_argument("job_dir")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("migrate", help="convert a subjects-as-phases BORIS project")
    p.add_argument("source")
    p.add_argument("out")
    p.add_argument("--subject", action="append", help="a real subject (repeatable)")
    p.add_argument("--category", default="Episode")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("import-ethogram", help="spreadsheet -> ethogram-only .boris")
    p.add_argument("source")
    p.add_argument("out")
    p.add_argument("--subjects", default="", help="a BORIS subjects spreadsheet")
    p.add_argument("--name", default="")
    p.set_defaults(func=cmd_import_ethogram)

    p = sub.add_parser("gui", help="launch the ABC client")
    p.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
