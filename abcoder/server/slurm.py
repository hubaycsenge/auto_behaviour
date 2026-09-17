"""SLURM submission and monitoring.

One array job per engine, one array task per shard of the video list. That is
the arrangement the deployment research recommends for corpus processing, and
it is the one that survives contact with a shared cluster: tasks are
embarrassingly parallel, a failed shard is re-queued on its own, and no
long-lived server has to stay up between jobs.

Engines get their own array job rather than sharing one because they want
different hardware -- the VLM wants an A100 for twelve hours, the audio engine
wants any node for twenty minutes -- and asking for the union of both would
leave the audio work queued behind the scarcest resource on the cluster.
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..common.jobspec import EngineSpec, JobLayout, JobSpec, SlurmSpec

#: sacct states that mean the task will never produce a result.
TERMINAL_FAILURE_STATES = {
    "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED",
}
TERMINAL_STATES = TERMINAL_FAILURE_STATES | {"COMPLETED"}


class SlurmError(RuntimeError):
    """A slurm command failed or could not be found."""


@dataclass
class SubmittedArray:
    """One submitted array job."""

    engine: str
    job_id: str
    n_tasks: int
    script: str
    submitted: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "job_id": self.job_id, "n_tasks": self.n_tasks,
                "script": self.script, "submitted": self.submitted}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SubmittedArray:
        return cls(engine=raw["engine"], job_id=raw["job_id"],
                   n_tasks=int(raw.get("n_tasks", 1)), script=raw.get("script", ""),
                   submitted=float(raw.get("submitted", 0)))


# --------------------------------------------------------------------------
# sharding
# --------------------------------------------------------------------------

def plan_shards(n_observations: int, per_task: int, max_tasks: int = 1000) -> list[tuple[int, int]]:
    """Split the observation list into ``(start, stop)`` index ranges.

    ``MaxArraySize`` on this cluster is 1001, so a corpus larger than that many
    tasks is packed into fewer, longer tasks rather than rejected at submission.
    """
    if n_observations <= 0:
        return []
    per_task = max(1, int(per_task))
    n_tasks = (n_observations + per_task - 1) // per_task
    if n_tasks > max_tasks:
        per_task = (n_observations + max_tasks - 1) // max_tasks
        n_tasks = (n_observations + per_task - 1) // per_task
    return [(i * per_task, min(n_observations, (i + 1) * per_task)) for i in range(n_tasks)]


# --------------------------------------------------------------------------
# script generation
# --------------------------------------------------------------------------

def render_sbatch(
    job: JobSpec,
    engine: EngineSpec,
    layout: JobLayout,
    shards: Sequence[tuple[int, int]],
    *,
    python: str,
    abc_root: str,
    hf_home: str = "",
    per_task: int = 1,
) -> str:
    """Generate the sbatch script for one engine's array job."""
    s: SlurmSpec = engine.slurm
    name = f"abc-{engine.name}-{job.job_id[:8]}"
    throttle = max(1, int(s.array_throttle))
    array = f"0-{len(shards) - 1}%{throttle}" if len(shards) > 1 else "0-0"

    directives = [
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={s.partition}",
        f"#SBATCH --array={array}",
        f"#SBATCH --cpus-per-task={int(s.cpus_per_task)}",
        f"#SBATCH --mem={s.mem}",
        f"#SBATCH --time={s.time}",
        f"#SBATCH --output={layout.logs}/{engine.name}-%A_%a.out",
        f"#SBATCH --error={layout.logs}/{engine.name}-%A_%a.err",
    ]
    if s.gres:
        directives.append(f"#SBATCH --gres={s.gres}")
    if s.nodelist:
        directives.append(f"#SBATCH --nodelist={s.nodelist}")
    if s.exclude:
        directives.append(f"#SBATCH --exclude={s.exclude}")
    if s.account:
        directives.append(f"#SBATCH --account={s.account}")
    if s.qos:
        directives.append(f"#SBATCH --qos={s.qos}")
    directives.extend(s.extra_sbatch)

    env_lines = [
        f"export PYTHONPATH={shlex.quote(abc_root)}:${{PYTHONPATH:-}}",
        "export PYTHONUNBUFFERED=1",
        # Each task owns its GPU; letting the libraries spawn a thread per core
        # on an 84-core node causes far more contention than it saves.
        f"export OMP_NUM_THREADS={int(s.cpus_per_task)}",
        "export TOKENIZERS_PARALLELISM=false",
    ]
    if hf_home:
        env_lines.append(f"export HF_HOME={shlex.quote(hf_home)}")
        # Weights are read once per task from shared storage; downloading them
        # concurrently from forty tasks is what actually saturates the NAS.
        env_lines.append("export HF_HUB_DISABLE_TELEMETRY=1")

    return f"""#!/bin/bash
{chr(10).join(directives)}

set -euo pipefail

echo "ABC {engine.name} | job {job.job_id} | task ${{SLURM_ARRAY_TASK_ID:-0}} | node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

{chr(10).join(env_lines)}

{shlex.quote(python)} -m abcoder.server.runner \\
    --job-dir {shlex.quote(str(layout.root))} \\
    --engine {shlex.quote(engine.name)} \\
    --shard-index "${{SLURM_ARRAY_TASK_ID:-0}}" \\
    --per-task {int(per_task)}

echo "ABC {engine.name} task ${{SLURM_ARRAY_TASK_ID:-0}} finished"
"""


# --------------------------------------------------------------------------
# submission
# --------------------------------------------------------------------------

def submit(
    job: JobSpec,
    layout: JobLayout,
    *,
    python: str,
    abc_root: str,
    hf_home: str = "",
    per_task: int = 1,
    dry_run: bool = False,
) -> list[SubmittedArray]:
    """Write and submit one array job per enabled engine."""
    layout.ensure()
    shards = plan_shards(len(job.observations), per_task)
    if not shards:
        raise SlurmError("job has no observations to submit")

    submitted: list[SubmittedArray] = []
    for engine in job.enabled_engines:
        script_text = render_sbatch(job, engine, layout, shards, python=python,
                                    abc_root=abc_root, hf_home=hf_home, per_task=per_task)
        script_path = layout.slurm / f"{engine.name}.sbatch"
        script_path.write_text(script_text, encoding="utf-8")
        script_path.chmod(0o755)

        if dry_run:
            submitted.append(SubmittedArray(engine=engine.name, job_id="DRYRUN",
                                            n_tasks=len(shards), script=str(script_path)))
            continue

        job_id = _sbatch(script_path)
        submitted.append(SubmittedArray(engine=engine.name, job_id=job_id,
                                        n_tasks=len(shards), script=str(script_path)))

    _write_status(layout, {"job_id": job.job_id, "submitted": [a.to_dict() for a in submitted],
                           "n_observations": len(job.observations),
                           "n_shards": len(shards), "per_task": per_task})
    return submitted


def _sbatch(script_path: pathlib.Path) -> str:
    try:
        proc = subprocess.run(["sbatch", "--parsable", str(script_path)],
                              capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise SlurmError("sbatch not found; is this a SLURM login node?") from exc
    if proc.returncode != 0:
        raise SlurmError(f"sbatch failed: {proc.stderr.strip() or proc.stdout.strip()}")
    # --parsable prints "jobid" or "jobid;cluster"
    return proc.stdout.strip().split(";")[0]


def cancel(job_ids: Iterable[str]) -> None:
    ids = [j for j in job_ids if j and j != "DRYRUN"]
    if not ids:
        return
    subprocess.run(["scancel", *ids], capture_output=True, text=True, check=False)


# --------------------------------------------------------------------------
# monitoring
# --------------------------------------------------------------------------

def queue_states(job_ids: Sequence[str]) -> dict[str, str]:
    """Current state of each array task still known to the controller."""
    ids = [j for j in job_ids if j and j != "DRYRUN"]
    if not ids:
        return {}
    proc = subprocess.run(
        ["squeue", "--jobs", ",".join(ids), "--noheader", "--array",
         "--format=%i|%T"],
        capture_output=True, text=True, check=False,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "|" in line:
            task, state = line.split("|", 1)
            out[task.strip()] = state.strip()
    return out


def accounting_states(job_ids: Sequence[str]) -> dict[str, str]:
    """Final state of each task from sacct, for jobs that have left the queue."""
    ids = [j for j in job_ids if j and j != "DRYRUN"]
    if not ids:
        return {}
    proc = subprocess.run(
        ["sacct", "--jobs", ",".join(ids), "--noheader", "--parsable2",
         "--format=JobID,State"],
        capture_output=True, text=True, check=False,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        task, state = parts[0].strip(), parts[1].strip().split()[0]
        # Skip the .batch/.extern step rows; only the task itself matters.
        if "." in task:
            continue
        out[task] = state
    return out


@dataclass
class JobProgress:
    """A snapshot of how far a job has got, for the client's progress view."""

    n_observations: int = 0
    n_engines: int = 0
    done: int = 0
    failed: int = 0
    pending: int = 0
    running: int = 0
    queue: dict[str, str] = field(default_factory=dict)
    finished: bool = False
    messages: list[str] = field(default_factory=list)

    @property
    def expected(self) -> int:
        return self.n_observations * self.n_engines

    @property
    def fraction(self) -> float:
        return (self.done + self.failed) / self.expected if self.expected else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations, "n_engines": self.n_engines,
            "done": self.done, "failed": self.failed, "pending": self.pending,
            "running": self.running, "queue": self.queue, "finished": self.finished,
            "expected": self.expected, "fraction": round(self.fraction, 4),
            "messages": self.messages,
        }


def progress(job: JobSpec, layout: JobLayout) -> JobProgress:
    """Work out where the job is, from result files plus the SLURM queue.

    Result files are the source of truth for what is *done*; SLURM is consulted
    only to distinguish "not started yet" from "will never start", which the
    filesystem cannot tell us.
    """
    engines = [e.name for e in job.enabled_engines]
    prog = JobProgress(n_observations=len(job.observations), n_engines=len(engines))

    for obs in job.observations:
        for engine in engines:
            path = layout.result_file(obs.observation_id, engine)
            if not path.is_file():
                prog.pending += 1
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    if json.load(fh).get("failed"):
                        prog.failed += 1
                    else:
                        prog.done += 1
            except (OSError, json.JSONDecodeError):
                # A half-written file means a task is mid-flush, not a failure.
                prog.pending += 1

    status = _read_status(layout)
    job_ids = [a["job_id"] for a in status.get("submitted", [])]
    prog.queue = queue_states(job_ids)
    prog.running = sum(1 for v in prog.queue.values() if v == "RUNNING")

    if prog.pending and not prog.queue:
        # Nothing left in the queue but results are missing: find out why, so
        # the client can show the reason instead of spinning forever.
        final = accounting_states(job_ids)
        dead = {t: s for t, s in final.items() if s in TERMINAL_FAILURE_STATES}
        if dead:
            sample = ", ".join(f"{t} {s}" for t, s in list(dead.items())[:5])
            prog.messages.append(
                f"{len(dead)} SLURM task(s) ended without writing results ({sample}). "
                f"See {layout.logs} for the reason."
            )
        elif final:
            prog.messages.append(
                "All SLURM tasks completed but some results are missing; check "
                f"{layout.logs}."
            )

    prog.finished = prog.pending == 0 or (not prog.queue and bool(status.get("submitted")))
    return prog


# --------------------------------------------------------------------------
# status file
# --------------------------------------------------------------------------

def _write_status(layout: JobLayout, extra: dict[str, Any]) -> None:
    data = _read_status(layout)
    data.update(extra)
    data["updated"] = time.time()
    layout.status.parent.mkdir(parents=True, exist_ok=True)
    tmp = layout.status.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, layout.status)


def _read_status(layout: JobLayout) -> dict[str, Any]:
    if not layout.status.is_file():
        return {}
    try:
        with open(layout.status, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def submitted_arrays(layout: JobLayout) -> list[SubmittedArray]:
    return [SubmittedArray.from_dict(a) for a in _read_status(layout).get("submitted", [])]
