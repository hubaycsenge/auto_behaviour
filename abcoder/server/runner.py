"""The per-task worker: ``python -m abcoder.server.runner``.

One invocation is one SLURM array task. It reads the job manifest, works out
which observations belong to its shard, loads its engine once, codes each
observation in turn, and writes one result file per observation.

Everything is written atomically and nothing is ever read back from a previous
task, so tasks are independent: re-running a shard overwrites its own results
and touches nothing else. That is what makes a failed shard safe to re-queue.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

from ..common import media as media_mod
from ..common.events import MediaInfo, ObservationResult
from ..common.jobspec import JobLayout, JobSpec
from .engines import EngineContext, build_engine
from .slurm import plan_shards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="abcoder.server.runner",
        description="Run one ABC engine over one shard of a job's videos.",
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--per-task", type=int, default=1)
    parser.add_argument("--observation", action="append", default=None,
                        help="code only these observation IDs (repeatable); "
                             "overrides the shard calculation")
    parser.add_argument("--force", action="store_true",
                        help="recode observations that already have a result file")
    args = parser.parse_args(argv)

    layout = JobLayout(args.job_dir)
    if not layout.manifest.is_file():
        print(f"no job.json in {args.job_dir}", file=sys.stderr)
        return 2
    job = JobSpec.load(layout.manifest)
    layout.ensure()

    engine_spec = next((e for e in job.engines if e.name == args.engine), None)
    if engine_spec is None:
        print(f"engine {args.engine!r} is not part of this job", file=sys.stderr)
        return 2

    if args.observation:
        wanted = set(args.observation)
        shard = [o for o in job.observations if o.observation_id in wanted]
    else:
        shards = plan_shards(len(job.observations), args.per_task)
        if not 0 <= args.shard_index < len(shards):
            print(f"shard {args.shard_index} is outside 0..{len(shards) - 1}", file=sys.stderr)
            return 2
        start, stop = shards[args.shard_index]
        shard = job.observations[start:stop]

    if not shard:
        print("nothing to do for this shard")
        return 0

    print(f"[abc] engine={args.engine} shard={args.shard_index} "
          f"observations={len(shard)} backend={media_mod.BACKEND}", flush=True)

    ctx = EngineContext(
        ethogram=job.ethogram,
        options=dict(engine_spec.options),
        behaviors=list(engine_spec.behaviors),
        job_dir=str(layout.root),
        cache_dir=os.environ.get("ABC_CACHE", str(pathlib.Path("~/.cache/abc").expanduser())),
    )
    engine = build_engine(args.engine, ctx)

    failures = 0
    try:
        # Model loading happens here, once, rather than per video. A setup
        # failure is written into every observation's result file so the client
        # shows the reason instead of an unexplained gap.
        try:
            engine.setup()
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            print(f"[abc] engine setup failed: {message}", file=sys.stderr, flush=True)
            for obs in shard:
                result = ObservationResult(
                    observation_id=obs.observation_id, failed=True,
                    error=f"engine {args.engine} could not start: {message}",
                    engine={"name": args.engine},
                    diagnostics={"traceback": traceback.format_exc()},
                )
                _write(layout, obs.observation_id, args.engine, result)
            return 1

        for i, obs in enumerate(shard, 1):
            out_path = layout.result_file(obs.observation_id, args.engine)
            if out_path.is_file() and not args.force:
                print(f"[abc] ({i}/{len(shard)}) {obs.observation_id}: already done, skipping",
                      flush=True)
                continue

            media = _media_for(obs)
            started = time.time()
            print(f"[abc] ({i}/{len(shard)}) {obs.observation_id}: "
                  f"{len(media)} file(s), {obs.duration:.0f}s", flush=True)

            result = engine.run(obs.observation_id, media)
            _write(layout, obs.observation_id, args.engine, result)

            if result.failed:
                failures += 1
                print(f"[abc]     FAILED: {result.error}", file=sys.stderr, flush=True)
            else:
                print(f"[abc]     {len(result.events)} event(s) in "
                      f"{time.time() - started:.1f}s"
                      + (f", {len(result.warnings)} warning(s)" if result.warnings else ""),
                      flush=True)
    finally:
        engine.teardown()

    print(f"[abc] shard finished: {len(shard) - failures} ok, {failures} failed", flush=True)
    return 1 if failures == len(shard) else 0


def _media_for(obs) -> list[MediaInfo]:
    """Build the MediaInfo list for an observation, probing anything unknown.

    The client probes media before submitting, so this is usually a no-op; it
    matters when a job.json was written by hand or the client had no codecs.
    """
    by_path = {m.path: m for m in obs.info}
    out: list[MediaInfo] = []
    for path in obs.server_media:
        info = by_path.get(path)
        if info is None or info.duration <= 0:
            try:
                info = media_mod.probe(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[abc]     could not probe {path}: {exc}", file=sys.stderr, flush=True)
                info = info or MediaInfo(path=path)
        # The manifest's path is what the server must actually open.
        out.append(MediaInfo(**{**info.to_dict(), "path": path}))
    return out


def _write(layout: JobLayout, obs_id: str, engine: str, result: ObservationResult) -> None:
    path = layout.result_file(obs_id, engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
