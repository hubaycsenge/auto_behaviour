#!/usr/bin/env bash
# The reference workflow, end to end from the command line.
#
# Codes the example footage with the mock engine and writes a BORIS project
# beside the videos. Nothing here needs a GPU -- the point is to prove the whole
# path works on a new deployment before you spend cluster time on it.
#
#   examples/reproduce_reference_workflow.sh [video-folder] [output.boris]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
videos="${1:-$root/example/Merged}"
output="${2:-$videos/abc_coded.boris}"

if [ ! -d "$videos" ]; then
  cat >&2 <<MSG
No video folder at: $videos

This script needs some videos to code. A public checkout ships no study data,
so point it at your own:

    examples/reproduce_reference_workflow.sh /path/to/videos [output.boris]

Any folder of .mp4/.mov/.avi files will do -- the mock engine does not look at
the pixels, it only proves that submission, SLURM, collection and the BORIS
writer all work end to end.
MSG
  exit 2
fi
python="${ABC_PYTHON:-python3}"
jobs_root="${ABC_JOBS_ROOT:-$HOME/abc_jobs}"

export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
scan_json="$work/scan.json"
submit_json="$work/submit.json"
status_json="$work/status.json"

echo "== 1. what can this machine run? =="
"$root/bin/abc" check > "$work/check.json"
head -20 "$work/check.json"

echo
echo "== 2. the videos, and the observation IDs they imply =="
"$root/bin/abc" scan "$videos" > "$scan_json"
"$python" - "$scan_json" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
for row in d["files"][:5]:
    print(f'   {row["observation_id"]:<16} {row["path"].rsplit("/", 1)[-1]}')
print(f'   ... {len(d["files"])} file(s) total')
if d["duplicate_observation_ids"]:
    print("   COLLISIONS:", list(d["duplicate_observation_ids"]))
PY

echo
echo "== 3. build the job =="
job_dir="$("$python" - "$videos" "$jobs_root" "$root" <<'PY'
import pathlib, sys
from abcoder.common.boris import ethogram_from_table
from abcoder.common.config import engine_defaults, suggest_ownership
from abcoder.common.ethogram import EPISODE_CATEGORY, Subject
from abcoder.common.jobspec import (STAGING_SHARED, EngineSpec, JobLayout, JobSpec,
                                    ObservationSpec, SlurmSpec)
from abcoder.common import media as M

videos = pathlib.Path(sys.argv[1])
jobs_root = pathlib.Path(sys.argv[2]).expanduser()
root = pathlib.Path(sys.argv[3])

# The ethogram, with subjects used the way BORIS means them: actors, not phases.
# A public checkout ships no study data, so fall back to the synthetic fixture
# the test suite uses -- same shape, no one's recordings in it.
sheet = root / "example" / "ethogram.xlsx"
if not sheet.is_file():
    sys.path.insert(0, str(root))
    from tests import fixtures
    sheet = fixtures.shared()["ethogram_xlsx"]
    print("   (using the synthetic fixture ethogram; example/ is not present)",
          file=sys.stderr)
etho = ethogram_from_table(sheet)
etho.subjects = [Subject("Dog", "the tested dog"), Subject("Owner", "the dog owner")]
for b in etho.behaviors:
    if b.code.lower().startswith("episode"):
        b.category = EPISODE_CATEGORY
etho.categories.append(EPISODE_CATEGORY)

job = JobSpec(project_name="ABC reference workflow", ethogram=etho,
              staging=STAGING_SHARED, client_source_dir=str(videos))
for path in M.discover_videos(videos)[:4]:
    try:
        info = M.probe(path)
    except Exception:
        info = None
    job.observations.append(ObservationSpec(
        observation_id=M.observation_id_for(path),
        server_media=[str(path.resolve())], client_media=[str(path.resolve())],
        info=[info] if info else []))

defaults = engine_defaults("mock")
job.engines = [EngineSpec("mock", options=defaults["options"],
                          slurm=SlurmSpec.from_dict(defaults["slurm"]))]
job.ownership = suggest_ownership(etho.codes, ["mock"])
job.job_dir = str(jobs_root / job.job_id)
JobLayout(job.job_dir).ensure()
job.save(pathlib.Path(job.job_dir) / "job.json")

problems = job.validate()
if problems:
    raise SystemExit("job is not valid: " + "; ".join(problems))
print(job.job_dir)
PY
)"
echo "   job: $job_dir"

echo
echo "== 4. submit =="
if command -v sbatch >/dev/null; then
  "$root/bin/abc" submit "$job_dir" --python "$python" --abc-root "$root" \
      > "$submit_json"
  "$python" - "$submit_json" <<'PY'
import json, pathlib, sys
for a in json.loads(pathlib.Path(sys.argv[1]).read_text())["submitted"]:
    print(f'   {a["engine"]}: slurm job {a["job_id"]}, {a["n_tasks"]} task(s)')
PY
  echo "   waiting for results..."
  total="$("$python" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['observations']))" "$job_dir/job.json")"
  # `ls` on an empty glob exits 2, and this script runs under `set -o pipefail`,
  # so count with find instead -- it succeeds when nothing matches yet.
  for _ in $(seq 1 60); do
    ready="$(find "$job_dir/results" -name '*.json' -type f | wc -l)"
    echo "   $ready/$total"
    [ "$ready" -ge "$total" ] && break
    sleep 5
  done
else
  echo "   no sbatch here; running the tasks locally instead"
  total="$("$python" -c "import json;print(len(json.load(open('$job_dir/job.json'))['observations']))")"
  for i in $(seq 0 $((total - 1))); do
    "$python" -m abcoder.server.runner --job-dir "$job_dir" --engine mock \
        --shard-index "$i" --per-task 1 >/dev/null
  done
fi

echo
echo "== 5. status =="
"$root/bin/abc" status "$job_dir" > "$status_json"
"$python" - "$status_json" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(f'   {d["done"]} done, {d["failed"]} failed, {d["pending"]} pending')
PY

echo
echo "== 6. write the BORIS project =="
"$root/bin/abc" collect "$job_dir" -o "$output"

echo
echo "== 7. check what came out =="
"$python" - "$output" <<'PY'
import pathlib, sys
from abcoder.common.boris import BorisProject, resolve_media_path

path = pathlib.Path(sys.argv[1])
project = BorisProject.load(path)
print(f"   {len(project.observation_ids)} observation(s), "
      f"{len(project.ethogram)} behaviour(s), "
      f"subjects {project.ethogram.subject_names}")
obs = project.observation_ids[0]
stored = project.observation_media(obs)[0]
resolved = resolve_media_path(stored, path)
print(f"   media stored as  {stored!r}  (relative: {not stored.startswith('/')})")
print(f"   resolves to      {resolved}  (exists: {resolved.exists()})")
events = project.observation_events(obs)
print(f"   {obs}: {len(events)} event(s); first three:")
for e in events[:3]:
    subject = e.subject or "no focal subject"
    span = f"{e.start:.2f}" + (f"-{e.stop:.2f}" if e.stop is not None else "")
    print(f"      {span:>16}  {e.behavior}  [{subject}]")
PY

echo
echo "Open it:  boris $output"
