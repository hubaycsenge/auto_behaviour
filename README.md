# ABC — Automated Behaviour Coder

An automated ethological event recorder that produces **BORIS projects**. You
point it at a folder of videos and an ethogram; it codes each video on the NIPG
SLURM cluster and hands back a `.boris` file you open, check and correct in
BORIS exactly as if you had coded it by hand.

```
   client (GUI, your machine)                server (NIPG cluster)
   ─────────────────────────                 ────────────────────
   pick videos + subjects + ethogram   ──▶   job.json in ~/abc_jobs/<id>/
   choose engines                            sbatch array, one task per video
                                             engine → results/<obs>.<engine>.json
   poll progress                       ◀──   status
   fuse + write  project.boris         ◀──   results
```

One video is one observation, and **the file name without its extension is the
observation ID** — `03_30_152.mp4` becomes observation `03_30_152`.

## Quick start

```bash
# on the cluster (nipg1)
git clone <this repo> ~/abc && cd ~/abc
setup/install_server.sh                 # core; add --vllm --audio --pose as needed
source ~/abc_env/abc-env.sh
abc check                               # what can this node run?

# on the machine you drive it from (can be the same one)
setup/install_client.sh                 # add --ssh if it has no /nas mount
bin/abc-gui
```

Then, in the GUI: **Videos** → choose the folder · **Ethogram** → import your
`.boris` or BORIS spreadsheet and name your subjects · **Engines** → tick
`mock` first and run it once end to end · **Run** → Submit, then Save BORIS
project.

To check a fresh install without opening the GUI, the same round trip runs from
the shell — scan, submit to SLURM, collect, and verify that the media paths in
the resulting project resolve. Point it at any folder of videos; the `mock`
engine never looks at the pixels.

```bash
examples/reproduce_reference_workflow.sh /path/to/videos
```

**New here?** [`docs/user-guide.md`](docs/user-guide.md) is the step-by-step
walkthrough, from laying out your videos to reviewing the coding in BORIS.

## Where the videos live

ABC works this out by experiment, not configuration: the client writes a probe
file into the source folder and asks the cluster to read it back.

| | |
|---|---|
| **Cluster can see the folder** (anything under `/nas/home`) | Nothing is copied. The engines read your videos in place. The *Delete videos from server* button is disabled — the path on the server *is* your original folder. |
| **Cluster cannot see it** (a laptop with no mount) | The selected videos are uploaded into the job directory when you submit, and *Delete videos from server* removes those copies. Your originals are never touched. |

The `.boris` file is always written on the client, with media paths **relative
to the project file**, so the project and its video folder move together.

## Engines

Selectable per job; several can run together, with a table saying which one is
authoritative for each behaviour.

| Engine | What it does | Where it runs |
|---|---|---|
| `mock` | Deterministic fake events. Proves the whole round trip without spending GPU time. **Run this first.** | anywhere |
| `vlm_vllm` | Qwen3-VL / Qwen2.5-VL over sampled frames, strict JSON via guided decoding, temperature 0. The primary zero-training coder. | nipg38 (A100), nipg10, nipg32 — compute capability ≥ 8.0 |
| `vlm_llamacpp` | The same prompting served by a local `llama-server` with a GGUF model + mmproj projector. | nipg6/30/31/33 (2080 Ti), nipg7/34-36 (Titan RTX), nipg3-5 (GTX 1080) |
| `audio` | Segments the soundtrack and classifies vocalisations — zero-shot with CLAP against your own ethogram descriptions, or acoustic rules. | anywhere; CLAP prefers a GPU |
| `pose` | YOLO + ByteTrack → distance, approach/withdrawal, orientation, immobility. Geometric, so it cannot hallucinate. | any GPU node |

`vlm_vllm` deliberately **refuses** to start on the Turing and Pascal nodes
rather than falling back to an unusably slow eager mode; use `vlm_llamacpp`
there.

## Subjects are actors, not phases

In BORIS a *subject* is who performed the behaviour. A project that puts the
phase of a trial in the subject field cannot use BORIS's per-subject time
budgets, and gives an engine no way to know which animal to watch.

ABC models trial phases the intended way — **state behaviours in an `Episode`
category**. Select them in the Ethogram tab and press *Mark as trial phase*;
they are then coded with no focal subject, which is what they are. To fix an
existing project:

```bash
abc migrate old.boris new.boris --subject Dog
```

## Command line

Everything the GUI does is scriptable; the GUI reaches a remote cluster by
running these over SSH.

```bash
abc check                             # versions, GPU, which engines are installed
abc engines                           # catalogue and defaults, as JSON
abc scan  /path/to/videos             # files → observation IDs, with collisions flagged
abc submit  <job-dir> [--dry-run]     # queue the SLURM array jobs
abc status  <job-dir> --per-observation
abc collect <job-dir> -o project.boris
abc purge   <job-dir> --videos        # delete uploaded media
abc migrate old.boris new.boris --subject Dog
abc import-ethogram sheet.xlsx ethogram.boris
```

## What this does not claim

Every event ABC writes is a **suggestion**. Published zero-shot agreement
between video-LLMs and expert human coders is *fair* at best (κ ≈ 0.43 against
human–human κ ≈ 0.69–0.83), predicted timestamps are coarse (~50 mIoU on
temporal-grounding benchmarks even for large models), and video models are
documented to invent actions and mis-order them. The rule-based audio
classifier cannot tell a dog's bark from a person's voice.

So: validate on your own footage, report κ against your own coding, and treat
ABC as something that gives a human coder a first pass to correct — not as a
replacement for one. `docs/validation.md` sets out how to measure that.

## Documentation

- [`docs/user-guide.md`](docs/user-guide.md) — **start here**: the full walkthrough
- [`docs/engines.md`](docs/engines.md) — every engine, its options, and how to tune it
- [`docs/validation.md`](docs/validation.md) — measuring agreement before you trust the output
- [`docs/deployment.md`](docs/deployment.md) — installing on NIPG, hardware map, troubleshooting
- [`docs/architecture.md`](docs/architecture.md) — how the pieces fit, and the job-directory layout
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — adding an engine, house style

## Repository layout

```
abcoder/
    common/          ethogram, events, BORIS I/O, job manifest, media, config
    server/          CLI, SLURM submission, per-task runner
        engines/     mock · vlm_vllm · vlm_llamacpp · audio · pose
    client/          Qt GUI, controller, transports (shared filesystem / SSH)
bin/                 abc, abc-gui
setup/               install scripts for server, client, llama.cpp, GGUF models
docs/                user guide, engines, validation, deployment, architecture
examples/            an end-to-end shell walkthrough
tests/               86 tests; no third-party packages required
```

## Development

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -t .
```

The suite needs nothing installed — it runs on synthetic BORIS fixtures
(`tests/fixtures.py`) that reproduce the format's awkward corners, so it works
in a checkout carrying no study data. A few tests run only when real example
data is present and skip otherwise.

## Status

Version 0.1.0. The client, the server, the BORIS round trip, the SLURM
submission path, the `mock` engine and the `audio` engine have been run
end to end on a real cluster. The `vlm_vllm` engine's configuration and failure
paths are tested but it has not yet been executed against a live model — see
[`CHANGELOG.md`](CHANGELOG.md) for the full list of known limitations.

## Licence

MIT — see [`LICENSE`](LICENSE).
