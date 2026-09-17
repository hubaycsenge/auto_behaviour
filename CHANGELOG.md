# Changelog

All notable changes to ABC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

The client/server job manifest carries its own `PROTOCOL_VERSION`
(`abcoder/version.py`), bumped only when the on-disk contract changes
incompatibly. A server that meets a newer manifest refuses it rather than
misreading it.

## [Unreleased]

## [0.1.0] — 2026-09-17

First working version: a folder of videos in, a reviewable BORIS project out.

### Added

- **Client GUI** (PySide6) in four tabs — Videos, Ethogram, Engines, Run — with
  every server call on a worker thread so the window never blocks on an upload.
- **Server CLI** (`abc`) covering `check`, `engines`, `scan`, `probe`, `submit`,
  `status`, `collect`, `purge`, `jobs`, `cancel`, `migrate`, `import-ethogram`
  and `run-task`. The GUI drives a remote cluster by running these over SSH, so
  everything the GUI can do is also scriptable.
- **Four analysis engines**, selectable and combinable per job:
  - `vlm_vllm` — Qwen-VL via vLLM offline batch, guided JSON decoding,
    windowed frame sampling, optional self-consistency voting.
  - `vlm_llamacpp` — the same prompting served by a local `llama-server` with a
    GGUF model and mmproj projector, for GPUs vLLM cannot use.
  - `audio` — hysteresis segmentation plus CLAP zero-shot classification
    against the ethogram's own descriptions, or acoustic rules.
  - `pose` — YOLO + ByteTrack proxemics: approach, withdrawal, orientation,
    proximity, immobility.
  - `mock` — deterministic, GPU-free, for proving the round trip.
- **Multi-engine fusion** with a per-behaviour ownership table, pre-filled from
  the wording of your ethogram.
- **SLURM submission** as one array job per engine, sharded across videos, with
  per-engine partition, GRES and throttle, and progress derived from result
  files rather than the queue.
- **BORIS 7.0 read/write**, including ethogram import from `.boris` and from
  BORIS's spreadsheet exports, state-event pairing, and media paths stored
  relative to the project file.
- **Automatic staging**: the client proves whether the cluster can see the
  video folder by writing a probe file and asking the server to hash it back.
  Shared filesystems copy nothing; otherwise videos upload and can be deleted
  from the server afterwards.
- **`abc migrate`** — converts projects that used the subject field to record
  trial phase into proper BORIS semantics (subjects as actors, phases as state
  behaviours in an `Episode` category).
- Documentation: user guide, architecture, engine reference, deployment notes
  and a validation protocol.
- 83 tests, runnable with no third-party dependencies.

### Known limitations

- The `vlm_vllm` engine has not yet been executed against a live model; its
  configuration and failure paths are tested, its output is not.
- The `audio` engine's rule classifier cannot separate a dog's vocalisation
  from a human voice. It rejects obvious speech and reports a calibrated
  posterior confidence, but on recordings where someone is talking, use the
  CLAP classifier.
- The `pose` engine approximates gaze with heading; confidence for orientation
  rules is capped accordingly.
- One observation may span several media files in the format, but the GUI does
  not yet group files into a multi-file observation.
