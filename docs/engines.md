# Engines

An engine takes one media file and an ethogram and returns a list of events.
Everything else — SLURM, staging, BORIS, fusion — happens around it. To add one,
subclass `abcoder.server.engines.base.Engine`, implement `analyse()`, and
register it:

```python
from abcoder.server.engines import register
register("simba", "mylab.simba_engine:SimbaEngine")
```

Load the model in `setup()`, not in `analyse()`: `setup()` is called once per
SLURM array task, `analyse()` once per video.

---

## `mock`

Deterministic pseudo-events from a hash of the observation ID. No GPU, no
model, no network. **Run it first on any new deployment** — it exercises
submission, sharding, result collection, fusion, path relativisation and BORIS
writing, all in about two seconds per video.

---

## `vlm_vllm` — video-LLM via vLLM

The primary zero-training coder. Samples frames with their timestamps, hands
them to a Qwen-VL model, and constrains the answer to strict JSON.

**Hardware.** vLLM needs compute capability ≥ 7.0, and bfloat16 needs ≥ 8.0. On
NIPG that is nipg38 (A100), nipg10 (3090) and nipg32 (A4000). The engine
*refuses* to start on Turing rather than falling back to `--enforce-eager`,
which runs below 10 tok/s; Qwen3-VL's vision path does not work on Turing under
vLLM at all. Use `vlm_llamacpp` there.

**Key options**

| option | default | what it controls |
|---|---|---|
| `model` | `Qwen/Qwen3-VL-8B-Instruct` | any video-capable model vLLM supports |
| `fps` | `1.0` | sampling rate within a window |
| `max_frames_per_window` | `96` | hard cap per request |
| `window_seconds` | `120` | how much video one request covers |
| `window_overlap_seconds` | `5` | so a behaviour on a boundary is seen whole once |
| `max_image_side` | `768` | long-side cap for each frame |
| `guided_json` | `true` | constrain output to the ethogram's behaviour enum |
| `self_consistency_samples` | `1` | >1 samples repeatedly and keeps the majority |
| `min_confidence` | `0.0` | drop events the model is less sure of than this |
| `extra_instructions` | `""` | appended to the prompt — the place for study-specific guidance |

**Why windows.** Accuracy rises with frame count to roughly 96–256 frames and
then *degrades*: more frames add visual noise at near-linear cost. A 500-second
session at 1 fps would be 500 frames in one request, well past that. Windowing
keeps density where it works and lets deduplication absorb the overlap.

**Why the enum matters.** With guided decoding the behaviour field is an `enum`
of your ethogram's codes, so the model physically cannot emit a category you did
not define. That removes the single largest source of unusable output.

**Tuning.** Fine, fast behaviours (a shake-off, a play bow) want higher `fps`
over shorter windows. Locating rare events in long footage wants the opposite.
If throughput is short, drop to `Qwen3-VL-30B-A3B` or the 8B dense model, or add
A100 workers.

---

## `vlm_llamacpp` — video-LLM via llama.cpp

Identical prompting, served by a local `llama-server`. This is what unlocks the
~31 GPUs vLLM cannot use.

Requires **two** files: the GGUF language weights *and* a matching `mmproj`
vision projector. llama.cpp loads the language model happily without the
projector and then never sees a frame, so ABC refuses to start unless both are
set. `setup/fetch_gguf.sh` downloads a matching pair.

```
setup/install_llamacpp.sh     # builds for sm_61;75;80;86 — one binary, every node
setup/fetch_gguf.sh           # Qwen2.5-VL-7B Q4_K_M + mmproj, ~5 GB
```

Defaults are lower than the vLLM engine's (`fps` 0.5, 24 frames, 512 px) because
throughput is lower. Start there and raise until the wall-clock hurts.

---

## `audio` — vocalisations

Frame sampling cannot hear. Five of the fourteen behaviours in the reference
ethogram are sounds, so they get their own engine, and the ownership table hands
it exactly those codes.

1. decode mono at 16 kHz;
2. segment with a **hysteresis** threshold placed between the noise floor and
   the loudest part of the recording — a percentile threshold is
   self-fulfilling, and without hysteresis an amplitude-modulated growl
   fragments into a dozen events;
3. measure F0, harmonicity, spectral centroid, bandwidth, flatness, duration;
4. classify.

**Two classifiers.**

`clap` is zero-shot, scoring each sound against **your ethogram's own
descriptions** as text prompts. No training data, and the classifier follows
your coding scheme rather than a fixed label set. Needs `torch` + `transformers`.

`rules` matches the acoustic signature of each canonical vocalisation type
(whine, high bark, low bark, growl, puff, howl, pant). It needs only numpy, so
it runs anywhere.

**A real limitation, measured.** On the reference footage the rule classifier
reported 122 "vocalisations" in 79 seconds — because a person is talking, and
F0, harmonicity, centroid, flatness and duration genuinely do not separate a
dog's bark from a human voice. Two things were added in response, and neither
of them fixes that:

- *decoy classes*: `speech` and `transient` compete for every sound and are
  never emitted, so the classifier can reject rather than always picking its
  least-bad option;
- *posterior confidence*: the number reported is the softmax posterior over all
  competing classes, so a sound that fits three classes equally cannot come out
  at 1.0. Confidence on that footage fell from a median of 1.00 to 0.45, which
  is the honest answer.

The engine also warns when detection density is implausible (>40/min). **If a
human is audible on your recordings, use `classifier: "clap"`.**

---

## `pose` — proxemics

Geometry, not interpretation: nothing here can hallucinate. YOLO detects and
ByteTrack follows the dog, the person and the robot; the engine then derives
distance, radial velocity, heading and speed and applies rules:

`approach_robot`, `approach_person`, `withdraw_robot`, `orient_robot`,
`orient_person`, `near_robot`, `near_person`, `immobile`.

Rules are matched to your ethogram by keyword — **code first, description
second**. That order matters: "Backing" is described as *"moving backwards but
not toward the owner while orienting at the agent"*, and searching the whole
string picks "orienting at the agent" and codes a retreat as an orientation.
Override with the `rules` option: `{"Backing": "withdraw_robot"}`.

**The robot.** COCO has no robot class. Either give `robot_roi` as
`[x0, y0, x1, y1]` in normalised coordinates when the robot's position is fixed,
or list stand-in detector classes. With neither, the engine codes only the rules
that do not need it and says which ones it skipped.

**Gaze.** It cannot read gaze. Every definition that turns on where the dog is
*looking* is approximated by heading — the direction of travel — confidence is
capped at 0.7 for those rules, and the evidence string on each event says so.

**Calibrate before trusting.** `approach_speed_px_per_s`, `near_fraction`,
`immobility_speed_px_per_s` and the rest are starting points for a fixed-camera
indoor setup, not constants. Code a few minutes by hand and tune against it.

---

## Running several at once

Enable several engines and use the ownership table. The arrangement both
research reviews recommend:

| behaviours | engine | why |
|---|---|---|
| vocalisations | `audio` | the VLM cannot hear |
| approach, withdrawal, orientation, proximity | `pose` | measured, not inferred |
| postures, tail, whole-body actions, trial phases | `vlm_vllm` | needs interpretation |

*Suggest* in the Engines tab fills this in from the wording of your ethogram.
Behaviours left unassigned keep every engine's events, deduplicated — useful
when you want to see where two methods agree.
