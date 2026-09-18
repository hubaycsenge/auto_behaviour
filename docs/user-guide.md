# User guide

This walks through coding a folder of videos from start to finish. It assumes
ABC is installed — if not, [`deployment.md`](deployment.md) covers that first.

---

## Before you start: what ABC is and is not

ABC gives a human coder a **first pass to correct**. It is not a replacement
for coding, and the project it writes is not data you should analyse
unreviewed. Everything below is built around that: every event carries a
confidence and a note saying which method produced it and why, so reviewing in
BORIS is fast and you can see what to distrust.

If you want to know how good the first pass actually is on your footage, read
[`validation.md`](validation.md) — that is a separate, and more important,
piece of work.

---

## 1. Lay out your videos

One video becomes one observation, and **the file name without its extension
becomes the observation ID**.

```
my_study/
    03_30_152.mp4     ->  observation "03_30_152"
    04_05_162.MP4     ->  observation "04_05_162"
    04_09_169.mov     ->  observation "04_09_169"
```

So name files the way you want the observations named. Two things to watch:

- **`clip.mp4` and `clip.MP4` in the same folder collide.** Both reduce to
  `clip`, and one would overwrite the other. ABC flags this in red and refuses
  to include either until you rename one.
- **Sub-folders are off by default.** A recursive scan can find
  `session1/run.mp4` and `session2/run.mp4`, which both want to be `run`. Turn
  *Include sub-folders* on only when you know the names are unique.

If one observation spans several files (a camera that split its recording),
ABC plays them back-to-back the way BORIS does — but the GUI does not group
them for you yet, so concatenate first or code them as separate observations.

---

## 2. Start the client

```bash
bin/abc-gui
```

The status bar at the bottom tells you where the server is and what it can do:

```
Shared filesystem, jobs in ~/abc_jobs · ABC 0.1.0 on Python 3.12.3 · NVIDIA A100-SXM4-40GB (cc 8.0)
```

If it says **Not connected**, you can still build an ethogram and pick videos —
only submitting needs the cluster. Check the SSH settings in
`~/.config/abc/client.json`, or use *Server → Reconnect*.

---

## 3. Videos tab — choose the folder

Press **Choose folder…**. ABC scans it, reads each file's duration and frame
rate, and shows you the observation IDs it derived.

| column | meaning |
|---|---|
| Use | include this video in the job |
| Observation ID | the file stem — this is what appears in BORIS |
| Duration, Size | read from the file itself |
| Note | why a file cannot be used |

Set **Save project as** to a path **inside or beside the video folder**. This
matters more than it looks: ABC stores media paths *relative to the `.boris`
file*, so a project saved next to the videos can be zipped, moved to another
machine or handed to a colleague and it will still find its media. A project
saved somewhere unrelated gets absolute paths and breaks the moment anything
moves.

---

## 4. Ethogram tab — subjects and behaviours

### Import what you already have

**Import…** reads any of:

- a `.boris` project (behaviours *and* subjects come across),
- a BORIS ethogram spreadsheet export (`.xlsx`, `.csv`, `.tsv`),
- a BORIS subjects spreadsheet.

BORIS exports those as two separate sheets, so after importing behaviours ABC
offers to import a subjects sheet too.

### Subjects are actors

A **subject** is who performed the behaviour: `Dog`, `Owner`, `Robot`. Not the
phase of the session, not the condition, not the trial number.

With **no subjects declared**, every event is coded against BORIS's *"No focal
subject"*, which is exactly right for single-animal work — you do not have to
invent a subject.

### Trial phases go in the Episode category

If your design has ordered phases — robot appears, first approach sequence,
pause, second sequence — those are **state behaviours**, not subjects. Select
them and press **Mark as trial phase**: ABC makes them state events in an
`Episode` category, and codes them with *no focal subject*, because a phase
belongs to the session rather than to any animal.

> **Converting an old project.** If you have projects where the subject field
> holds the phase, *File → Convert a subjects-as-phases project…* rewrites
> them, or from the shell:
>
> ```bash
> abc migrate old.boris new.boris --subject Dog
> ```
>
> Every phase becomes a state behaviour in the `Episode` category, and every
> real behaviour event is re-attributed to the subject you name. The original
> file is not modified.

### The warning strip

The coloured strip at the top tells you what ABC thinks of the scheme. The one
to take seriously:

> *`Episode1_start` is both a behaviour code and a subject name…*

That is the subject-as-phase pattern, and it will make the coding hard to
analyse in BORIS later.

---

## 5. Engines tab — how the videos get coded

Tick one or more engines. Greyed-out entries are not installed on the server;
hover for the reason.

**Run `mock` first.** It emits deterministic fake events in about two seconds
per video, and it exercises the entire path — submission, SLURM, results,
fusion, relative paths, BORIS writing. If the `.boris` file it produces opens
in BORIS with working video, everything downstream of your real engine already
works.

| engine | use it for | needs |
|---|---|---|
| `mock` | proving the round trip | nothing |
| `vlm_vllm` | the main coding pass | an Ampere GPU (A100 / 3090 / A4000) |
| `vlm_llamacpp` | the same, on older GPUs | a GGUF model + mmproj projector |
| `audio` | vocalisations | PyAV; CLAP prefers a GPU |
| `pose` | approach, distance, orientation, immobility | any GPU |

### Options and Resources

Each engine has an **Options** tab (its own settings, as JSON) and a
**Resources** tab (what to ask SLURM for). The defaults are tuned for this
cluster. The ones you are most likely to touch:

- `vlm_vllm` → `model`, `fps`, `window_seconds`, `extra_instructions`
- `vlm_llamacpp` → `model_path` and `mmproj_path` (**both required**)
- `audio` → `classifier` (`clap` when a person is audible on the recording)
- `pose` → `robot_roi`, and the approach/proximity thresholds

Press **Apply** after editing, or the change is not kept.

### Running several engines together

This is where ABC gets good. Tick `vlm_vllm`, `audio` and `pose`, then press
**Suggest** under the ownership table. ABC reads your ethogram's wording and
assigns each behaviour to the method that can actually measure it:

| behaviour | coded by | why |
|---|---|---|
| Whine, Excited bark, Growl… | `audio` | the video model cannot hear |
| Approaching robot, Backing… | `pose` | geometry, not interpretation |
| Tucked tail, Play bow, Jumping… | `vlm_vllm` | needs a model that can see |
| Episode1_start… | `vlm_vllm` | needs interpretation of the scene |

Change any row you disagree with. Behaviours left on *"(any engine that finds
it)"* keep every engine's events, deduplicated — useful when you want to see
where two independent methods agree.

---

## 6. Run tab — submit and wait

### Check where your videos are

The strip at the top says one of two things:

> **The cluster can read your video folder directly, so nothing will be
> copied.** Best case. Zero transfer, and *Delete videos from server* is
> disabled on purpose — the path on the server *is* your folder, and deleting
> it would delete your originals.

> **The cluster cannot see your video folder, so the selected files will be
> uploaded.** ABC copies the videos into the job directory when you submit.
> *Delete videos from server* removes those copies afterwards.

ABC decides this by experiment, not configuration: it writes a small probe file
into your folder and asks the cluster to read it back.

### Run options

- **Videos per task** — raise it when the model takes longer to load than to
  run, or when you have more than ~1000 videos (the cluster's array limit).
- **Minimum confidence** — leave at 0 for a first pass. A low-confidence event
  a human can reject costs less than a missed one they never see.
- **Provenance** — keep on. It writes `[engine confidence]` and the engine's
  one-line reasoning into the BORIS comment field, which is what makes the
  coding reviewable.

### Submit

**Check without submitting** validates the job and writes the sbatch scripts
without queueing anything. Worth doing once.

**Submit to the cluster** queues one array job per engine. The progress bar
counts finished observation-engine pairs; the table shows per-observation
state; the log shows what the server said.

You can close the client. The job keeps running, and *Server → Re-open a
previous job…* picks it back up.

---

## 7. Save the BORIS project

Press **Save BORIS project** once results are in. You do not have to wait for
all of them — ABC writes what exists and tells you what was skipped.

You will get a summary like:

```
12 observation(s), 431 event(s)
Saved to /data/my_study/coded.boris
```

Then open it:

```bash
boris /data/my_study/coded.boris
```

### What you will see in BORIS

- your ethogram and subjects, unchanged;
- one observation per video, with the video already linked;
- events on the timeline, each with a comment like
  `[vlm_vllm 0.82] | dog moves toward the robot with its head lowered`;
- trial phases as state events with no focal subject;
- a project description recording the ABC version, job ID, engines and the
  caveat that all of this is machine-suggested.

### Reviewing

Work through it as you would your own coding. Useful habits:

- Sort or filter by the confidence in the comment; the low ones are where the
  errors are.
- Treat every timestamp as approximate. Video models place event boundaries
  coarsely — nudge them rather than trusting them.
- Delete freely. A first pass is useful even at 60% precision if rejecting is
  faster than finding.

Save from BORIS as normal. ABC never touches the file again.

---

## 8. Clean up

If your videos were uploaded, press **Delete videos from server** once the
project is saved. Your originals are untouched; the coding results stay on the
server, so you can re-save the project later — but re-running an engine would
need another upload.

To remove a whole job directory including its results:

```bash
abc purge <job-dir> --all
```

---

## Doing it from the shell

Everything above is scriptable. See
[`examples/reproduce_reference_workflow.sh`](../examples/reproduce_reference_workflow.sh)
for a working end-to-end script, and:

```bash
abc check                             # what can this machine run?
abc scan  /path/to/videos             # files -> observation IDs

# build a job: the command-line equivalent of tabs 1-3
abc new-job --videos /path/to/videos \
            --ethogram ethogram.xlsx \
            --subject Dog --subject Owner \
            --episode-prefix Episode \
            --engine vlm_llamacpp --own

abc submit  <job-dir> [--dry-run]
abc status  <job-dir> --per-observation
abc collect <job-dir> -o project.boris
abc purge   <job-dir> --videos
```

---

## Troubleshooting

**"No output path for the BORIS project."** Set *Save project as* on the
Videos tab.

**A file shows a red note.** Usually an observation-ID collision — two files
with the same stem. Rename one.

**Durations show as `-`.** No media backend on the client: `pip install av`.
ABC still works, but BORIS will not draw a timeline correctly without them.

**An engine is greyed out.** It is not installed on the server. `abc check` on
the *target node* — not the login node — says what is missing.

**The job finishes but observations show "failed".** Open the log pane; the
error is usually a missing model file or a GPU the engine refuses to run on.
`<job-dir>/logs/` has the full output.

**BORIS asks me to locate the video files.** The project was saved somewhere
the relative path does not reach. Save it inside or beside the video folder.

**The audio engine found hundreds of vocalisations.** Something else is making
noise — usually a person talking. The rule classifier cannot separate a bark
from a human voice. Switch that engine's `classifier` option to `"clap"`, and
raise `min_confidence`.
