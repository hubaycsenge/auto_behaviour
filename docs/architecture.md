# Architecture

## The shape of it

ABC is two halves joined by a directory.

The **client** is a Qt application that knows about your videos, your ethogram
and your BORIS project. The **server** is a set of SLURM jobs that know how to
turn a video into a list of events. They meet at a *job directory* on shared
storage: the client writes `job.json` into it, the server writes results beside
it, and neither needs the other to be running at the time.

That choice is what makes the system survivable on a cluster. A job you
submitted on Tuesday is still a readable, re-runnable directory on Friday. If
the client crashes mid-job, nothing is lost — reopen it with *Server → Re-open a
previous job*. If a node dies, its array task is re-queued and overwrites only
its own result files.

## Job directory layout

```
~/abc_jobs/20260917-093538-0ef0f6/
    job.json                      the manifest: ethogram, subjects, videos, engines
    status.json                   which SLURM arrays were submitted
    videos/                       uploaded media (only when staging = "upload")
    results/<obs>.<engine>.json   one ObservationResult per video per engine
    merged/<obs>.json             fused result, when several engines ran
    logs/<engine>-<jobid>_<task>.out
    slurm/<engine>.sbatch         the generated script, kept for inspection
```

Result files are the source of truth for progress. SLURM is consulted only to
tell "not started yet" apart from "will never start", which the filesystem
cannot say on its own.

## The data model

Four types carry everything (`abcoder/common/`):

- **`Ethogram`** — `Behavior` rows, `Subject` rows, categories. Maps losslessly
  onto BORIS's `behaviors_conf` / `subjects_conf` / `behavioral_categories`.
- **`Event`** — one coded occurrence: behaviour, subject, start, optional stop,
  confidence, the engine that produced it, and a one-line justification.
- **`JobSpec`** — the client/server contract, versioned by `PROTOCOL_VERSION`.
  A server that meets a newer manifest refuses it rather than misreading it.
- **`ObservationResult`** — what one engine produced for one video, plus
  diagnostics and warnings.

Engines emit `Event`s and nothing else. They never touch the BORIS format, so
adding an analysis method is one file.

## From engine output to a BORIS project

Raw engine output is not trusted. Every result goes through
`abcoder.common.events.clean()` in a fixed order:

1. **drop unknown behaviours** — models invent plausible-sounding categories,
   and BORIS validates behaviour codes on load, so an invented one would
   corrupt the project;
2. **coerce subjects** onto the declared list; trial-phase behaviours keep the
   empty subject, because a phase belongs to the session and attributing it to
   a dog would double-count it in BORIS's time budgets;
3. **clamp to the media** — an event past the end of the clip is worse than
   useless, since BORIS cannot seek to it;
4. **close open states** at the next onset of the same behaviour or at the end
   of the media, warning each time it had to guess;
5. **deduplicate** near-identical events, which is how the overlap between
   analysis windows is absorbed;
6. **enforce exclusivity** — BORIS refuses to load a project where two mutually
   excluded states overlap on one subject, so this must happen before writing.

When several engines ran, `fuse()` applies the ownership table first: a
behaviour with an owner keeps only that engine's events; a behaviour without one
pools every engine's and lets deduplication collapse the agreement.

`project_builder.py` then writes the project **on the client**, because the
paths inside have to describe the client's copies of the videos. A state event
becomes two BORIS rows (onset and offset, paired by ordinal position); a point
event becomes one. The sixth column is the frame index, computed from the
media's real frame rate.

## Paths

The rule: **a `.boris` file and its videos travel together**, so media paths are
stored relative to the project file and always with forward slashes. BORIS
accepts forward slashes everywhere; a backslash written on Windows makes the
project unopenable on Linux.

Absolute paths are written only when no relative route exists, and a
server-side path is never written into a client project.

Reading is more forgiving than writing: `resolve_media_path` handles the doubled
separator BORIS itself sometimes emits (`Merged//clip.mp4`) and normalises
Windows separators, so projects made elsewhere open correctly.

## Why one array job per engine

Engines want different hardware. The video-LLM wants an A100 for twelve hours;
the audio engine wants any node for twenty minutes. A single job asking for the
union would leave the cheap work queued behind the scarcest resource on the
cluster, so each engine gets its own array with its own `--partition`, `--gres`
and `--time`, and its own throttle so one job cannot fill the queue.

Within an array, one task is one video by default. Raise *Videos per task* when
the model takes longer to load than to run, or when the corpus is larger than
this cluster's `MaxArraySize` of 1001 — `plan_shards` packs into fewer, longer
tasks rather than failing at submission.

## Transports

`detect_transport` decides how the client reaches the server:

1. If `sbatch` exists locally, this *is* a cluster machine — use the filesystem
   directly and submit locally.
2. Otherwise connect over SSH, and if the source folder looks like it might be
   shared, **prove it**: write a probe file, ask the server to hash it back. A
   laptop with the NAS mounted over SMB is exactly the case worth catching, and
   it saves hours of uploading.
3. Failing that, upload.

A transport that cannot answer the probe is treated as not shared. That costs an
upload but is never wrong in the dangerous direction.
