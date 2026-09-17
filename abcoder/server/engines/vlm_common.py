"""Prompting and parsing shared by the two video-LLM engines.

Both engines ask the same question of the same kind of model; they differ only
in how the model is served (vLLM in-process on Ampere, a llama.cpp server on
Turing/Pascal). Everything that decides annotation *quality* lives here, so the
two stay comparable and a prompt change applies to both.

Three things in this file do most of the work:

* the behaviour enum in :func:`events_schema` -- with guided decoding, a model
  physically cannot emit a behaviour that is not in the ethogram, which removes
  the single largest source of unusable output;
* absolute timestamps on every frame, which is what lets the model report event
  times in video time instead of window time;
* windowing with overlap, so a long session is coded at a sane frame density
  instead of being sampled ever more thinly.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...common.ethogram import Ethogram
from ...common.events import Event
from ...common.media import Frame, sample_times

SYSTEM_PROMPT = (
    "You are an expert ethologist coding video for a behavioural research "
    "study. You annotate only what is visible in the frames you are shown. "
    "You never infer an event from expectation, from the order events usually "
    "occur in, or from a single ambiguous frame. When you are unsure, you "
    "either omit the event or report it with low confidence. You answer with "
    "JSON only."
)


@dataclass
class Window:
    """One slice of a video handed to the model in a single request."""

    index: int
    start: float
    stop: float
    frames: list[Frame]

    @property
    def duration(self) -> float:
        return self.stop - self.start


def plan_windows(
    duration: float,
    window_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    """Split ``[0, duration]`` into overlapping windows.

    The overlap exists so that a behaviour straddling a boundary is seen whole
    at least once; :func:`~abcoder.common.events.deduplicate` collapses the
    resulting double reports.
    """
    if duration <= 0:
        return []
    window_seconds = max(1.0, float(window_seconds))
    overlap = max(0.0, min(float(overlap_seconds), window_seconds / 2))
    if duration <= window_seconds:
        return [(0.0, duration)]

    step = window_seconds - overlap
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        stop = min(duration, start + window_seconds)
        out.append((round(start, 3), round(stop, 3)))
        if stop >= duration:
            break
        start += step
    return out


def window_frame_times(
    start: float, stop: float, fps: float, max_frames: int
) -> list[float]:
    return sample_times(start, stop, fps, max_frames)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def events_schema(ethogram: Ethogram, behaviors: Sequence[str] | None = None) -> dict[str, Any]:
    """JSON schema for one window's answer, with the ethogram baked in.

    Passed to vLLM as ``guided_json`` / to llama.cpp as a GBNF-backed
    ``json_schema`` response format. The ``enum`` constraints are the point:
    they make an invented behaviour code or an unknown subject impossible
    rather than merely unlikely.
    """
    codes = list(behaviors) if behaviors else ethogram.codes
    subjects = ethogram.subject_names or [""]
    state_codes = {b.code for b in ethogram.state_behaviors()}

    return {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "behavior": {"type": "string", "enum": codes},
                        "subject": {"type": "string", "enum": subjects},
                        "start": {"type": "number"},
                        "stop": {"type": ["number", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "string", "maxLength": 200},
                    },
                    "required": ["behavior", "subject", "start", "stop", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["events"],
        "additionalProperties": False,
        # Not part of the constraint, but carried along so the engine can tell
        # states from points without re-deriving it.
        "x-state-behaviors": sorted(state_codes),
    }


def schema_for_decoding(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip ABC's private ``x-`` keys before handing the schema to a decoder."""
    return {k: v for k, v in schema.items() if not k.startswith("x-")}


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

def build_prompt(
    ethogram: Ethogram,
    window: Window,
    *,
    behaviors: Sequence[str] | None = None,
    video_duration: float = 0.0,
    observation_id: str = "",
    extra_instructions: str = "",
) -> str:
    """The text half of one request. Frames are attached by the caller."""
    etho_text = ethogram.describe_for_prompt(behaviors)
    subjects = ethogram.subject_names
    frame_times = ", ".join(f"{f.time:.2f}" for f in window.frames)

    if subjects:
        subject_block = (
            "SUBJECTS -- every event must be attributed to exactly one of these:\n"
            + "\n".join(
                f'- "{s.name}"' + (f" -- {s.description}" if s.description else "")
                for s in ethogram.subjects
            )
            + "\n\nIf you cannot tell which subject performed a behaviour, pick the "
              "most likely one and lower the confidence accordingly."
        )
    else:
        subject_block = (
            'SUBJECTS: this project codes a single actor, so set "subject" to the '
            "empty string for every event."
        )

    session_level = [b.code for b in ethogram.episode_behaviors()]
    session_block = ""
    if session_level:
        session_block = (
            "\nSESSION PHASES\n"
            "These behaviours describe what phase the session is in, not what an "
            "individual does: " + ", ".join(f'"{c}"' for c in session_level) + ". "
            "They do not overlap, they cover the session in order, and their "
            'subject must be the empty string.\n'
        )

    return f"""You are coding an ethogram for observation "{observation_id}".

You are shown {len(window.frames)} frames sampled from a {video_duration:.0f}-second
video, covering the interval from {window.start:.2f}s to {window.stop:.2f}s.
The frames are in chronological order and were taken at these times, in seconds
from the start of the whole video:
{frame_times}

ETHOGRAM -- the only behaviours you may report:
{etho_text}

{subject_block}
{session_block}
RULES
1. Report every occurrence you can see in these frames, and nothing else.
2. Times are in SECONDS FROM THE START OF THE WHOLE VIDEO, matching the frame
   times listed above. They must fall between {window.start:.2f} and {window.stop:.2f}.
3. A POINT behaviour has "stop": null. A STATE behaviour must have a numeric
   "stop" greater than its "start". If a state is still ongoing at the last
   frame, set "stop" to {window.stop:.2f}.
4. "confidence" is your honest probability that the event is really there,
   between 0 and 1. Use values below 0.5 freely; a low-confidence event is more
   useful than a missing one, and far more useful than a confident guess.
5. "evidence" is at most one short sentence naming what in the frames shows the
   behaviour. Do not restate the definition.
6. Frames are sampled, not continuous. Do not report a behaviour you merely
   expect to have happened between two frames.
7. If nothing in the ethogram occurs in this interval, return {{"events": []}}.
{extra_instructions}
Answer with JSON matching this shape and nothing else:
{{"events": [{{"behavior": "...", "subject": "...", "start": 0.0, "stop": null,
              "confidence": 0.0, "evidence": "..."}}]}}"""


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(
    text: str,
    ethogram: Ethogram,
    window: Window,
    *,
    min_confidence: float = 0.0,
    source: str = "",
) -> tuple[list[Event], list[str]]:
    """Turn one model response into events.

    Tolerant of the ways a model breaks the format even under guided decoding:
    fenced code blocks, a preamble, a trailing explanation, or a bare list
    instead of the wrapper object. Returns the events plus warnings describing
    anything that had to be repaired.
    """
    warnings: list[str] = []
    payload = _extract_json(text)
    if payload is None:
        return [], [f"window {window.index}: response was not JSON ({text[:120]!r})"]

    items = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return [], [f"window {window.index}: no event list in response"]

    state_codes = {b.code for b in ethogram.state_behaviors()}
    out: list[Event] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        behavior = str(item.get("behavior", "")).strip()
        if not behavior:
            continue
        try:
            start = float(item.get("start"))
        except (TypeError, ValueError):
            warnings.append(f"window {window.index}: {behavior!r} had no usable start time")
            continue
        if not math.isfinite(start):
            continue

        stop_raw = item.get("stop", None)
        stop: float | None
        try:
            stop = None if stop_raw is None else float(stop_raw)
        except (TypeError, ValueError):
            stop = None
        if stop is not None and not math.isfinite(stop):
            stop = None

        is_state = behavior in state_codes
        if is_state and stop is None:
            stop = window.stop
            warnings.append(
                f"window {window.index}: state {behavior!r} had no offset; closed at window end"
            )
        if not is_state and stop is not None:
            stop = None  # a point event cannot have a duration

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        if confidence < min_confidence:
            continue

        # Clamp into the window. A model that reports 12.0 when the window runs
        # 120-240 has answered in window-relative time; shift it rather than
        # discard the event, but say so.
        if start < window.start - 1.0 or start > window.stop + 1.0:
            shifted = window.start + start
            if window.start <= shifted <= window.stop:
                warnings.append(
                    f"window {window.index}: {behavior!r} at {start:.2f}s looked "
                    f"window-relative; shifted to {shifted:.2f}s"
                )
                if stop is not None:
                    stop = window.start + stop
                start = shifted
            else:
                warnings.append(
                    f"window {window.index}: dropped {behavior!r} at {start:.2f}s, "
                    f"outside the window [{window.start:.2f}, {window.stop:.2f}]"
                )
                continue

        start = min(max(start, window.start), window.stop)
        if stop is not None:
            stop = min(max(stop, start), window.stop)

        out.append(Event(
            behavior=behavior,
            start=round(start, 3),
            stop=None if stop is None else round(stop, 3),
            subject=str(item.get("subject", "") or ""),
            confidence=round(confidence, 3),
            evidence=str(item.get("evidence", "") or "")[:200],
            source=source,
        ))

    return out, warnings


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# --------------------------------------------------------------------------
# self-consistency
# --------------------------------------------------------------------------

def vote(
    samples: Sequence[Sequence[Event]],
    *,
    tolerance: float = 1.5,
    min_votes: int = 2,
) -> list[Event]:
    """Keep events that several independent samples agree on.

    Video-LLMs hallucinate actions and mis-order them; majority voting over
    repeated samples at non-zero temperature is the cheapest defence that does
    not need a second model. With a single sample this is the identity.
    """
    if len(samples) <= 1:
        return list(samples[0]) if samples else []

    clusters: list[dict[str, Any]] = []
    for events in samples:
        for e in events:
            for c in clusters:
                rep: Event = c["rep"]
                if rep.behavior == e.behavior and rep.subject == e.subject \
                        and abs(rep.start - e.start) <= tolerance:
                    c["members"].append(e)
                    break
            else:
                clusters.append({"rep": e, "members": [e]})

    out: list[Event] = []
    for c in clusters:
        members: list[Event] = c["members"]
        if len(members) < min_votes:
            continue
        starts = sorted(m.start for m in members)
        stops = sorted(m.stop for m in members if m.stop is not None)
        rep: Event = members[0]
        agreement = len(members) / len(samples)
        out.append(Event(**{
            **rep.to_dict(),
            "start": round(starts[len(starts) // 2], 3),
            "stop": round(stops[len(stops) // 2], 3) if stops else None,
            # Confidence becomes the model's own estimate tempered by how often
            # it actually reproduced the event.
            "confidence": round(
                min(1.0, (sum(m.confidence or 0.0 for m in members) / len(members)) * agreement),
                3,
            ),
            "comment": f"{len(members)}/{len(samples)} samples",
        }))
    out.sort(key=lambda e: e.start)
    return out
