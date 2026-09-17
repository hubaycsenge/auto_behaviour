"""Canonical event model shared by every engine.

An :class:`Event` is what an engine produces and what the BORIS writer
consumes. Engines never touch the BORIS format directly -- they emit these,
and :mod:`abcoder.common.boris` turns them into ``events`` rows. That keeps a
new engine to one file and guarantees that every engine's output obeys the same
sanity rules (no negative times, states that stop after they start, behaviours
that actually exist in the ethogram).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .ethogram import Ethogram

#: BORIS writes an empty subject for "No focal subject". ABC uses the same
#: convention so a project with no subjects round-trips cleanly.
NO_FOCAL_SUBJECT = ""


@dataclass
class Event:
    """One coded behaviour occurrence.

    ``stop`` is ``None`` for point events. For state events it is the offset; a
    state whose offset the engine could not determine is still emitted with
    ``stop=None`` and is closed later by :func:`close_open_states`.
    """

    behavior: str
    start: float
    stop: float | None = None
    subject: str = NO_FOCAL_SUBJECT
    modifiers: str = ""
    comment: str = ""
    confidence: float | None = None
    source: str = ""          # engine that produced it, for provenance
    evidence: str = ""        # free-text justification, kept out of BORIS comments unless asked

    def __post_init__(self) -> None:
        self.behavior = (self.behavior or "").strip()
        self.subject = (self.subject or NO_FOCAL_SUBJECT).strip()
        self.start = float(self.start)
        if self.stop is not None:
            self.stop = float(self.stop)

    @property
    def is_state(self) -> bool:
        return self.stop is not None

    @property
    def duration(self) -> float:
        return 0.0 if self.stop is None else max(0.0, self.stop - self.start)

    def overlaps(self, other: Event) -> bool:
        """True if the two events share any time, treating points as instants."""
        a0, a1 = self.start, self.stop if self.stop is not None else self.start
        b0, b1 = other.start, other.stop if other.stop is not None else other.start
        return a0 <= b1 and b0 <= a1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Event:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class MediaInfo:
    """What ABC knows about one media file."""

    path: str                 # path as the *server* saw it
    duration: float = 0.0
    fps: float = 0.0
    has_video: bool = True
    has_audio: bool = False
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MediaInfo:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class ObservationResult:
    """Everything one engine produced for one observation."""

    observation_id: str
    media: list[MediaInfo] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    engine: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failed: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "media": [m.to_dict() for m in self.media],
            "events": [e.to_dict() for e in self.events],
            "engine": self.engine,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "failed": self.failed,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ObservationResult:
        return cls(
            observation_id=raw.get("observation_id", ""),
            media=[MediaInfo.from_dict(m) for m in raw.get("media", [])],
            events=[Event.from_dict(e) for e in raw.get("events", [])],
            engine=raw.get("engine", {}),
            diagnostics=raw.get("diagnostics", {}),
            warnings=list(raw.get("warnings", [])),
            failed=bool(raw.get("failed", False)),
            error=raw.get("error", ""),
        )

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> ObservationResult:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------

def clamp_to_media(events: Sequence[Event], duration: float) -> list[Event]:
    """Drop events outside the media and trim states that run past the end.

    A VLM asked for timestamps will occasionally invent one past the end of the
    clip; BORIS shows such events but cannot seek to them, so they are worse
    than useless.
    """
    if duration <= 0:
        return list(events)
    out: list[Event] = []
    for e in events:
        if e.start > duration or e.start < 0:
            continue
        if e.stop is not None and e.stop > duration:
            e = Event(**{**e.to_dict(), "stop": duration})
        out.append(e)
    return out


def close_open_states(
    events: Sequence[Event],
    ethogram: Ethogram,
    duration: float,
    min_duration: float = 0.04,
) -> tuple[list[Event], list[str]]:
    """Give every state event an offset.

    A state whose offset is missing is closed at the onset of the next event of
    the same behaviour-and-subject, or at the end of the media. Returns the
    repaired events and a list of warnings describing what had to be guessed.
    """
    warnings: list[str] = []
    out: list[Event] = []
    by_key: dict[tuple[str, str], list[Event]] = {}

    for e in events:
        beh = ethogram.get(e.behavior)
        if beh is not None and not beh.is_state:
            out.append(e)
            continue
        if beh is None and e.stop is None:
            out.append(e)
            continue
        by_key.setdefault((e.behavior, e.subject), []).append(e)

    for (behavior, subject), group in by_key.items():
        group.sort(key=lambda x: x.start)
        for i, e in enumerate(group):
            if e.stop is None:
                nxt = group[i + 1].start if i + 1 < len(group) else duration
                stop = nxt if nxt > e.start else e.start + min_duration
                warnings.append(
                    f"state {behavior!r} ({subject or 'no focal subject'}) at "
                    f"{e.start:.2f}s had no offset; closed at {stop:.2f}s"
                )
                e = Event(**{**e.to_dict(), "stop": stop})
            elif e.stop <= e.start:
                warnings.append(
                    f"state {behavior!r} at {e.start:.2f}s ended at or before it "
                    f"started; extended to {min_duration}s"
                )
                e = Event(**{**e.to_dict(), "stop": e.start + min_duration})
            out.append(e)

    out.sort(key=lambda x: (x.start, x.behavior, x.subject))
    return out, warnings


def drop_unknown_behaviors(
    events: Sequence[Event], ethogram: Ethogram
) -> tuple[list[Event], list[str]]:
    """Remove events whose behaviour is not in the ethogram.

    Models invent plausible-sounding categories. Keeping them would silently
    corrupt the BORIS project, which validates behaviour codes on load.
    """
    kept: list[Event] = []
    warnings: list[str] = []
    invented: dict[str, int] = {}
    for e in events:
        if ethogram.has(e.behavior):
            kept.append(e)
        else:
            invented[e.behavior] = invented.get(e.behavior, 0) + 1
    for code, n in sorted(invented.items(), key=lambda kv: -kv[1]):
        warnings.append(f"dropped {n} event(s) of unknown behaviour {code!r}")
    return kept, warnings


def coerce_subjects(
    events: Sequence[Event], ethogram: Ethogram
) -> tuple[list[Event], list[str]]:
    """Force every event onto a subject the project actually declares.

    With no subjects declared, everything becomes "no focal subject" (BORIS's
    own single-actor convention). With exactly one subject, unattributed events
    go to it. Otherwise an unrecognised subject is a hard problem, so the event
    keeps its onset but is reassigned to the first subject and flagged.
    """
    names = ethogram.subject_names
    warnings: list[str] = []
    out: list[Event] = []

    # Trial phases belong to the session, not to an actor, so they keep the
    # empty subject even in a project that declares one. Attributing them to a
    # dog would make BORIS's per-subject time budgets double-count the session.
    session_level = {b.code for b in ethogram.episode_behaviors()}

    if not names:
        for e in events:
            if e.subject:
                e = Event(**{**e.to_dict(), "subject": NO_FOCAL_SUBJECT})
            out.append(e)
        return out, warnings

    unknown: dict[str, int] = {}
    for e in events:
        if e.behavior in session_level:
            out.append(e if not e.subject
                       else Event(**{**e.to_dict(), "subject": NO_FOCAL_SUBJECT}))
            continue
        if e.subject in names:
            out.append(e)
            continue
        if not e.subject and len(names) == 1:
            out.append(Event(**{**e.to_dict(), "subject": names[0]}))
            continue
        unknown[e.subject] = unknown.get(e.subject, 0) + 1
        out.append(Event(**{**e.to_dict(), "subject": names[0],
                            "comment": (e.comment + " [subject reassigned by ABC]").strip()}))
    for subj, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
        label = subj or "(blank)"
        warnings.append(
            f"{n} event(s) named subject {label!r}, which is not in the project; "
            f"reassigned to {names[0]!r}"
        )
    return out, warnings


def deduplicate(events: Sequence[Event], tolerance: float = 0.25) -> list[Event]:
    """Collapse events of the same behaviour/subject that land within *tolerance*.

    Window-based engines see the same behaviour twice when it straddles a window
    boundary. The survivor is the one with the higher confidence, or the earlier
    one when confidences tie.
    """
    ordered = sorted(events, key=lambda e: (e.behavior, e.subject, e.start))
    out: list[Event] = []
    for e in ordered:
        merged = False
        for i, kept in enumerate(out):
            if kept.behavior != e.behavior or kept.subject != e.subject:
                continue
            if abs(kept.start - e.start) > tolerance:
                continue
            if (kept.stop is None) != (e.stop is None):
                continue
            if kept.stop is not None and e.stop is not None:
                # Same state seen twice: keep the union of the two intervals.
                out[i] = Event(**{
                    **kept.to_dict(),
                    "start": min(kept.start, e.start),
                    "stop": max(kept.stop, e.stop),
                    "confidence": _max_conf(kept.confidence, e.confidence),
                })
            elif (e.confidence or 0) > (kept.confidence or 0):
                out[i] = e
            merged = True
            break
        if not merged:
            out.append(e)
    out.sort(key=lambda e: (e.start, e.behavior, e.subject))
    return out


def merge_adjacent_states(
    events: Sequence[Event], gap: float = 0.0
) -> list[Event]:
    """Join consecutive states of the same behaviour/subject separated by <= *gap*."""
    if gap < 0:
        return list(events)
    states = [e for e in events if e.stop is not None]
    points = [e for e in events if e.stop is None]
    states.sort(key=lambda e: (e.behavior, e.subject, e.start))

    out: list[Event] = []
    for e in states:
        if out and out[-1].behavior == e.behavior and out[-1].subject == e.subject \
                and e.start - (out[-1].stop or 0) <= gap:
            prev = out[-1]
            out[-1] = Event(**{
                **prev.to_dict(),
                "stop": max(prev.stop or 0, e.stop or 0),
                "confidence": _max_conf(prev.confidence, e.confidence),
            })
        else:
            out.append(e)
    combined = out + points
    combined.sort(key=lambda e: (e.start, e.behavior, e.subject))
    return combined


def enforce_exclusivity(
    events: Sequence[Event], ethogram: Ethogram
) -> tuple[list[Event], list[str]]:
    """Apply BORIS ``excluded`` rules: a state ends when an excluded one starts.

    BORIS refuses to load a project where two mutually excluded states overlap
    on the same subject, so this has to happen before writing, not after.
    """
    warnings: list[str] = []
    excl: dict[str, set[str]] = {}
    for b in ethogram.behaviors:
        raw = (b.excluded or "").strip()
        if raw:
            excl[b.code] = {c.strip() for c in raw.split(",") if c.strip()}

    if not excl:
        return list(events), warnings

    states = sorted((e for e in events if e.stop is not None), key=lambda e: e.start)
    points = [e for e in events if e.stop is None]
    out: list[Event] = []

    for e in states:
        conflicts = excl.get(e.behavior, set())
        # Index by position, never by value: Event compares by field, so
        # out.index(other) would find the first *equal* event rather than this
        # one, and truncate the wrong occurrence when a behaviour repeats.
        for idx, other in enumerate(out):
            if other.subject != e.subject:
                continue
            mutually_exclusive = (other.behavior in conflicts
                                  or e.behavior in excl.get(other.behavior, set()))
            if not mutually_exclusive:
                continue
            overlaps = (other.stop is not None and other.stop > e.start
                        and other.start < (e.stop or 0))
            # Truncate the earlier state at the newcomer's onset.
            if overlaps and other.start < e.start:
                out[idx] = Event(**{**other.to_dict(), "stop": e.start})
                warnings.append(
                    f"{other.behavior!r} truncated at {e.start:.2f}s because "
                    f"the mutually excluded {e.behavior!r} started there"
                )
        out.append(e)

    combined = out + points
    combined.sort(key=lambda e: (e.start, e.behavior, e.subject))
    return combined, warnings


def clean(
    events: Sequence[Event],
    ethogram: Ethogram,
    duration: float,
    *,
    dedup_tolerance: float = 0.25,
    merge_gap: float = 0.0,
) -> tuple[list[Event], list[str]]:
    """Run the full repair pipeline an engine's raw output has to pass.

    Order matters: unknown behaviours go first so later steps do not reason
    about codes the ethogram has never heard of, and exclusivity runs last
    because it depends on final state boundaries.
    """
    warnings: list[str] = []

    events = [e for e in events if _finite(e.start) and (e.stop is None or _finite(e.stop))]
    events, dropped = drop_unknown_behaviors(events, ethogram)
    warnings += dropped
    events, reattributed = coerce_subjects(events, ethogram)
    warnings += reattributed
    events = clamp_to_media(events, duration)
    events, closed = close_open_states(events, ethogram, duration)
    warnings += closed
    events = deduplicate(events, dedup_tolerance)
    if merge_gap > 0:
        events = merge_adjacent_states(events, merge_gap)
    events, truncated = enforce_exclusivity(events, ethogram)
    warnings += truncated
    events.sort(key=lambda e: (e.start, e.behavior, e.subject))
    return events, warnings


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _max_conf(a: float | None, b: float | None) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    return max(vals) if vals else None


# --------------------------------------------------------------------------
# multi-engine fusion
# --------------------------------------------------------------------------

def fuse(
    results: Iterable[ObservationResult],
    ethogram: Ethogram,
    ownership: dict[str, str] | None = None,
    duration: float = 0.0,
) -> tuple[list[Event], list[str]]:
    """Combine several engines' results for one observation.

    *ownership* maps a behaviour code to the engine that is authoritative for
    it -- the hybrid arrangement the research recommends, where the audio engine
    owns vocalisations, the pose engine owns proxemics, and the VLM owns the
    rest. A behaviour with no owner keeps every engine's events and relies on
    :func:`deduplicate` to collapse agreement.
    """
    ownership = ownership or {}
    warnings: list[str] = []
    pooled: list[Event] = []

    for res in results:
        name = res.engine.get("name", "")
        for e in res.events:
            owner = ownership.get(e.behavior)
            if owner and owner != name:
                continue
            pooled.append(Event(**{**e.to_dict(), "source": e.source or name}))
        warnings += [f"[{name}] {w}" for w in res.warnings]

    for code in ownership:
        if not ethogram.has(code):
            warnings.append(f"ownership rule names unknown behaviour {code!r}")

    cleaned, w = clean(pooled, ethogram, duration)
    return cleaned, warnings + w
