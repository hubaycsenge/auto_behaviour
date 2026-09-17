"""Ethogram, subject and behavioural-category models.

These are ABC's in-memory representation of the coding scheme. They map
losslessly onto the BORIS project fields ``behaviors_conf``, ``subjects_conf``
and ``behavioral_categories`` (see :mod:`abc.common.boris`), and they are what
the analysis engines are handed so they know what they are looking for.

Design note on subjects
-----------------------
In BORIS, a *subject* is the actor a behaviour is attributed to -- the dog, the
owner, the robot. A *behaviour* is what the actor does. Some projects abuse the
subject field to store the phase of a trial; ABC deliberately does not. Trial
phases belong in a behavioural category of state events (see
``Ethogram.episode_behaviors``), which is the representation BORIS's own
time-budget and transition analyses expect.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

# BORIS behaviour type strings. BORIS itself is picky about these exact spellings.
POINT = "Point event"
STATE = "State event"
POINT_WITH_CODING_MAP = "Point event with coding map"
STATE_WITH_CODING_MAP = "State event with coding map"

BEHAVIOR_TYPES = (POINT, STATE, POINT_WITH_CODING_MAP, STATE_WITH_CODING_MAP)

#: Category name ABC uses for trial-phase state behaviours. Nothing enforces
#: this name -- it is the default the client pre-fills and the engines look for
#: when asked to segment a session into phases.
EPISODE_CATEGORY = "Episode"

_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# A readable default palette; behaviours without a colour cycle through it so
# BORIS's timeline stays legible instead of rendering everything in grey.
DEFAULT_PALETTE = [
    "#aa0000", "#00007f", "#01750b", "#aa557f", "#ffff00", "#00ffff",
    "#ff5500", "#aaaa00", "#ff557f", "#5555ff", "#ffbc14", "#55aa00",
    "#55aaff", "#7f3f00", "#00aa7f", "#7f007f", "#557f00", "#0055ff",
]


def is_state(behavior_type: str) -> bool:
    """True if *behavior_type* is one of the BORIS state-event flavours."""
    return "State" in (behavior_type or "")


def normalise_color(color: str | None, index: int = 0) -> str:
    """Return a ``#rrggbb`` colour, falling back to the palette at *index*."""
    if color and _COLOR_RE.match(color.strip()):
        return color.strip().lower()
    return DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]


@dataclass
class Behavior:
    """One row of the ethogram."""

    code: str
    type: str = POINT
    description: str = ""
    key: str = ""
    color: str = ""
    category: str = ""
    excluded: str = ""
    modifiers: Any = ""
    coding_map: str = ""

    def __post_init__(self) -> None:
        self.code = (self.code or "").strip()
        self.type = self.type if self.type in BEHAVIOR_TYPES else POINT
        # BORIS stores a single-character key; longer strings break its shortcut table.
        self.key = (self.key or "")[:1]

    @property
    def is_state(self) -> bool:
        return is_state(self.type)

    def to_boris(self, index: int = 0) -> dict[str, Any]:
        """Serialise to a ``behaviors_conf`` entry."""
        return {
            "type": self.type,
            "key": self.key,
            "code": self.code,
            "description": self.description,
            "color": normalise_color(self.color, index),
            "category": self.category,
            "modifiers": self.modifiers if self.modifiers else "",
            "excluded": self.excluded,
            "coding map": self.coding_map,
        }

    @classmethod
    def from_boris(cls, raw: dict[str, Any]) -> Behavior:
        return cls(
            code=raw.get("code", ""),
            type=raw.get("type", POINT),
            description=raw.get("description", ""),
            key=raw.get("key", ""),
            color=raw.get("color", ""),
            category=raw.get("category", ""),
            excluded=raw.get("excluded", ""),
            modifiers=raw.get("modifiers", ""),
            coding_map=raw.get("coding map", raw.get("coding_map", "")),
        )


@dataclass
class Subject:
    """One actor that behaviours can be attributed to."""

    name: str
    description: str = ""
    key: str = ""

    def __post_init__(self) -> None:
        self.name = (self.name or "").strip()
        self.key = (self.key or "")[:1]

    def to_boris(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "description": self.description}

    @classmethod
    def from_boris(cls, raw: dict[str, Any]) -> Subject:
        return cls(
            name=raw.get("name", ""),
            description=raw.get("description", ""),
            key=raw.get("key", ""),
        )


@dataclass
class Ethogram:
    """A complete coding scheme: behaviours, subjects and categories."""

    behaviors: list[Behavior] = field(default_factory=list)
    subjects: list[Subject] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    # -- container protocol -------------------------------------------------
    def __iter__(self) -> Iterator[Behavior]:
        return iter(self.behaviors)

    def __len__(self) -> int:
        return len(self.behaviors)

    # -- lookups ------------------------------------------------------------
    @property
    def codes(self) -> list[str]:
        return [b.code for b in self.behaviors]

    @property
    def subject_names(self) -> list[str]:
        return [s.name for s in self.subjects]

    def get(self, code: str) -> Behavior | None:
        for b in self.behaviors:
            if b.code == code:
                return b
        return None

    def has(self, code: str) -> bool:
        return self.get(code) is not None

    def state_behaviors(self) -> list[Behavior]:
        return [b for b in self.behaviors if b.is_state]

    def point_behaviors(self) -> list[Behavior]:
        return [b for b in self.behaviors if not b.is_state]

    def in_category(self, category: str) -> list[Behavior]:
        return [b for b in self.behaviors if b.category == category]

    def episode_behaviors(self) -> list[Behavior]:
        """State behaviours in the :data:`EPISODE_CATEGORY` category.

        Empty when the project does not model trial phases, which is the normal
        case -- callers must handle that.
        """
        return [b for b in self.in_category(EPISODE_CATEGORY) if b.is_state]

    # -- mutation -----------------------------------------------------------
    def add(self, behavior: Behavior) -> None:
        if self.has(behavior.code):
            raise ValueError(f"duplicate behaviour code: {behavior.code!r}")
        self.behaviors.append(behavior)
        if behavior.category and behavior.category not in self.categories:
            self.categories.append(behavior.category)

    def add_subject(self, subject: Subject) -> None:
        if subject.name in self.subject_names:
            raise ValueError(f"duplicate subject: {subject.name!r}")
        self.subjects.append(subject)

    def remove(self, code: str) -> None:
        self.behaviors = [b for b in self.behaviors if b.code != code]

    def remove_subject(self, name: str) -> None:
        self.subjects = [s for s in self.subjects if s.name != name]

    # -- validation ---------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means usable."""
        problems: list[str] = []
        if not self.behaviors:
            problems.append("The ethogram has no behaviours.")

        seen: set[str] = set()
        for b in self.behaviors:
            if not b.code:
                problems.append("A behaviour has an empty code.")
            elif b.code in seen:
                problems.append(f"Duplicate behaviour code: {b.code!r}.")
            seen.add(b.code)
            if b.type not in BEHAVIOR_TYPES:
                problems.append(f"{b.code!r} has unknown type {b.type!r}.")

        subj_seen: set[str] = set()
        for s in self.subjects:
            if not s.name:
                problems.append("A subject has an empty name.")
            elif s.name in subj_seen:
                problems.append(f"Duplicate subject: {s.name!r}.")
            subj_seen.add(s.name)

        # A behaviour code that collides with a subject name makes coded files
        # ambiguous to read and is the hallmark of the subject-as-phase idiom.
        collisions = seen & subj_seen
        for c in sorted(collisions):
            problems.append(
                f"{c!r} is both a behaviour code and a subject name. In BORIS a "
                f"subject is an actor and a behaviour is what it does; consider "
                f"moving {c!r} into the {EPISODE_CATEGORY!r} category as a state "
                f"behaviour instead of a subject."
            )
        return problems

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "behaviors": [asdict(b) for b in self.behaviors],
            "subjects": [asdict(s) for s in self.subjects],
            "categories": list(self.categories),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Ethogram:
        return cls(
            behaviors=[Behavior(**b) for b in raw.get("behaviors", [])],
            subjects=[Subject(**s) for s in raw.get("subjects", [])],
            categories=list(raw.get("categories", [])),
        )

    def save_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path) -> Ethogram:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- prompt rendering ---------------------------------------------------
    def describe_for_prompt(self, include: Iterable[str] | None = None) -> str:
        """Render the ethogram as the numbered definition list shown to a VLM.

        Only behaviours in *include* are rendered when it is given, which is how
        an engine restricts itself to the behaviours it is responsible for in a
        multi-engine job.
        """
        wanted = set(include) if include is not None else None
        lines: list[str] = []
        n = 0
        for b in self.behaviors:
            if wanted is not None and b.code not in wanted:
                continue
            n += 1
            kind = "STATE (has a start and an end)" if b.is_state else "POINT (a single instant)"
            desc = b.description.strip() or "(no description given)"
            lines.append(f'{n}. "{b.code}" [{kind}] -- {desc}')
        return "\n".join(lines)
