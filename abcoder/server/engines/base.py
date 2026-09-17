"""The engine interface.

An engine turns one media file plus an ethogram into a list of
:class:`~abcoder.common.events.Event`. That is the whole contract. Everything
else -- SLURM, staging, BORIS, fusion -- is handled around it, so adding a new
analysis method means writing one subclass and registering it.

Engines are constructed on the compute node, inside the SLURM task, and are
expected to load their model in :meth:`Engine.setup` so that the cost is paid
once per array task rather than once per video.
"""

from __future__ import annotations

import abc as _abc
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ...common.ethogram import Ethogram
from ...common.events import Event, MediaInfo, ObservationResult, clean


@dataclass
class EngineContext:
    """What an engine is told about the job it is part of."""

    ethogram: Ethogram
    options: dict[str, Any] = field(default_factory=dict)
    #: Behaviour codes this engine is responsible for; empty means all of them.
    behaviors: list[str] = field(default_factory=list)
    job_dir: str = ""
    cache_dir: str = ""

    def target_behaviors(self) -> list[str]:
        """The behaviour codes this engine should actually look for."""
        if self.behaviors:
            return [c for c in self.behaviors if self.ethogram.has(c)]
        return self.ethogram.codes

    def target_ethogram(self) -> Ethogram:
        """A view of the ethogram narrowed to this engine's behaviours."""
        wanted = set(self.target_behaviors())
        return Ethogram(
            behaviors=[b for b in self.ethogram.behaviors if b.code in wanted],
            subjects=list(self.ethogram.subjects),
            categories=list(self.ethogram.categories),
        )

    def opt(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)


class EngineUnavailable(RuntimeError):
    """The engine cannot run here (missing model, wrong GPU, absent dependency).

    Raised from :meth:`Engine.setup`. The runner turns it into a failed result
    with a readable message instead of a stack trace, because this is almost
    always a deployment problem the user can fix.
    """


class Engine(_abc.ABC):
    """Base class for every analysis engine."""

    #: Registry key; must match the name used in :data:`abcoder.common.config.ENGINE_PRESETS`.
    name: str = ""
    #: One line shown in the client's engine picker.
    label: str = ""
    #: True when the engine needs a GPU allocation to be useful.
    needs_gpu: bool = False

    def __init__(self, ctx: EngineContext) -> None:
        self.ctx = ctx
        self._ready = False

    # -- lifecycle ----------------------------------------------------------
    def setup(self) -> None:
        """Load models. Called once per array task, before any video."""
        self._ready = True

    def teardown(self) -> None:
        """Release models and subprocesses. Always called, even after a failure."""
        self._ready = False

    def __enter__(self) -> Engine:
        self.setup()
        return self

    def __exit__(self, *exc) -> None:
        self.teardown()

    # -- the actual work ----------------------------------------------------
    @_abc.abstractmethod
    def analyse(self, media: MediaInfo) -> list[Event]:
        """Code one media file. Return raw events; cleaning happens downstream."""

    # -- reporting ----------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Provenance recorded in every result file."""
        return {"name": self.name, "label": self.label, "options": dict(self.ctx.options)}

    def run(self, observation_id: str, media: Sequence[MediaInfo]) -> ObservationResult:
        """Analyse an observation and package the result.

        Catches everything: one bad video must not take out the array task that
        still has forty others to code.
        """
        result = ObservationResult(observation_id=observation_id, media=list(media),
                                   engine=self.describe())
        started = time.time()
        try:
            if not self._ready:
                self.setup()
            raw: list[Event] = []
            offset = 0.0
            for m in media:
                for e in self.analyse(m):
                    # Several files in one observation play back-to-back in
                    # BORIS, so later files' times are offset by what came before.
                    raw.append(Event(**{**e.to_dict(),
                                        "start": e.start + offset,
                                        "stop": None if e.stop is None else e.stop + offset,
                                        "source": e.source or self.name}))
                offset += m.duration
            duration = sum(m.duration for m in media)
            result.events, result.warnings = clean(raw, self.ctx.ethogram, duration)
            result.diagnostics["raw_event_count"] = len(raw)
            result.diagnostics["kept_event_count"] = len(result.events)
        except EngineUnavailable as exc:
            result.failed = True
            result.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            result.failed = True
            result.error = f"{type(exc).__name__}: {exc}"
            result.diagnostics["traceback"] = traceback.format_exc()
        finally:
            result.diagnostics["seconds"] = round(time.time() - started, 2)
            # Engines report problems they met mid-analysis by appending to a
            # bucket on the context, because they have no handle on the result
            # object. Drain it here so nothing is lost and the next observation
            # starts clean.
            bucket = self.ctx.options.pop("_warnings", None)
            if bucket:
                result.warnings.extend(bucket)
            diag = self.ctx.options.pop("_diagnostics", None)
            if diag:
                result.diagnostics.update(diag)
        return result
