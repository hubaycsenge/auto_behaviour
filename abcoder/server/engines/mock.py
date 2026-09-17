"""A deterministic engine that needs no GPU and no model.

Its only job is to prove the pipeline: submit -> SLURM -> results -> fusion ->
BORIS project on the client, with paths that resolve. Run it first on any new
deployment. The events it emits are plausible-looking nonsense, derived from a
hash of the observation ID so the same video always produces the same coding.
"""

from __future__ import annotations

import hashlib
import random

from ...common.events import Event, MediaInfo
from .base import Engine


class MockEngine(Engine):
    name = "mock"
    label = "Mock (no GPU, for testing)"
    needs_gpu = False

    def analyse(self, media: MediaInfo) -> list[Event]:
        etho = self.ctx.target_ethogram()
        if not etho.behaviors or media.duration <= 0:
            return []

        seed = int(self.ctx.opt("seed", 0)) + int(
            hashlib.sha256(media.path.encode()).hexdigest()[:8], 16
        )
        rng = random.Random(seed)

        subjects = etho.subject_names or [""]
        points = etho.point_behaviors()
        states = etho.state_behaviors()
        events: list[Event] = []

        # Lay the state behaviours out as a non-overlapping sequence covering
        # the middle 80% of the clip -- the shape a trial-phase coding has.
        if states:
            span_start = media.duration * 0.1
            span = media.duration * 0.8
            step = span / len(states)
            for i, b in enumerate(states):
                start = span_start + i * step
                events.append(Event(behavior=b.code, start=round(start, 2),
                                    stop=round(start + step * 0.95, 2),
                                    subject="" if b.category else rng.choice(subjects),
                                    confidence=round(rng.uniform(0.5, 0.95), 2),
                                    comment="mock engine"))

        if points:
            rate = float(self.ctx.opt("events_per_minute", 4.0))
            n = max(1, int(media.duration / 60.0 * rate))
            for _ in range(n):
                b = rng.choice(points)
                events.append(Event(behavior=b.code,
                                    start=round(rng.uniform(0, media.duration), 2),
                                    subject=rng.choice(subjects),
                                    confidence=round(rng.uniform(0.4, 0.99), 2),
                                    comment="mock engine"))

        events.sort(key=lambda e: e.start)
        return events
