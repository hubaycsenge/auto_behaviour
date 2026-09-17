"""Pose / proxemics engine: geometry instead of interpretation.

Detects and tracks the dog, the person and the robot, then derives behaviours
from measured quantities -- who moved toward whom, how fast, how close, for how
long. Nothing here can hallucinate an action, which is the point: it gives a
VLM's output something to be checked against, and it covers the behaviours the
reference ethogram states in explicitly geometric terms ("moving towards the
robot", "moving backwards but not toward the owner").

What it cannot do is read gaze. Every ethogram definition that turns on where
the dog is *looking* is approximated here by heading -- the direction the dog is
travelling -- and the evidence string on every event says so. Treat these
events as candidates for a human to confirm, and calibrate the thresholds
against hand-coded video before trusting the output; the defaults are starting
points for a fixed-camera indoor setup, not constants.

Robot detection: COCO has no robot class. Give the engine a ``robot_roi`` when
the robot's start position is fixed, list detector classes that stand in for it,
or leave both empty -- the engine then codes only the behaviours that do not
need it, and says which ones it skipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ...common.ethogram import Behavior
from ...common.events import Event, MediaInfo
from .base import Engine, EngineUnavailable
from .matching import match_keywords

#: Geometric rules the engine can evaluate, and the words that select each one
#: from an ethogram entry. Every rule reduces to a predicate over the tracks.
RULES: dict[str, dict[str, Any]] = {
    "approach_robot": {
        "keywords": ["approach robot", "approaching robot", "toward robot",
                     "towards robot", "approach agent", "approaching agent"],
        "needs": ("dog", "robot"),
        "description": "distance to the robot falling faster than the approach threshold",
    },
    "approach_person": {
        "keywords": ["approach owner", "approaching owner", "toward owner",
                     "towards owner", "approach person", "approach handler"],
        "needs": ("dog", "person"),
        "description": "distance to the person falling faster than the approach threshold",
    },
    "withdraw_robot": {
        "keywords": ["backing", "back away", "retreat", "withdraw", "moving backwards",
                     "away from robot", "avoid"],
        "needs": ("dog", "robot"),
        "description": "distance to the robot rising while the person is not being approached",
    },
    "orient_robot": {
        "keywords": ["orient at robot", "orienting at robot", "orienting at the agent",
                     "look at robot", "facing robot"],
        "needs": ("dog", "robot"),
        "description": "heading within the orientation cone of the bearing to the robot",
    },
    "orient_person": {
        "keywords": ["orient at owner", "orienting at owner", "orienting at the owner",
                     "look at owner", "facing owner", "orient at person"],
        "needs": ("dog", "person"),
        "description": "heading within the orientation cone of the bearing to the person",
    },
    "near_person": {
        "keywords": ["near owner", "close to owner", "proximity to owner",
                     "staying close to the owner", "contact with owner"],
        "needs": ("dog", "person"),
        "description": "distance to the person below the proximity threshold",
    },
    "near_robot": {
        "keywords": ["near robot", "close to robot", "proximity to robot", "contact with robot"],
        "needs": ("dog", "robot"),
        "description": "distance to the robot below the proximity threshold",
    },
    "immobile": {
        "keywords": ["immobil", "freez", "motionless", "stand still", "stationary"],
        "needs": ("dog",),
        "description": "speed below the immobility threshold for the minimum duration",
    },
}


@dataclass
class Track:
    """One actor's trajectory: centre position and box size over time."""

    label: str
    times: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)
    h: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)


class PoseEngine(Engine):
    name = "pose"
    label = "Pose / proxemics"
    needs_gpu = True

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._model: Any = None

    # -- lifecycle ----------------------------------------------------------
    def setup(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise EngineUnavailable(
                "ultralytics is not installed. Run `setup/install_server.sh --pose`."
            ) from exc
        model_name = str(self.ctx.opt("model", "yolo11m.pt"))
        try:
            self._model = YOLO(model_name)
        except Exception as exc:  # noqa: BLE001 - weight download or load failure
            raise EngineUnavailable(f"could not load detector {model_name!r}: {exc}") from exc
        self._ready = True

    def teardown(self) -> None:
        self._model = None
        self._ready = False

    # -- analysis -----------------------------------------------------------
    def analyse(self, media: MediaInfo) -> list[Event]:
        if not media.has_video or media.duration <= 0:
            return []

        tracks, frame_diagonal = self._track(media)
        available = {name for name, t in tracks.items() if len(t) >= 2}
        if "dog" not in available:
            self._warn(f"no dog track found in {media.path}; nothing to code")
            return []

        assignments = self._assign_rules()
        if not assignments:
            self._warn(
                "no ethogram behaviour matched a geometric rule; set the engine "
                "option 'rules' to map behaviour codes to rule names explicitly "
                f"(available: {', '.join(sorted(RULES))})"
            )
            return []

        series = self._derive(tracks, frame_diagonal)
        subject = self._default_subject()
        events: list[Event] = []

        for behavior, rule in assignments:
            needs = set(RULES[rule]["needs"])
            missing = needs - available
            if missing:
                self._warn(
                    f"skipped {behavior.code!r}: rule {rule!r} needs a track for "
                    f"{', '.join(sorted(missing))}"
                )
                continue
            intervals = self._evaluate(rule, series)
            for start, stop, strength in intervals:
                events.append(Event(
                    behavior=behavior.code,
                    start=round(start, 3),
                    stop=round(stop, 3) if behavior.is_state else None,
                    subject=subject,
                    confidence=round(min(1.0, strength), 3),
                    evidence=f"{rule}: {RULES[rule]['description']}"
                             + (" (heading used as a proxy for gaze)"
                                if "orient" in rule else ""),
                    source=self.name,
                ))
        return events

    def _default_subject(self) -> str:
        names = self.ctx.ethogram.subject_names
        if not names:
            return ""
        # The dog track is the one being coded; prefer a subject that says so.
        for n in names:
            if "dog" in n.lower():
                return n
        return names[0]

    # -- tracking -----------------------------------------------------------
    def _track(self, media: MediaInfo) -> tuple[dict[str, Track], float]:
        """Run the detector over sampled frames and build one track per actor.

        Only the largest detection of each kind is kept per frame. With one dog
        and one handler in shot that is right; with several dogs it silently
        follows whichever is biggest, so multi-animal sessions need a different
        engine.
        """
        opt = self.ctx.opt
        sample_fps = float(opt("fps", 5.0))
        stride = max(1, int(round((media.fps or 25.0) / max(sample_fps, 0.1))))
        class_map = dict(opt("classes", {}) or {})
        wanted: dict[str, set[str]] = {
            role: {c.lower() for c in names}
            for role, names in class_map.items() if names
        }

        tracks: dict[str, Track] = {}
        diagonal = math.hypot(media.width or 1920, media.height or 1080)

        results = self._model.track(
            source=media.path,
            stream=True,
            persist=True,
            tracker=str(opt("tracker", "bytetrack.yaml")),
            conf=float(opt("conf", 0.35)),
            imgsz=int(opt("imgsz", 960)),
            device=str(opt("device", "cuda")),
            vid_stride=stride,
            verbose=False,
        )

        fps = media.fps or 25.0
        for index, frame_result in enumerate(results):
            time = index * stride / fps
            boxes = getattr(frame_result, "boxes", None)
            if boxes is None or boxes.xywh is None:
                continue
            names = frame_result.names or {}
            best: dict[str, tuple[float, Sequence[float]]] = {}
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i]) if boxes.cls is not None else -1
                label = str(names.get(cls_id, "")).lower()
                xywh = [float(v) for v in boxes.xywh[i]]
                area = xywh[2] * xywh[3]
                for role, accepted in wanted.items():
                    if label in accepted and (role not in best or area > best[role][0]):
                        best[role] = (area, xywh)
            for role, (_, xywh) in best.items():
                t = tracks.setdefault(role, Track(label=role))
                t.times.append(time)
                t.x.append(xywh[0])
                t.y.append(xywh[1])
                t.w.append(xywh[2])
                t.h.append(xywh[3])

        roi = list(opt("robot_roi", []) or [])
        if "robot" not in tracks and len(roi) == 4 and "dog" in tracks:
            # A static robot: synthesise a constant track from the ROI so the
            # robot-relative rules can still be evaluated.
            width = media.width or 1920
            height = media.height or 1080
            cx = (roi[0] + roi[2]) / 2 * width
            cy = (roi[1] + roi[3]) / 2 * height
            dog = tracks["dog"]
            tracks["robot"] = Track(
                label="robot", times=list(dog.times),
                x=[cx] * len(dog), y=[cy] * len(dog),
                w=[abs(roi[2] - roi[0]) * width] * len(dog),
                h=[abs(roi[3] - roi[1]) * height] * len(dog),
            )
            self._warn("robot position taken from the configured static ROI")

        return tracks, diagonal

    # -- derived series -----------------------------------------------------
    def _derive(self, tracks: dict[str, Track], diagonal: float) -> dict[str, Any]:
        """Resample every track onto the dog's timeline and compute the geometry."""
        dog = tracks["dog"]
        times = dog.times
        smoothing = float(self.ctx.opt("smoothing_seconds", 0.6))
        dt = _median_step(times)
        window = max(1, int(round(smoothing / dt))) if dt > 0 else 1

        dx = _smooth(dog.x, window)
        dy = _smooth(dog.y, window)
        speed = _derivative(dx, dy, times)

        series: dict[str, Any] = {
            "times": times, "dog_x": dx, "dog_y": dy, "speed": speed,
            "diagonal": diagonal,
        }

        for role in ("robot", "person"):
            other = tracks.get(role)
            if other is None or len(other) < 2:
                continue
            ox = _resample(other.times, _smooth(other.x, window), times)
            oy = _resample(other.times, _smooth(other.y, window), times)
            distance = [math.hypot(a - b, c - d)
                        for a, b, c, d in zip(dx, ox, dy, oy, strict=True)]
            series[f"{role}_x"] = ox
            series[f"{role}_y"] = oy
            series[f"dist_{role}"] = distance
            # Radial velocity: negative means closing on the target.
            series[f"radial_{role}"] = _gradient(distance, times)
            series[f"bearing_{role}"] = [
                math.atan2(oy_i - dy_i, ox_i - dx_i)
                for dx_i, dy_i, ox_i, oy_i in zip(dx, dy, ox, oy, strict=True)
            ]
        series["heading"] = _heading(dx, dy, times)
        return series

    # -- rule evaluation ----------------------------------------------------
    def _evaluate(self, rule: str, series: dict[str, Any]) -> list[tuple[float, float, float]]:
        times = series["times"]
        if len(times) < 2:
            return []
        opt = self.ctx.opt
        diagonal = series["diagonal"] or 1.0

        approach_speed = float(opt("approach_speed_px_per_s", 25.0))
        min_seconds = float(opt("approach_min_seconds", 0.8))
        near = float(opt("near_fraction", 0.25)) * diagonal
        immobile_speed = float(opt("immobility_speed_px_per_s", 6.0))
        immobile_seconds = float(opt("immobility_min_seconds", 2.0))
        cone = math.radians(float(opt("orientation_cone_degrees", 45.0)))
        moving = float(opt("orientation_min_speed_px_per_s", 8.0))

        if rule == "approach_robot" or rule == "approach_person":
            role = "robot" if rule.endswith("robot") else "person"
            radial = series.get(f"radial_{role}")
            if radial is None:
                return []
            mask = [v < -approach_speed for v in radial]
            strength = [min(1.0, abs(v) / (approach_speed * 3)) for v in radial]
            return _intervals(times, mask, strength, min_seconds)

        if rule == "withdraw_robot":
            radial = series.get("radial_robot")
            if radial is None:
                return []
            to_person = series.get("radial_person")
            mask = []
            for i, v in enumerate(radial):
                receding = v > approach_speed
                # "not toward the owner" -- the ethogram's own qualifier.
                not_to_owner = True if to_person is None else to_person[i] > -approach_speed
                mask.append(receding and not_to_owner)
            strength = [min(1.0, abs(v) / (approach_speed * 3)) for v in radial]
            return _intervals(times, mask, strength, min_seconds)

        if rule in ("orient_robot", "orient_person"):
            role = "robot" if rule.endswith("robot") else "person"
            bearing = series.get(f"bearing_{role}")
            if bearing is None:
                return []
            heading = series["heading"]
            speed = series["speed"]
            mask, strength = [], []
            for b, h, s in zip(bearing, heading, speed, strict=True):
                if s < moving or h is None:
                    # Heading is meaningless when the dog is barely moving, and
                    # heading is all this rule has to stand in for gaze.
                    mask.append(False)
                    strength.append(0.0)
                    continue
                delta = abs(_wrap(b - h))
                mask.append(delta < cone)
                # Confidence falls off with angular error, and is capped: this
                # is heading, not gaze, and should never look certain.
                strength.append(max(0.0, 0.7 * (1.0 - delta / cone)))
            return _intervals(times, mask, strength, min_seconds)

        if rule in ("near_person", "near_robot"):
            role = "person" if rule.endswith("person") else "robot"
            distance = series.get(f"dist_{role}")
            if distance is None:
                return []
            mask = [d < near for d in distance]
            strength = [max(0.0, min(1.0, 1.0 - d / near)) for d in distance]
            return _intervals(times, mask, strength, min_seconds)

        if rule == "immobile":
            speed = series["speed"]
            mask = [s < immobile_speed for s in speed]
            strength = [max(0.0, min(1.0, 1.0 - s / max(immobile_speed, 1e-6))) for s in speed]
            return _intervals(times, mask, strength, immobile_seconds)

        return []

    # -- ethogram interpretation -------------------------------------------
    def _assign_rules(self) -> list[tuple[Behavior, str]]:
        """Decide which rule codes which behaviour.

        An explicit ``rules`` option wins; otherwise the engine matches the
        keywords in :data:`RULES` against the entry, code first -- see
        :func:`~abcoder.server.engines.matching.match_keywords` for why the
        order matters.
        """
        etho = self.ctx.ethogram
        explicit = dict(self.ctx.opt("rules", {}) or {})
        allowed = set(self.ctx.target_behaviors())
        out: list[tuple[Behavior, str]] = []

        for behavior in etho.behaviors:
            if behavior.code not in allowed:
                continue
            rule = explicit.get(behavior.code)
            if rule:
                if rule not in RULES:
                    self._warn(f"unknown rule {rule!r} for {behavior.code!r}")
                    continue
                out.append((behavior, rule))
                continue
            matched = match_keywords(
                behavior.code, behavior.description,
                {name: spec["keywords"] for name, spec in RULES.items()},
            )
            if matched:
                out.append((behavior, matched))
        return out

    def _warn(self, message: str) -> None:
        bucket = self.ctx.options.setdefault("_warnings", [])
        if len(bucket) < 200:
            bucket.append(message)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["detector"] = self.ctx.opt("model", "")
        d["rules"] = {b.code: r for b, r in self._assign_rules()}
        return d


# --------------------------------------------------------------------------
# small numeric helpers (pure python -- the arrays here are seconds long, not
# minutes, and this keeps the engine importable without numpy)
# --------------------------------------------------------------------------

def _median_step(times: Sequence[float]) -> float:
    if len(times) < 2:
        return 0.0
    steps = sorted(b - a for a, b in zip(times, times[1:], strict=False))
    return steps[len(steps) // 2]


def _smooth(values: Sequence[float], window: int) -> list[float]:
    """Centred moving average; the boundary shrinks the window rather than padding."""
    if window <= 1 or len(values) < 3:
        return list(values)
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _gradient(values: Sequence[float], times: Sequence[float]) -> list[float]:
    """Central-difference derivative with respect to *times*."""
    n = len(values)
    if n < 2:
        return [0.0] * n
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        dt = times[hi] - times[lo]
        out[i] = (values[hi] - values[lo]) / dt if dt > 0 else 0.0
    return out


def _derivative(xs: Sequence[float], ys: Sequence[float], times: Sequence[float]) -> list[float]:
    vx = _gradient(xs, times)
    vy = _gradient(ys, times)
    return [math.hypot(a, b) for a, b in zip(vx, vy, strict=True)]


def _heading(xs: Sequence[float], ys: Sequence[float],
             times: Sequence[float]) -> list[float | None]:
    """Direction of travel, or ``None`` where the actor is effectively still."""
    vx = _gradient(xs, times)
    vy = _gradient(ys, times)
    return [math.atan2(b, a) if math.hypot(a, b) > 1e-6 else None
            for a, b in zip(vx, vy, strict=True)]


def _resample(src_times: Sequence[float], values: Sequence[float],
              dst_times: Sequence[float]) -> list[float]:
    """Linear interpolation of *values* onto *dst_times*, held flat at the ends."""
    if not src_times:
        return [0.0] * len(dst_times)
    out: list[float] = []
    j = 0
    for t in dst_times:
        while j + 1 < len(src_times) and src_times[j + 1] < t:
            j += 1
        if j + 1 >= len(src_times):
            out.append(values[-1])
        elif t <= src_times[0]:
            out.append(values[0])
        else:
            t0, t1 = src_times[j], src_times[j + 1]
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            out.append(values[j] + frac * (values[j + 1] - values[j]))
    return out


def _wrap(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _intervals(
    times: Sequence[float],
    mask: Sequence[bool],
    strength: Sequence[float],
    min_seconds: float,
) -> list[tuple[float, float, float]]:
    """Contiguous True runs of *mask* lasting at least *min_seconds*."""
    out: list[tuple[float, float, float]] = []
    start: int | None = None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            stop = i - 1
            t0, t1 = times[start], times[min(stop, len(times) - 1)]
            if t1 - t0 >= min_seconds:
                window = [strength[k] for k in range(start, min(stop + 1, len(strength)))]
                out.append((t0, t1, max(window) if window else 0.0))
            start = None
    return out
