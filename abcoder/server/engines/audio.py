"""Audio engine: find and classify vocalisations.

Five of the fourteen behaviours in the ethogram this system was designed
against are sounds -- whine, excited bark, aggressive bark, growl, puffing. A
frame sampler never sees them, and a video-LLM asked about them from silent
stills will confabulate. So they get their own engine, and in a multi-engine
job the fusion step hands it ownership of exactly those codes.

The pipeline is deliberately classical:

1. decode mono audio at 16 kHz;
2. segment on short-time energy against an adaptive threshold;
3. measure each segment -- F0, harmonicity, spectral centroid, bandwidth,
   flatness, duration;
4. classify, either zero-shot with CLAP against the ethogram's own
   descriptions, or with acoustic rules keyed to canonical dog vocalisation
   types.

Only numpy is required for steps 1-4 with the rule classifier, so this engine
runs on any node, GPU or not. CLAP adds a transformers dependency and prefers a
GPU but fits comfortably on an 11 GB 2080 Ti.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...common.ethogram import Behavior
from ...common.events import Event, MediaInfo
from ...common.media import MediaError, extract_audio
from .base import Engine, EngineUnavailable
from .matching import match_keywords

#: Canonical dog vocalisation types, with the acoustic signature each is
#: recognised by and the words that identify it in an ethogram. The numbers are
#: starting points drawn from the descriptions in the reference ethogram, not
#: universal constants -- they want calibrating against hand-coded audio, and
#: every one of them is exposed as an engine option.
VOCAL_CLASSES: dict[str, dict[str, Any]] = {
    "whine": {
        "keywords": ["whine", "whimper", "whinge"],
        "prompt": "a dog whining, a short high-pitched tonal call",
        "f0": (350.0, 1400.0), "harmonicity": (0.35, 1.0),
        "centroid": (500.0, 4000.0), "flatness": (0.0, 0.45),
        "duration": (0.05, 2.5),
    },
    "bark_high": {
        "keywords": ["excited bark", "high bark", "alarm bark", "play bark"],
        "prompt": "a dog barking excitedly, a high-pitched bark",
        "f0": (300.0, 1200.0), "harmonicity": (0.15, 1.0),
        "centroid": (900.0, 5000.0), "flatness": (0.0, 0.6),
        "duration": (0.04, 0.6),
    },
    "bark_low": {
        "keywords": ["aggressive bark", "low bark", "threat bark", "deep bark"],
        "prompt": "a dog barking aggressively, a low-pitched noisy bark",
        "f0": (80.0, 500.0), "harmonicity": (0.05, 0.8),
        "centroid": (300.0, 2200.0), "flatness": (0.05, 0.8),
        "duration": (0.04, 0.8),
    },
    "growl": {
        "keywords": ["growl", "rumble", "snarl"],
        "prompt": "a dog growling, a low broadband noisy rumble",
        "f0": (40.0, 320.0), "harmonicity": (0.0, 0.55),
        "centroid": (100.0, 1200.0), "flatness": (0.1, 1.0),
        "duration": (0.15, 6.0),
    },
    "puff": {
        "keywords": ["puff", "huff", "snort", "grunt", "blow"],
        "prompt": "a dog puffing or huffing air through its mouth, a brief breathy sound",
        "f0": (0.0, 200.0), "harmonicity": (0.0, 0.3),
        "centroid": (200.0, 3500.0), "flatness": (0.3, 1.0),
        "duration": (0.03, 0.5),
    },
    "howl": {
        "keywords": ["howl", "bay"],
        "prompt": "a dog howling, a long sustained tonal call",
        "f0": (250.0, 1200.0), "harmonicity": (0.45, 1.0),
        "centroid": (400.0, 3000.0), "flatness": (0.0, 0.35),
        "duration": (0.8, 10.0),
    },
    "pant": {
        "keywords": ["pant", "breath"],
        "prompt": "a dog panting, rhythmic breathing with an open mouth",
        "f0": (0.0, 200.0), "harmonicity": (0.0, 0.25),
        "centroid": (600.0, 5000.0), "flatness": (0.4, 1.0),
        "duration": (0.05, 1.0),
    },
}

#: Sounds that are *not* dog vocalisations but land in the same feature space.
#: (Continuous room tone is deliberately absent: it *is* the noise floor, so
#: the segmenter never emits it as an event, and a class broad enough to catch
#: it also swallows genuine growls.)
#: They are scored alongside the real classes and never emitted: without them
#: the classifier has no way to say "this is not any of them", and every
#: syllable of the owner talking is reported as a bark. They do not make the
#: rule classifier able to tell a dog from a person -- the five features here
#: genuinely do not separate those -- but they stop the most common confusions
#: and they make the confidence number mean something.
DECOY_CLASSES: dict[str, dict[str, Any]] = {
    "speech": {
        "keywords": [],
        "prompt": "a person speaking",
        # Adult speech: F0 in the 85-300 Hz range, strongly harmonic, and
        # syllable-length. A bark overlaps, which is why this only ever
        # suppresses -- it never claims to have identified speech.
        "f0": (80.0, 300.0), "harmonicity": (0.45, 1.0),
        "centroid": (200.0, 2000.0), "flatness": (0.0, 0.40),
        "duration": (0.08, 0.60),
    },
    "transient": {
        "keywords": [],
        "prompt": "a door, a footstep or an object being handled",
        "f0": (0.0, 80.0), "harmonicity": (0.0, 0.20),
        "centroid": (1500.0, 8000.0), "flatness": (0.5, 1.0),
        "duration": (0.02, 0.25),
    },
}

#: Words that mark an ethogram entry as something you hear rather than see.
VOCAL_HINT_WORDS = (
    "bark", "growl", "whine", "whimper", "howl", "yelp", "puff", "huff",
    "snort", "grunt", "vocalis", "vocaliz", "call", "pant", "squeal", "snarl",
)


@dataclass
class Segment:
    """One detected sound, with its measurements."""

    start: float
    stop: float
    rms: float
    f0: float
    harmonicity: float
    centroid: float
    bandwidth: float
    flatness: float

    @property
    def duration(self) -> float:
        return self.stop - self.start


class AudioEngine(Engine):
    name = "audio"
    label = "Audio (vocalisations)"
    needs_gpu = False

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._clap: Any = None
        self._clap_processor: Any = None
        self._text_embeddings: Any = None
        self._clap_codes: list[str] = []
        self._mode = "rules"

    # -- lifecycle ----------------------------------------------------------
    def setup(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError as exc:
            raise EngineUnavailable("numpy is required for the audio engine.") from exc

        requested = str(self.ctx.opt("classifier", "auto")).lower()
        if requested in ("clap", "auto"):
            try:
                self._load_clap()
                self._mode = "clap"
            except Exception as exc:  # noqa: BLE001 - any load failure means fall back
                if requested == "clap":
                    raise EngineUnavailable(f"CLAP could not be loaded: {exc}") from exc
                self._warn(f"CLAP unavailable ({exc}); using acoustic rules instead")
                self._mode = "rules"
        else:
            self._mode = "rules"
        self._ready = True

    def _load_clap(self) -> None:
        """Load CLAP and pre-embed one text prompt per vocal behaviour.

        The prompts come from the ethogram itself -- the researcher's own
        wording of what the sound is -- which is the whole appeal of a
        zero-shot audio model here: no training data, and the classifier
        follows the coding scheme rather than a fixed label set.
        """
        import torch
        from transformers import ClapModel, ClapProcessor

        model_id = str(self.ctx.opt("clap_model", "laion/clap-htsat-unfused"))
        self._clap_processor = ClapProcessor.from_pretrained(model_id)
        self._clap = ClapModel.from_pretrained(model_id)
        self._clap.eval()
        if torch.cuda.is_available():
            self._clap = self._clap.cuda()

        prompts: list[str] = []
        codes: list[str] = []
        for behavior in self._vocal_behaviors():
            codes.append(behavior.code)
            prompts.append(self._text_prompt(behavior))
        if not codes:
            raise RuntimeError("no vocal behaviours in the ethogram")

        inputs = self._clap_processor(text=prompts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            emb = self._clap.get_text_features(**inputs)
        self._text_embeddings = emb / emb.norm(dim=-1, keepdim=True)
        self._clap_codes = codes

    def _text_prompt(self, behavior: Behavior) -> str:
        """The natural-language description CLAP is asked to match."""
        desc = (behavior.description or "").strip().rstrip(".")
        cls = self._canonical_class(behavior)
        if desc:
            return f"the sound of a dog: {desc}"
        if cls:
            return VOCAL_CLASSES[cls]["prompt"]
        return f"the sound of a dog {behavior.code.lower()}"

    def teardown(self) -> None:
        self._clap = None
        self._clap_processor = None
        self._text_embeddings = None
        self._ready = False

    # -- ethogram interpretation -------------------------------------------
    def _vocal_behaviors(self) -> list[Behavior]:
        """Which ethogram entries this engine is responsible for.

        An explicit ``behaviors`` list (from the job's ownership table) wins.
        Otherwise the engine picks the entries whose code or description reads
        like a sound, and says which ones it chose in the diagnostics so the
        guess is never silent.
        """
        etho = self.ctx.ethogram
        explicit = self.ctx.opt("behaviors") or self.ctx.behaviors
        if explicit:
            return [b for b in etho.behaviors if b.code in set(explicit)]
        out: list[Behavior] = []
        for b in etho.behaviors:
            haystack = f"{b.code} {b.description}".lower()
            if any(word in haystack for word in VOCAL_HINT_WORDS):
                out.append(b)
        return out

    def _canonical_class(self, behavior: Behavior) -> str | None:
        """Map an ethogram entry onto one of :data:`VOCAL_CLASSES`."""
        return match_keywords(
            behavior.code, behavior.description,
            {cls: spec["keywords"] for cls, spec in VOCAL_CLASSES.items()},
        )

    # -- analysis -----------------------------------------------------------
    def analyse(self, media: MediaInfo) -> list[Event]:

        targets = self._vocal_behaviors()
        if not targets:
            self._warn("no vocal behaviours found in the ethogram; nothing to do")
            return []
        if not media.has_audio:
            self._warn(f"{media.path} has no audio stream")
            return []

        sr = int(self.ctx.opt("sample_rate", 16000))
        try:
            samples, sr = extract_audio(media.path, sample_rate=sr)
        except MediaError as exc:
            self._warn(str(exc))
            return []

        segments = self._segment(samples, sr)
        self.ctx.options.setdefault("_diagnostics", {})["segments"] = len(segments)
        if not segments:
            return []

        subject = self._default_subject()
        min_conf = float(self.ctx.opt("min_confidence", 0.25))
        events: list[Event] = []

        if self._mode == "clap":
            scored = self._classify_clap(samples, sr, segments)
        else:
            scored = self._classify_rules(segments, targets)

        for seg, (code, confidence, why) in zip(segments, scored, strict=True):
            if not code or confidence < min_conf:
                continue
            behavior = self.ctx.ethogram.get(code)
            wants_state = behavior.is_state if behavior else False
            events.append(Event(
                behavior=code,
                start=round(seg.start, 3),
                stop=round(seg.stop, 3) if wants_state else None,
                subject=subject,
                confidence=round(float(confidence), 3),
                evidence=why,
                source=self.name,
            ))

        # A plausibility check on the result as a whole. Dogs do not vocalise
        # ninety times a minute; a rate like that means the microphone is
        # picking up something else -- usually a person talking, which the rule
        # classifier cannot reliably separate from a bark.
        per_minute = len(events) / max(media.duration / 60.0, 1e-6)
        limit = float(self.ctx.opt("implausible_rate_per_minute", 40.0))
        if per_minute > limit:
            self._warn(
                f"{len(events)} sound(s) were classified as vocalisations in "
                f"{media.duration:.0f}s ({per_minute:.0f}/min before deduplication), "
                f"which is implausibly dense for a dog. "
                + ("The rule classifier cannot tell a bark from a human voice; if "
                   "someone is speaking on this recording, run the engine with "
                   "classifier='clap', which can."
                   if self._mode == "rules" else
                   "Check the recording for another sound source, and consider "
                   "raising min_confidence.")
            )
        self.ctx.options.setdefault("_diagnostics", {})["events_per_minute"] = round(per_minute, 1)
        return events

    def _default_subject(self) -> str:
        """Which actor a sound is attributed to.

        Audio alone cannot tell two dogs apart, so with more than one subject
        declared the engine attributes to the first and the evidence string says
        it was not identified. That is honest, and the human coder can reassign.
        """
        names = self.ctx.ethogram.subject_names
        return names[0] if names else ""

    # -- segmentation -------------------------------------------------------
    def _segment(self, samples, sr: int) -> list[Segment]:
        import numpy as np

        frame = int(0.025 * sr)
        hop = int(0.010 * sr)
        if samples.size < frame * 2:
            return []

        n_frames = 1 + (samples.size - frame) // hop
        idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = samples[idx] * np.hanning(frame)[None, :]

        energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        spectrum = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
        freqs = np.fft.rfftfreq(frame, 1.0 / sr)

        # Adaptive threshold, placed between the noise floor and the loudest
        # part of the recording rather than at a fixed percentile of it. A
        # percentile threshold is self-fulfilling -- "the top 8% of frames" is
        # 8% of frames whether or not the dog made a sound -- so the floor and
        # the peak are estimated separately and the trigger sits a fraction of
        # the way between them.
        noise = float(np.percentile(energy, float(self.ctx.opt("noise_percentile", 10.0))))
        peak = float(np.percentile(energy, float(self.ctx.opt("peak_percentile", 99.0))))
        span = max(peak - noise, 1e-9)
        onset = noise + float(self.ctx.opt("onset_fraction", 0.25)) * span
        offset = noise + float(self.ctx.opt("offset_fraction", 0.12)) * span
        # Never trigger on something that is barely above the noise floor.
        onset = max(onset, noise * float(self.ctx.opt("noise_multiplier", 2.0)))

        min_len = float(self.ctx.opt("min_event_seconds", 0.06))
        max_len = float(self.ctx.opt("max_event_seconds", 4.0))
        min_gap = float(self.ctx.opt("min_gap_seconds", 0.05))

        # Hysteresis: a sound has to cross the onset threshold to start, but
        # only has to fall below the lower one to end. Without it, an amplitude
        # -modulated call such as a growl fragments into a dozen events.
        spans = _hysteresis_runs(energy, onset, offset)
        merged: list[tuple[int, int]] = []
        for start, stop in spans:
            if merged and (start - merged[-1][1]) * hop / sr <= min_gap:
                merged[-1] = (merged[-1][0], stop)
            else:
                merged.append((start, stop))

        out: list[Segment] = []
        for start, stop in merged:
            t0 = start * hop / sr
            t1 = (stop * hop + frame) / sr
            if not (min_len <= t1 - t0 <= max_len):
                continue
            band = spectrum[start:stop + 1]
            if band.size == 0:
                continue
            mean_spec = band.mean(axis=0)
            power = mean_spec ** 2
            total = float(power.sum()) + 1e-12
            centroid = float((freqs * power).sum() / total)
            bandwidth = float(math.sqrt(((freqs - centroid) ** 2 * power).sum() / total))
            # Spectral flatness: geometric over arithmetic mean. Near 1 for
            # noise (a growl), near 0 for a tonal call (a whine).
            flatness = float(np.exp(np.mean(np.log(mean_spec))) / (np.mean(mean_spec) + 1e-12))
            seg_samples = samples[int(t0 * sr):int(t1 * sr)]
            f0, harmonicity = _pitch(seg_samples, sr)
            out.append(Segment(
                start=t0, stop=t1,
                rms=float(energy[start:stop + 1].max()),
                f0=f0, harmonicity=harmonicity,
                centroid=centroid, bandwidth=bandwidth, flatness=flatness,
            ))
        return out

    # -- classification -----------------------------------------------------
    def _classify_rules(
        self, segments: Sequence[Segment], targets: Sequence[Behavior]
    ) -> list[tuple[str, float, str]]:
        """Score each segment against the acoustic signature of each behaviour.

        A behaviour whose ethogram entry maps onto no canonical class cannot be
        scored this way; it is skipped rather than guessed at, and the warning
        says so.
        """
        mapped: list[tuple[Behavior, dict[str, Any]]] = []
        unmapped: list[str] = []
        for b in targets:
            cls = self._canonical_class(b)
            if cls:
                mapped.append((b, VOCAL_CLASSES[cls]))
            else:
                unmapped.append(b.code)
        if unmapped:
            self._warn(
                "the rule classifier has no acoustic signature for "
                + ", ".join(repr(c) for c in unmapped)
                + "; install CLAP (classifier='clap') to score them from their descriptions"
            )
        if not mapped:
            return [("", 0.0, "") for _ in segments]

        # Every class competes, including the decoys. Scoring only the real
        # behaviours means the best of them always wins, however poorly it
        # fits -- which is how 79 seconds of a person talking becomes 122 barks.
        candidates: list[tuple[str, dict[str, Any], bool]] = (
            [(b.code, spec, True) for b, spec in mapped]
            + [(name, spec, False) for name, spec in DECOY_CLASSES.items()]
        )
        sharpness = float(self.ctx.opt("rule_sharpness", 12.0))
        margin = float(self.ctx.opt("rule_margin", 0.05))

        out: list[tuple[str, float, str]] = []
        for seg in segments:
            scored = sorted(
                ((label, _signature_score(seg, spec), emit) for label, spec, emit in candidates),
                key=lambda t: -t[1],
            )
            label, score, emit = scored[0]
            runner_up = scored[1][1] if len(scored) > 1 else 0.0
            features = (f"f0={seg.f0:.0f}Hz centroid={seg.centroid:.0f}Hz "
                        f"flatness={seg.flatness:.2f} dur={seg.duration:.2f}s")

            if not emit:
                # A decoy won: this sound resembles speech or object handling
                # more than anything in the ethogram. Reject it outright, but
                # only when the decoy wins by a clear margin -- a bark and a
                # shouted word really are close in these features, and a
                # marginal call should reach the human coder.
                if score - runner_up >= margin:
                    out.append(("", 0.0, f"rejected as {label}; {features}"))
                    continue
                label, score, emit = scored[1]
                if not emit:
                    out.append(("", 0.0, f"rejected as {label}; {features}"))
                    continue
            # Confidence as a posterior over the competing classes rather than
            # the winner's raw fit, so a sound that suits three classes equally
            # well cannot come out at 1.0.
            posterior = _softmax_top([s for _, s, _ in scored], sharpness)
            out.append((label, posterior, f"rules: {features}"))
        return out

    def _classify_clap(self, samples, sr: int, segments: Sequence[Segment]):
        import numpy as np
        import torch

        clips = []
        for seg in segments:
            clip = samples[int(seg.start * sr):int(seg.stop * sr)]
            # CLAP expects 48 kHz and at least a fraction of a second; pad short
            # calls rather than dropping them, since a 60 ms puff is exactly the
            # kind of event this engine exists to catch.
            if clip.size < sr // 4:
                clip = np.pad(clip, (0, sr // 4 - clip.size))
            clips.append(clip.astype("float32"))

        batch = int(self.ctx.opt("clap_batch", 16))
        results: list[tuple[str, float, str]] = []
        for begin in range(0, len(clips), batch):
            chunk = clips[begin:begin + batch]
            inputs = self._clap_processor(audios=chunk, sampling_rate=sr,
                                          return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                emb = self._clap.get_audio_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            similarity = emb @ self._text_embeddings.T
            temperature = float(self.ctx.opt("clap_temperature", 20.0))
            probs = torch.softmax(similarity * temperature, dim=-1)
            for row in probs:
                j = int(row.argmax())
                results.append((self._clap_codes[j], float(row[j]),
                                f"CLAP zero-shot, p={float(row[j]):.2f}"))
        return results

    def _warn(self, message: str) -> None:
        bucket = self.ctx.options.setdefault("_warnings", [])
        if len(bucket) < 200:
            bucket.append(message)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["classifier"] = self._mode
        d["behaviors_claimed"] = [b.code for b in self._vocal_behaviors()]
        return d


# --------------------------------------------------------------------------
# signal helpers (numpy only -- no scipy, so this runs anywhere)
# --------------------------------------------------------------------------

def _runs(mask) -> list[tuple[int, int]]:
    """Contiguous True spans in a boolean array, as ``(start, end)`` indices."""
    import numpy as np

    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _hysteresis_runs(energy, onset: float, offset: float) -> list[tuple[int, int]]:
    """Spans that cross *onset* somewhere, extended out to where they fall below *offset*."""

    above_low = energy > offset
    candidates = _runs(above_low)
    return [(a, b) for a, b in candidates if float(energy[a:b + 1].max()) > onset]


def _pitch(clip, sr: int, fmin: float = 50.0, fmax: float = 1500.0,
           harmonicity_floor: float = 0.25) -> tuple[float, float]:
    """Estimate F0 and harmonicity by autocorrelation.

    Returns ``(f0_hz, harmonicity)`` where harmonicity is the normalised
    autocorrelation at the winning lag -- near 1 for a clean tonal call, near 0
    for broadband noise. ``f0`` is 0 when nothing periodic was found, which is
    itself the signature of a puff or a growl.
    """
    import numpy as np

    if clip.size < int(sr / fmin) * 2:
        return 0.0, 0.0
    clip = clip - clip.mean()
    energy = float(np.dot(clip, clip))
    if energy <= 1e-9:
        return 0.0, 0.0

    corr = np.correlate(clip, clip, mode="full")[clip.size - 1:]
    corr = corr / (corr[0] + 1e-12)

    lo = max(1, int(sr / fmax))
    hi = min(corr.size - 1, int(sr / fmin))
    if hi <= lo:
        return 0.0, 0.0

    # Skip past the autocorrelation's central lobe before looking for a peak.
    # Band-limited noise -- a growl is exactly that -- stays strongly correlated
    # at short lags, so taking the plain maximum over the search range reports
    # the width of the lobe as a pitch and calls a rumble a whine. The first
    # local minimum marks where the lobe ends and genuine periodicity begins.
    first_min = lo
    while first_min < hi - 1 and corr[first_min + 1] < corr[first_min]:
        first_min += 1
    start = max(lo, first_min)
    if start >= hi - 1:
        return 0.0, 0.0

    window = corr[start:hi]
    lag = int(np.argmax(window)) + start
    peak = float(corr[lag])
    if peak < float(harmonicity_floor):
        return 0.0, max(0.0, peak)
    return float(sr) / lag, min(1.0, peak)


def _signature_score(seg: Segment, spec: dict[str, Any]) -> float:
    """How well a segment matches one vocal class, in ``[0, 1]``.

    Each measured feature contributes a soft in-range score rather than a hard
    pass/fail, so a call that is slightly out of range on one dimension still
    competes instead of being discarded.
    """
    parts = [
        _range_score(seg.f0, *spec["f0"]),
        _range_score(seg.harmonicity, *spec["harmonicity"]),
        _range_score(seg.centroid, *spec["centroid"]),
        _range_score(seg.flatness, *spec["flatness"]),
        _range_score(seg.duration, *spec["duration"]),
    ]
    return float(sum(parts) / len(parts))


def _softmax_top(scores: Sequence[float], sharpness: float) -> float:
    """Posterior probability of the best-scoring class among *scores*."""
    if not scores:
        return 0.0
    top = max(scores)
    weights = [math.exp(sharpness * (s - top)) for s in scores]
    total = sum(weights) or 1.0
    return float(max(weights) / total)


def _range_score(value: float, low: float, high: float, softness: float = 0.35) -> float:
    """1.0 inside ``[low, high]``, decaying smoothly outside it."""
    if high <= low:
        return 0.0
    if low <= value <= high:
        return 1.0
    span = high - low
    distance = (low - value) if value < low else (value - high)
    return float(math.exp(-(distance / (span * softness + 1e-9)) ** 2))
