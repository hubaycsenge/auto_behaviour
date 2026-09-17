"""Media probing and frame/audio sampling.

The NIPG nodes have no system ffmpeg, so nothing here may shell out to one.
Three backends are tried in order and all of them ship as wheels:

``av`` (PyAV)
    Preferred. Bundles its own ffmpeg libraries, seeks accurately, and exposes
    stream metadata without decoding.
``imageio_ffmpeg``
    Fallback. Ships a static ffmpeg binary; used through its own reader API.
``cv2`` (opencv-python)
    Last resort. Good enough for probing and uniform sampling, but its frame
    timestamps drift on variable-frame-rate files.

Import this module anywhere: it degrades to :data:`BACKEND` ``= None`` rather
than raising, so the client can inspect a project on a machine with no codecs
installed and only fail when it actually needs pixels.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

from .events import MediaInfo

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv",
    ".flv", ".webm", ".mts", ".m2ts", ".ts", ".3gp", ".ogv", ".mxf",
}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def _detect_backend() -> str | None:
    for name in ("av", "imageio_ffmpeg", "cv2"):
        try:
            __import__(name)
            return name
        except Exception:
            continue
    return None


BACKEND = _detect_backend()


class MediaError(RuntimeError):
    """Raised when a media file cannot be read with any available backend."""


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def observation_id_for(path: str | os.PathLike) -> str:
    """The observation ID for a media file: its name without the extension.

    This is the rule the whole system is built on -- ``03_30_152.mp4`` codes as
    observation ``03_30_152`` -- so it lives in one place.
    """
    return pathlib.Path(path).stem


def discover_videos(
    folder: str | os.PathLike,
    *,
    recursive: bool = False,
    extensions: Sequence[str] | None = None,
) -> list[pathlib.Path]:
    """List media files in *folder*, sorted, ready to become observations.

    Hidden files and the sidecar files ABC itself writes are skipped.
    """
    folder = pathlib.Path(folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    exts = {e.lower() for e in (extensions or MEDIA_EXTENSIONS)}
    it = folder.rglob("*") if recursive else folder.glob("*")
    out = [
        p for p in it
        if p.is_file()
        and p.suffix.lower() in exts
        and not p.name.startswith(".")
        and not p.name.startswith("_abc_")
    ]
    return sorted(out, key=lambda p: p.name.lower())


def duplicate_observation_ids(paths: Sequence[pathlib.Path]) -> dict[str, list[pathlib.Path]]:
    """Group *paths* that would collapse onto the same observation ID.

    ``clip.mp4`` and ``clip.MP4`` in the same folder, or the same stem found
    twice in a recursive scan, would silently overwrite each other -- the
    client refuses to submit until the user resolves them.
    """
    groups: dict[str, list[pathlib.Path]] = {}
    for p in paths:
        groups.setdefault(observation_id_for(p), []).append(p)
    return {k: v for k, v in groups.items() if len(v) > 1}


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------

def probe(path: str | os.PathLike) -> MediaInfo:
    """Read duration, fps, resolution and stream presence without decoding."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if BACKEND == "av":
        return _probe_av(path)
    if BACKEND == "imageio_ffmpeg":
        return _probe_imageio(path)
    if BACKEND == "cv2":
        return _probe_cv2(path)
    raise MediaError(
        "No media backend available. Install one of: av, imageio-ffmpeg, opencv-python."
    )


def _probe_av(path: pathlib.Path) -> MediaInfo:
    import av

    with av.open(str(path)) as container:
        vstreams = [s for s in container.streams if s.type == "video"]
        astreams = [s for s in container.streams if s.type == "audio"]
        duration = float(container.duration / 1_000_000) if container.duration else 0.0

        fps = width = height = 0
        if vstreams:
            v = vstreams[0]
            if v.average_rate:
                fps = float(v.average_rate)
            width, height = int(v.codec_context.width), int(v.codec_context.height)
            if not duration and v.duration and v.time_base:
                duration = float(v.duration * v.time_base)
        elif astreams and not duration:
            a = astreams[0]
            if a.duration and a.time_base:
                duration = float(a.duration * a.time_base)

    return MediaInfo(path=str(path), duration=round(duration, 3), fps=round(fps, 6),
                     has_video=bool(vstreams), has_audio=bool(astreams),
                     width=width, height=height)


def _probe_imageio(path: pathlib.Path) -> MediaInfo:
    import imageio_ffmpeg

    meta = imageio_ffmpeg.read_frames(str(path)).__next__()
    size = meta.get("size") or (0, 0)
    return MediaInfo(path=str(path), duration=round(float(meta.get("duration") or 0.0), 3),
                     fps=round(float(meta.get("fps") or 0.0), 6),
                     has_video=True, has_audio=bool(meta.get("audio_codec")),
                     width=int(size[0]), height=int(size[1]))


def _probe_cv2(path: pathlib.Path) -> MediaInfo:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise MediaError(f"cannot open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        return MediaInfo(
            path=str(path),
            duration=round(n / fps, 3) if fps > 0 else 0.0,
            fps=round(fps, 6), has_video=True, has_audio=False,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


# --------------------------------------------------------------------------
# frame sampling
# --------------------------------------------------------------------------

@dataclass
class Frame:
    """One sampled frame: the image plus the time it was taken from."""

    time: float
    image: object          # PIL.Image.Image
    index: int = 0

    @property
    def timestamp(self) -> str:
        m, s = divmod(self.time, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def sample_times(
    start: float,
    stop: float,
    fps: float,
    max_frames: int,
) -> list[float]:
    """Timestamps to sample between *start* and *stop* at *fps*, capped.

    Sampling density is the single biggest accuracy/cost lever for a video VLM:
    accuracy climbs to roughly 96-256 frames and then degrades as extra frames
    add visual noise. When the requested rate would exceed *max_frames* the
    interval is sampled uniformly instead of truncated, so the whole window
    stays covered rather than only its first seconds.
    """
    if stop <= start:
        return [max(0.0, start)]
    span = stop - start
    n = int(math.floor(span * fps)) + 1 if fps > 0 else max_frames
    n = max(1, min(n, max_frames))
    if n == 1:
        return [start + span / 2]
    step = span / (n - 1) if n > 1 else span
    return [round(start + i * step, 3) for i in range(n)]


def extract_frames(
    path: str | os.PathLike,
    times: Sequence[float],
    *,
    max_side: int = 768,
) -> list[Frame]:
    """Decode the frames nearest to *times*, downscaled so the long side <= *max_side*.

    Resolution matters as much as frame count: published zero-shot behaviour
    work degrades sharply below roughly 0.28 mm/pixel on the animal, while
    oversized frames blow up the token budget quadratically. 768 is the cap the
    Qwen video benchmarks use.
    """
    if not times:
        return []
    if BACKEND == "av":
        return _extract_av(pathlib.Path(path), times, max_side)
    if BACKEND == "imageio_ffmpeg":
        return _extract_imageio(pathlib.Path(path), times, max_side)
    if BACKEND == "cv2":
        return _extract_cv2(pathlib.Path(path), times, max_side)
    raise MediaError("No media backend available for frame extraction.")


def _fit(img, max_side: int):
    from PIL import Image

    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def _extract_av(path: pathlib.Path, times: Sequence[float], max_side: int) -> list[Frame]:
    import av

    out: list[Frame] = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise MediaError(f"{path} has no video stream")
        stream.thread_type = "AUTO"
        tb = stream.time_base

        for i, t in enumerate(sorted(times)):
            # Seek to the keyframe at or before t, then decode forward. Seeking
            # per frame is slower than a single linear pass but is the only way
            # to hit scattered timestamps in a long file without holding it all.
            try:
                container.seek(int(t / tb), stream=stream, backward=True, any_frame=False)
            except Exception:
                container.seek(0)
            picked = None
            for frame in container.decode(stream):
                ftime = float(frame.pts * tb) if frame.pts is not None else 0.0
                picked = frame
                if ftime >= t:
                    break
            if picked is None:
                continue
            ftime = float(picked.pts * tb) if picked.pts is not None else t
            out.append(Frame(time=round(ftime, 3),
                             image=_fit(picked.to_image(), max_side), index=i))
    return out


def _extract_imageio(path: pathlib.Path, times: Sequence[float], max_side: int) -> list[Frame]:
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image

    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    w, h = meta["size"]
    fps = float(meta.get("fps") or 25.0)
    wanted = {int(round(t * fps)): t for t in times}
    out: list[Frame] = []
    for idx, raw in enumerate(reader):
        if idx in wanted:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            out.append(Frame(time=round(idx / fps, 3),
                             image=_fit(Image.fromarray(arr), max_side), index=len(out)))
            if len(out) == len(wanted):
                break
    return out


def _extract_cv2(path: pathlib.Path, times: Sequence[float], max_side: int) -> list[Frame]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise MediaError(f"cannot open {path}")
        out: list[Frame] = []
        for i, t in enumerate(sorted(times)):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(Frame(time=round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3),
                             image=_fit(Image.fromarray(rgb), max_side), index=i))
        return out
    finally:
        cap.release()


def frame_to_png_bytes(frame: Frame, quality: int = 85) -> bytes:
    """Encode a frame as JPEG bytes (for base64 payloads to a model server)."""
    buf = io.BytesIO()
    frame.image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def extract_audio(
    path: str | os.PathLike,
    *,
    sample_rate: int = 16000,
    start: float = 0.0,
    stop: float | None = None,
):
    """Decode mono audio as a float32 numpy array in ``[-1, 1]``.

    Returns ``(samples, sample_rate)``. Raises :class:`MediaError` when the file
    carries no audio, which the audio engine reports as a skipped observation
    rather than a failure.
    """
    import numpy as np

    if BACKEND != "av":
        raise MediaError(
            "Audio extraction needs PyAV (pip install av); the current backend is "
            f"{BACKEND!r}."
        )
    import av

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise MediaError(f"{path} has no audio stream")
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)

        if start > 0:
            with contextlib.suppress(Exception):
                container.seek(int(start / stream.time_base), stream=stream, backward=True)

        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            ftime = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if stop is not None and ftime > stop:
                break
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                chunks.append(arr.reshape(-1).astype("float32"))

    if not chunks:
        raise MediaError(f"decoded no audio from {path}")
    samples = np.concatenate(chunks)
    if start > 0:
        samples = samples[int(start * sample_rate):]
    if stop is not None:
        samples = samples[: int((stop - max(0.0, start)) * sample_rate)]
    peak = float(abs(samples).max()) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    return samples, sample_rate
