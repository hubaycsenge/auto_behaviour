"""Video-LLM engine backed by vLLM's offline batch API.

Runs Qwen3-VL (or any video-capable model vLLM supports) inside the SLURM task
itself -- no long-lived server, no Ray. That is the arrangement that survives a
cluster: an array task owns its GPU, loads the model once, codes its shard of
the corpus, and exits. A failed shard is re-queued without disturbing anything
else.

Hardware floor: vLLM needs compute capability >= 7.0 and bfloat16 needs >= 8.0.
On the NIPG cluster that means nipg38 (A100), nipg10 (3090) and nipg32 (A4000).
The Turing and Pascal nodes must use :mod:`.vlm_llamacpp` instead; this engine
refuses to start on them rather than falling back to an unusably slow eager mode.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Sequence
from typing import Any

from ...common.events import Event, MediaInfo
from ...common.media import extract_frames, frame_to_png_bytes
from . import vlm_common as vc
from .base import Engine, EngineUnavailable


class VllmEngine(Engine):
    name = "vlm_vllm"
    label = "Video-LLM via vLLM (A100 / Ampere)"
    needs_gpu = True

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._llm: Any = None
        self._sampling: Any = None
        self._schema: dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------------
    def setup(self) -> None:
        self._check_gpu()
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise EngineUnavailable(
                "vLLM is not installed in this environment. Run "
                "`setup/install_server.sh --vllm` on a node with an Ampere GPU."
            ) from exc

        opt = self.ctx.opt
        model = opt("model", "Qwen/Qwen3-VL-8B-Instruct")
        max_frames = int(opt("max_frames_per_window", 96))

        engine_kwargs: dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": int(opt("tensor_parallel_size", 1)),
            "dtype": opt("dtype", "bfloat16"),
            "max_model_len": int(opt("max_model_len", 32768)),
            "gpu_memory_utilization": float(opt("gpu_memory_utilization", 0.90)),
            # Without this the engine reserves room for one image and rejects
            # every request the moment a window carries more than that.
            "limit_mm_per_prompt": {"image": max_frames},
            "trust_remote_code": True,
            "enforce_eager": bool(opt("enforce_eager", False)),
        }
        if opt("download_dir"):
            engine_kwargs["download_dir"] = opt("download_dir")

        try:
            self._llm = LLM(**engine_kwargs)
        except TypeError:
            # Older vLLM releases do not accept every keyword; drop the optional
            # ones and retry rather than failing the whole job.
            for key in ("limit_mm_per_prompt", "enforce_eager", "download_dir"):
                engine_kwargs.pop(key, None)
            self._llm = LLM(**engine_kwargs)

        self._schema = vc.events_schema(self.ctx.ethogram, self.ctx.target_behaviors())
        n_samples = max(1, int(opt("self_consistency_samples", 1)))
        temperature = float(opt("temperature", 0.0))
        if n_samples > 1 and temperature == 0.0:
            # Identical samples cannot vote; give the sampler something to vary.
            temperature = 0.3

        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": int(opt("max_tokens", 2048)),
            "n": n_samples,
            "top_p": float(opt("top_p", 1.0)),
        }
        if opt("guided_json", True):
            sampling_kwargs.update(self._guided_kwargs(SamplingParams))
        self._sampling = SamplingParams(**sampling_kwargs)
        self._ready = True

    def _guided_kwargs(self, sampling_params_cls) -> dict[str, Any]:
        """Build the guided-decoding argument this vLLM version understands."""
        schema = vc.schema_for_decoding(self._schema)
        try:
            from vllm.sampling_params import GuidedDecodingParams
            return {"guided_decoding": GuidedDecodingParams(json=schema)}
        except ImportError:
            pass
        try:
            from vllm.sampling_params import StructuredOutputsParams  # vLLM >= 0.11
            return {"structured_outputs": StructuredOutputsParams(json=schema)}
        except ImportError:
            pass
        if "guided_json" in sampling_params_cls.__init__.__code__.co_varnames:
            return {"guided_json": schema}
        return {}

    def _check_gpu(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise EngineUnavailable("PyTorch is not installed on this node.") from exc
        if not torch.cuda.is_available():
            raise EngineUnavailable(
                "No CUDA device visible. The SLURM task needs --gres=gpu:N."
            )
        major, minor = torch.cuda.get_device_capability(0)
        sm = major * 10 + minor
        name = torch.cuda.get_device_name(0)
        if sm < 70:
            raise EngineUnavailable(
                f"{name} is compute capability {major}.{minor}; vLLM requires 7.0 or "
                f"higher. Use the vlm_llamacpp engine on this node."
            )
        dtype = str(self.ctx.opt("dtype", "bfloat16"))
        if sm < 80 and "bfloat16" in dtype:
            raise EngineUnavailable(
                f"{name} is compute capability {major}.{minor} and has no bfloat16. "
                f"Set the engine option dtype='half', or use vlm_llamacpp -- note "
                f"that Qwen3-VL's vision path does not work on Turing under vLLM at all."
            )

    def teardown(self) -> None:
        self._llm = None
        self._ready = False

    # -- analysis -----------------------------------------------------------
    def analyse(self, media: MediaInfo) -> list[Event]:
        opt = self.ctx.opt
        etho = self.ctx.ethogram
        targets = self.ctx.target_behaviors()

        if not media.has_video:
            return []

        duration = media.duration
        windows = vc.plan_windows(duration, float(opt("window_seconds", 120.0)),
                                  float(opt("window_overlap_seconds", 5.0)))
        fps = float(opt("fps", 1.0))
        max_frames = int(opt("max_frames_per_window", 96))
        max_side = int(opt("max_image_side", 768))
        obs_id = os.path.splitext(os.path.basename(media.path))[0]

        conversations: list[list[dict[str, Any]]] = []
        prepared: list[vc.Window] = []

        for i, (w_start, w_stop) in enumerate(windows):
            times = vc.window_frame_times(w_start, w_stop, fps, max_frames)
            frames = extract_frames(media.path, times, max_side=max_side)
            if not frames:
                continue
            window = vc.Window(index=i, start=w_start, stop=w_stop, frames=frames)
            prepared.append(window)
            conversations.append(self._conversation(window, etho, targets, duration, obs_id))

        if not conversations:
            return []

        batch = max(1, int(opt("windows_per_batch", 4)))
        events: list[Event] = []
        n_samples = max(1, int(opt("self_consistency_samples", 1)))
        min_conf = float(opt("min_confidence", 0.0))

        for begin in range(0, len(conversations), batch):
            chunk = conversations[begin:begin + batch]
            outputs = self._llm.chat(chunk, self._sampling)
            for window, output in zip(prepared[begin:begin + batch], outputs, strict=True):
                per_sample: list[list[Event]] = []
                for completion in output.outputs:
                    parsed, warns = vc.parse_response(
                        completion.text, etho, window,
                        min_confidence=min_conf, source=self.name,
                    )
                    per_sample.append(parsed)
                    for w in warns:
                        self._warn(w)
                events.extend(
                    vc.vote(per_sample, min_votes=max(2, (n_samples // 2) + 1))
                    if n_samples > 1 else (per_sample[0] if per_sample else [])
                )

        return events

    def _conversation(
        self,
        window: vc.Window,
        etho,
        targets: Sequence[str],
        duration: float,
        obs_id: str,
    ) -> list[dict[str, Any]]:
        """One chat conversation: the frames, then the instructions.

        Frames are sent as base64 data URIs rather than file paths so the engine
        works identically whether the media sits on shared storage or in the
        job's upload directory.
        """
        content: list[dict[str, Any]] = []
        for frame in window.frames:
            b64 = base64.b64encode(frame_to_png_bytes(frame)).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": vc.build_prompt(
            etho, window, behaviors=targets, video_duration=duration,
            observation_id=obs_id,
            extra_instructions=str(self.ctx.opt("extra_instructions", "")),
        )})
        return [
            {"role": "system", "content": vc.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _warn(self, message: str) -> None:
        # Warnings raised mid-analysis are attached to the result by the runner
        # via the diagnostics channel; keep a bounded list so a pathological
        # video cannot produce a megabyte of JSON.
        bucket = self.ctx.options.setdefault("_warnings", [])
        if len(bucket) < 200:
            bucket.append(message)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["model"] = self.ctx.opt("model", "")
        d["backend"] = "vllm"
        return d
