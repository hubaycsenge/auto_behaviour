"""Video-LLM engine backed by a local llama.cpp server.

This is the escape hatch for the two thirds of the NIPG cluster vLLM cannot
use. The GTX 1080 nodes sit below vLLM's compute-capability floor entirely, and
on the 2080 Ti / Titan RTX nodes Qwen3-VL's vision path does not work under
vLLM at all -- upstream closed Turing support as "not planned". llama.cpp still
runs both, via CUDA or Vulkan, with a GGUF model and a matching ``mmproj``
vision projector.

The engine starts one ``llama-server`` per SLURM task, talks to it over the
OpenAI-compatible HTTP API on localhost, and shuts it down on teardown. The
prompt and the output contract are identical to the vLLM engine, so results
from the two are directly comparable -- which matters, because the sensible way
to use this cluster is to validate on the A100 and then fan the corpus out
across the cheap nodes.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from ...common.events import Event, MediaInfo
from ...common.media import extract_frames, frame_to_png_bytes
from . import vlm_common as vc
from .base import Engine, EngineUnavailable


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaCppEngine(Engine):
    name = "vlm_llamacpp"
    label = "Video-LLM via llama.cpp (Turing / Pascal)"
    needs_gpu = True

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._proc: subprocess.Popen | None = None
        self._base_url = ""
        self._schema: dict[str, Any] = {}
        self._log_handle = None

    # -- lifecycle ----------------------------------------------------------
    def setup(self) -> None:
        opt = self.ctx.opt
        binary = self._resolve_binary(str(opt("server_binary", "llama-server")))
        model = str(opt("model_path", "") or "")
        mmproj = str(opt("mmproj_path", "") or "")

        if not model:
            raise EngineUnavailable(
                "vlm_llamacpp needs a GGUF model. Set the engine option "
                "'model_path' (and 'mmproj_path' for the vision projector); "
                "`setup/fetch_gguf.sh` downloads a matching pair."
            )
        if not pathlib.Path(model).is_file():
            raise EngineUnavailable(f"GGUF model not found: {model}")
        if mmproj and not pathlib.Path(mmproj).is_file():
            raise EngineUnavailable(f"mmproj projector not found: {mmproj}")
        if not mmproj:
            raise EngineUnavailable(
                "vlm_llamacpp needs 'mmproj_path'. Without the vision projector "
                "llama.cpp loads the language model only and will never see a frame."
            )

        port = int(opt("port", 0)) or _free_port()
        self._base_url = f"http://127.0.0.1:{port}"

        cmd = [
            binary,
            "--model", model,
            "--mmproj", mmproj,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--n-gpu-layers", str(int(opt("n_gpu_layers", 99))),
            "--ctx-size", str(int(opt("context_size", 16384))),
            "--threads", str(int(opt("threads", 8))),
            # Multiple images per request need the batch to be large enough to
            # hold their embeddings, or the server rejects the prompt.
            "--batch-size", str(int(opt("batch_size", 2048))),
            "--ubatch-size", str(int(opt("ubatch_size", 512))),
        ]
        if opt("flash_attn", False):
            cmd.append("--flash-attn")
        cmd += [str(x) for x in opt("extra_args", [])]

        log_dir = pathlib.Path(self.ctx.job_dir or ".") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Held open for the server subprocess's lifetime and closed in
        # teardown(), so a context manager is not usable here.
        log_path = log_dir / f"llama-server-{port}.log"
        self._log_handle = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

        env = dict(os.environ)
        cache = pathlib.Path(self.ctx.cache_dir or "~/.cache/llama").expanduser()
        env.setdefault("LLAMA_CACHE", str(cache))

        try:
            self._proc = subprocess.Popen(
                cmd, stdout=self._log_handle, stderr=subprocess.STDOUT, env=env
            )
        except OSError as exc:
            raise EngineUnavailable(f"cannot start {binary}: {exc}") from exc

        self._wait_for_health(float(opt("startup_timeout", 600)))
        self._schema = vc.events_schema(self.ctx.ethogram, self.ctx.target_behaviors())
        self._ready = True

    def _resolve_binary(self, binary: str) -> str:
        found = shutil.which(binary) or (binary if pathlib.Path(binary).is_file() else None)
        if not found:
            raise EngineUnavailable(
                f"{binary!r} is not on PATH. Build llama.cpp with "
                "`setup/install_llamacpp.sh`, or set the engine option "
                "'server_binary' to its full path."
            )
        return found

    def _wait_for_health(self, timeout: float) -> None:
        """Poll /health until the model is loaded.

        A 7B GGUF on a 2080 Ti takes minutes to load from cold NAS storage, so
        the default timeout is generous; what we must not do is start sending
        requests into a server that is still mapping weights.
        """
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise EngineUnavailable(
                    f"llama-server exited with code {self._proc.returncode} during "
                    f"startup; see logs/llama-server-*.log"
                )
            try:
                with urllib.request.urlopen(f"{self._base_url}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last = str(exc)
            time.sleep(2.0)
        raise EngineUnavailable(
            f"llama-server did not become healthy within {timeout:.0f}s ({last})"
        )

    def teardown(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._ready = False

    # -- analysis -----------------------------------------------------------
    def analyse(self, media: MediaInfo) -> list[Event]:
        opt = self.ctx.opt
        etho = self.ctx.ethogram
        targets = self.ctx.target_behaviors()
        if not media.has_video:
            return []

        windows = vc.plan_windows(media.duration, float(opt("window_seconds", 60.0)),
                                  float(opt("window_overlap_seconds", 5.0)))
        fps = float(opt("fps", 0.5))
        max_frames = int(opt("max_frames_per_window", 24))
        max_side = int(opt("max_image_side", 512))
        obs_id = os.path.splitext(os.path.basename(media.path))[0]
        min_conf = float(opt("min_confidence", 0.0))

        events: list[Event] = []
        for i, (w_start, w_stop) in enumerate(windows):
            times = vc.window_frame_times(w_start, w_stop, fps, max_frames)
            frames = extract_frames(media.path, times, max_side=max_side)
            if not frames:
                continue
            window = vc.Window(index=i, start=w_start, stop=w_stop, frames=frames)
            text = self._complete(window, etho, targets, media.duration, obs_id)
            if text is None:
                continue
            parsed, warns = vc.parse_response(text, etho, window,
                                              min_confidence=min_conf, source=self.name)
            events.extend(parsed)
            for w in warns:
                self._warn(w)
        return events

    def _complete(self, window, etho, targets: Sequence[str], duration: float,
                  obs_id: str) -> str | None:
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

        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": vc.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": float(self.ctx.opt("temperature", 0.0)),
            "max_tokens": int(self.ctx.opt("max_tokens", 1536)),
            "stream": False,
        }
        if self.ctx.opt("guided_json", True):
            # llama.cpp compiles a JSON schema to a GBNF grammar, which gives
            # the same hard guarantee about behaviour codes as vLLM's guided
            # decoding does.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "abc_events",
                                "schema": vc.schema_for_decoding(self._schema),
                                "strict": True},
            }

        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        timeout = float(self.ctx.opt("request_timeout", 900))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            self._warn(f"window {window.index}: llama-server returned {exc.code}: {detail}")
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._warn(f"window {window.index}: request failed: {exc}")
            return None

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            self._warn(f"window {window.index}: unexpected response shape")
            return None

    def _warn(self, message: str) -> None:
        bucket = self.ctx.options.setdefault("_warnings", [])
        if len(bucket) < 200:
            bucket.append(message)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["model"] = os.path.basename(str(self.ctx.opt("model_path", "")))
        d["backend"] = "llama.cpp"
        return d
