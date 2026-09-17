"""Site configuration and hardware-aware engine presets.

The defaults below encode what the NIPG cluster actually is, so a user who
never opens a config file still gets a job that lands on a GPU that can run the
engine they picked. Everything is overridable from ``~/.config/abc/config.json``
(client) or ``$ABC_CONFIG`` (server) -- see :func:`load_config`.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
from typing import Any

# --------------------------------------------------------------------------
# hardware map
# --------------------------------------------------------------------------
# Compute capability decides what an engine can run, and the two research
# documents this system was built from are emphatic about the cliff edges:
#   sm_80+ (A100, 3090, A4000)  -- vLLM with bf16 and FlashAttention-2.
#   sm_75  (2080 Ti, Titan RTX) -- no bf16, no FA2. Qwen3-VL is unusable under
#                                  vLLM here; llama.cpp GGUF is the escape hatch.
#   sm_61  (GTX 1080/1080 Ti)   -- below vLLM's floor entirely. llama.cpp and
#                                  classical CV models only.
NODE_CLASSES: dict[str, dict[str, Any]] = {
    "ampere_a100": {
        "nodes": "nipg38", "partition": "extralarge", "gres": "gpu:a100",
        "vram_gb": 40, "count": 8, "sm": 80, "bf16": True, "vllm": True,
    },
    "ampere_consumer": {
        "nodes": "nipg10,nipg32", "partition": "medium", "gres": "gpu",
        "vram_gb": 16, "count": 6, "sm": 86, "bf16": True, "vllm": True,
    },
    "turing_24g": {
        "nodes": "nipg7,nipg34,nipg35,nipg36", "partition": "large", "gres": "gpu:titanrtx",
        "vram_gb": 24, "count": 8, "sm": 75, "bf16": False, "vllm": False,
    },
    "turing_11g": {
        "nodes": "nipg6,nipg30,nipg31,nipg33", "partition": "medium", "gres": "gpu:2080ti",
        "vram_gb": 11, "count": 13, "sm": 75, "bf16": False, "vllm": False,
    },
    "pascal": {
        "nodes": "nipg3,nipg4,nipg5", "partition": "small", "gres": "gpu:1080",
        "vram_gb": 8, "count": 6, "sm": 61, "bf16": False, "vllm": False,
    },
}


# --------------------------------------------------------------------------
# engine presets
# --------------------------------------------------------------------------

def _slurm(partition: str, gres: str, *, cpus: int = 8, mem: str = "48G",
           time: str = "12:00:00", nodelist: str = "", throttle: int = 4) -> dict[str, Any]:
    return {
        "partition": partition, "gres": gres, "nodelist": nodelist, "exclude": "",
        "cpus_per_task": cpus, "mem": mem, "time": time, "account": "", "qos": "",
        "array_throttle": throttle, "extra_sbatch": [],
    }


ENGINE_PRESETS: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------------- mock --
    "mock": {
        "label": "Mock (no GPU, for testing)",
        "description": (
            "Emits deterministic pseudo-events without looking at the pixels. "
            "Use it to check the whole client/server/BORIS round trip before "
            "spending GPU hours."
        ),
        "options": {"events_per_minute": 4.0, "seed": 0},
        "slurm": _slurm("small", "", cpus=2, mem="4G", time="00:30:00", throttle=8),
    },
    # ------------------------------------------------------------ vlm_vllm --
    "vlm_vllm": {
        "label": "Video-LLM via vLLM (A100 / Ampere)",
        "description": (
            "Samples frames with timestamps and asks a Qwen-VL model for events "
            "in strict JSON (guided decoding, temperature 0). The primary "
            "zero-training coder. Needs compute capability >= 8.0, so it runs "
            "on nipg38, nipg10 and nipg32 only."
        ),
        "options": {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.90,
            # Sampling: accuracy peaks around 96-256 frames per request and
            # degrades beyond that, so a long video is split into windows
            # rather than sampled ever more sparsely.
            "fps": 1.0,
            "max_frames_per_window": 96,
            "window_seconds": 120.0,
            "window_overlap_seconds": 5.0,
            "max_image_side": 768,
            "temperature": 0.0,
            "max_tokens": 2048,
            "guided_json": True,
            "self_consistency_samples": 1,
            "min_confidence": 0.0,
            "prompt_style": "ethogram",
            "extra_instructions": "",
        },
        "slurm": _slurm("extralarge", "gpu:a100:1", cpus=12, mem="96G",
                        time="12:00:00", nodelist="nipg38", throttle=4),
        "requires": {"min_sm": 80, "min_vram_gb": 24},
    },
    # -------------------------------------------------------- vlm_llamacpp --
    "vlm_llamacpp": {
        "label": "Video-LLM via llama.cpp (Turing / Pascal)",
        "description": (
            "Same prompting as the vLLM engine, but served by a local "
            "llama-server with a GGUF model and an mmproj vision projector. "
            "Slower, and it unlocks the 2080 Ti / Titan RTX / GTX 1080 nodes "
            "that vLLM cannot use at all."
        ),
        "options": {
            "server_binary": "llama-server",
            "model_path": "",          # GGUF; set by the setup script or the user
            "mmproj_path": "",         # vision projector that matches the model
            "n_gpu_layers": 99,
            "context_size": 16384,
            "port": 0,                 # 0 = pick a free port per task
            "threads": 8,
            "fps": 0.5,
            "max_frames_per_window": 24,
            "window_seconds": 60.0,
            "window_overlap_seconds": 5.0,
            "max_image_side": 512,
            "temperature": 0.0,
            "max_tokens": 1536,
            "startup_timeout": 600,
            "request_timeout": 900,
            "extra_instructions": "",
        },
        "slurm": _slurm("medium", "gpu:1", cpus=8, mem="32G", time="16:00:00", throttle=6),
        "requires": {"min_sm": 61, "min_vram_gb": 8},
    },
    # ----------------------------------------------------------- audio -----
    "audio": {
        "label": "Audio (vocalisations)",
        "description": (
            "Segments the soundtrack and classifies each vocalisation. Zero-shot "
            "scoring with CLAP against the ethogram's own descriptions when the "
            "model is available, falling back to acoustic rules (F0, "
            "harmonicity, bandwidth, duration). Cheap, CPU-or-small-GPU, and it "
            "hears what a silent frame sampler cannot."
        ),
        "options": {
            "sample_rate": 16000,
            "min_event_seconds": 0.06,
            "max_event_seconds": 4.0,
            "noise_percentile": 10.0,
            "peak_percentile": 99.0,
            "onset_fraction": 0.25,
            "offset_fraction": 0.12,
            "noise_multiplier": 2.0,
            "min_gap_seconds": 0.05,
            "classifier": "auto",          # auto | clap | rules
            "clap_model": "laion/clap-htsat-unfused",
            # The rule classifier scores decoy classes (speech, handling noise,
            # room tone) alongside the real ones and reports a posterior, so
            # this threshold is meaningful. Raise it when a person is audible.
            "rule_sharpness": 12.0,
            "rule_margin": 0.05,
            "implausible_rate_per_minute": 40.0,
            "min_confidence": 0.5,
            "emit_as": "point",            # point | state, when the ethogram allows both
            "behaviors": [],               # empty = every behaviour ABC judges vocal
        },
        "slurm": _slurm("small", "gpu:1", cpus=8, mem="24G", time="04:00:00", throttle=8),
        "requires": {"min_sm": 0, "min_vram_gb": 0},
    },
    # ------------------------------------------------------------- pose ----
    "pose": {
        "label": "Pose / proxemics",
        "description": (
            "Detects and tracks the dog, the person and the robot, then derives "
            "geometric behaviours: approach, withdrawal, orientation, proximity "
            "and immobility. No language model, so nothing is hallucinated -- "
            "but the thresholds are setup-specific and want calibrating against "
            "hand-coded video."
        ),
        "options": {
            "detector": "yolo",
            "model": "yolo11m.pt",
            "tracker": "bytetrack.yaml",
            "fps": 5.0,
            "device": "cuda",
            "conf": 0.35,
            "imgsz": 960,
            "classes": {"dog": ["dog", "cat", "horse", "sheep", "cow", "bear"],
                        "person": ["person"],
                        "robot": []},
            # [x0, y0, x1, y1] in normalised coords, when the robot is static
            "robot_roi": [],
            "smoothing_seconds": 0.6,
            "approach_speed_px_per_s": 25.0,
            "approach_min_seconds": 0.8,
            "near_fraction": 0.25,      # "close" = within this fraction of the frame diagonal
            "immobility_speed_px_per_s": 6.0,
            "immobility_min_seconds": 2.0,
            "rules": {},                # behaviour code -> rule name; see the pose engine
        },
        "slurm": _slurm("medium", "gpu:1", cpus=8, mem="32G", time="08:00:00", throttle=6),
        "requires": {"min_sm": 61, "min_vram_gb": 6},
    },
}


#: Which engine should own which kind of behaviour when several run together.
#: The hybrid arrangement both research documents recommend: let the specialised
#: perception models own what they measure directly, and leave interpretation to
#: the VLM.
DEFAULT_OWNERSHIP_HINTS: dict[str, list[str]] = {
    "audio": ["whine", "bark", "growl", "puff", "howl", "yelp", "whimper", "vocal", "grunt"],
    "pose": ["approach", "withdraw", "back", "retreat", "immobil", "freez", "proximity",
             "distance", "orient"],
}


DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        # Where jobs live. Must be on shared storage so every compute node sees it.
        "jobs_root": "~/abc_jobs",
        "python": "",                # interpreter for engine tasks; "" = the venv's own
        "venv": "~/abc_env",
        "hf_home": "~/.cache/huggingface",
        "default_partition": "small",
        "max_concurrent_jobs": 4,
    },
    "client": {
        "ssh_host": "nipg1.inf.elte.hu",
        "ssh_user": "",
        "ssh_port": 22,
        "server_abc": "abc",        # the abc command on the server
        "shared_roots": ["~/", "/nas/home"],
        "poll_seconds": 15,
        "recursive_scan": False,
        "last_source_dir": "",
        "last_project_dir": "",
    },
    "engines": {name: {"options": copy.deepcopy(p["options"]),
                       "slurm": copy.deepcopy(p["slurm"])}
                for name, p in ENGINE_PRESETS.items()},
    "hardware": NODE_CLASSES,
}


def config_path(scope: str = "client") -> pathlib.Path:
    """Where the config file lives for *scope* (``client`` or ``server``)."""
    override = os.environ.get("ABC_CONFIG")
    if override:
        return pathlib.Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return pathlib.Path(base).expanduser() / "abc" / f"{scope}.json"


def load_config(scope: str = "client") -> dict[str, Any]:
    """Load config, deep-merged over :data:`DEFAULT_CONFIG`."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = config_path(scope)
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = deep_merge(cfg, json.load(fh))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            raise RuntimeError(f"cannot read config {path}: {exc}") from exc
    return cfg


def save_config(cfg: dict[str, Any], scope: str = "client") -> pathlib.Path:
    path = config_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    return path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def engine_defaults(name: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """The effective ``{options, slurm}`` for an engine, config applied."""
    if name not in ENGINE_PRESETS:
        raise KeyError(f"unknown engine {name!r}; known: {', '.join(sorted(ENGINE_PRESETS))}")
    preset = {"options": copy.deepcopy(ENGINE_PRESETS[name]["options"]),
              "slurm": copy.deepcopy(ENGINE_PRESETS[name]["slurm"])}
    if cfg:
        preset = deep_merge(preset, cfg.get("engines", {}).get(name, {}))
    return preset


def suggest_ownership(behavior_codes, engine_names) -> dict[str, str]:
    """Guess which engine should own each behaviour, from its code and the
    keyword hints in :data:`DEFAULT_OWNERSHIP_HINTS`.

    Only ever a starting point -- the client shows the result in an editable
    table because the right answer depends on how the ethogram is worded.
    """
    active = set(engine_names)
    out: dict[str, str] = {}
    for code in behavior_codes:
        low = code.lower()
        for engine, keywords in DEFAULT_OWNERSHIP_HINTS.items():
            if engine in active and any(k in low for k in keywords):
                out[code] = engine
                break
    return out
