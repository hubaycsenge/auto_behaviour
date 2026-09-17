"""Engine registry.

Engines are imported lazily: the client lists them and edits their options on a
laptop with no torch installed, and only the compute node that actually runs one
pays for importing vLLM.
"""

from __future__ import annotations

import importlib

from .base import Engine, EngineContext, EngineUnavailable

#: engine name -> "module:ClassName", resolved on first use.
_REGISTRY: dict[str, str] = {
    "mock": "abcoder.server.engines.mock:MockEngine",
    "vlm_vllm": "abcoder.server.engines.vlm_vllm:VllmEngine",
    "vlm_llamacpp": "abcoder.server.engines.vlm_llamacpp:LlamaCppEngine",
    "audio": "abcoder.server.engines.audio:AudioEngine",
    "pose": "abcoder.server.engines.pose:PoseEngine",
}


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


def register(name: str, target: str) -> None:
    """Add a third-party engine, e.g. ``register("simba", "mylab.simba:Engine")``."""
    _REGISTRY[name] = target


def load_engine_class(name: str) -> type[Engine]:
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown engine {name!r}; available: {', '.join(available_engines())}"
        ) from None
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_engine(name: str, ctx: EngineContext) -> Engine:
    return load_engine_class(name)(ctx)


__all__ = [
    "Engine", "EngineContext", "EngineUnavailable",
    "available_engines", "register", "load_engine_class", "build_engine",
]
