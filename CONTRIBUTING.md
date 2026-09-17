# Contributing

## Running the tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -t .
```

They need no third-party packages — that is deliberate, so the suite runs on a
bare login node. Tests that need real media or the example project skip
themselves when it is absent.

## Adding an analysis engine

An engine turns one media file plus an ethogram into a list of `Event`s. That
is the whole contract; SLURM, staging, fusion and BORIS are handled around it.

```python
from abcoder.server.engines.base import Engine
from abcoder.common.events import Event, MediaInfo

class MyEngine(Engine):
    name = "mine"
    label = "My engine"
    needs_gpu = True

    def setup(self):
        # Load models here, not in analyse(): setup() runs once per SLURM
        # array task, analyse() once per video.
        self._model = load_it()
        self._ready = True

    def analyse(self, media: MediaInfo) -> list[Event]:
        return [Event(behavior="Bark", start=12.5, subject="Dog", confidence=0.8)]

    def teardown(self):
        self._model = None
        self._ready = False
```

Then register it and give it defaults:

```python
from abcoder.server.engines import register
register("mine", "mypackage.myengine:MyEngine")
```

Add an entry to `ENGINE_PRESETS` in `abcoder/common/config.py` with a `label`,
a `description`, its `options`, its SLURM request and any hardware `requires` —
the client builds its engine picker from that, so an engine without a preset is
invisible in the GUI.

Things the base class already does for you: catching exceptions per video so
one bad file cannot kill a shard, offsetting times across multi-file
observations, and running every event through the cleaning pipeline. Do not
reimplement those.

Raise `EngineUnavailable` from `setup()` when the engine cannot run here —
missing model, wrong GPU, absent dependency. It is reported as a readable
message rather than a stack trace, because it is nearly always a deployment
problem the user can fix.

## House style

- Comments explain **why**, not what. If a line needs a comment to say what it
  does, rewrite the line.
- Errors should say what to do next. `"llama-server is not on PATH (run
  setup/install_llamacpp.sh)"` beats `"FileNotFoundError"`.
- Never import a heavy framework at module import time. The client inspects
  projects on machines with no torch, no Qt and no codecs; engines import their
  dependencies inside `setup()`.
- `abc check` must stay fast and must never block — it runs on every client
  connect. Probe with `importlib.util.find_spec` and `nvidia-smi`; do not
  import the frameworks you are reporting on.
- Anything that writes a file the user might be reading writes to a temporary
  path and renames.

## Claims about accuracy

Do not add a claim about how well an engine performs without a measurement to
back it. `docs/validation.md` sets out the protocol. Where a method has a known
limitation — the audio rule classifier cannot tell a bark from a voice, the
pose engine cannot read gaze — say so in the docs, in the engine's docstring,
and where practical in the output itself.
