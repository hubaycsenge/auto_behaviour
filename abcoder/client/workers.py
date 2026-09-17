"""Background work for the GUI.

Every server call -- probing 50 videos, uploading 3 GB, polling SLURM -- happens
on a worker thread. The rule the rest of the client depends on: nothing touches
a Qt widget from inside :meth:`Worker.run`; results come back as signals.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class WorkerSignals(QObject):
    """Signals a :class:`Worker` can emit. Must live on a QObject."""

    finished = Signal(object)          # the callable's return value
    failed = Signal(str, str)          # message, traceback
    progress = Signal(str, float)      # status text, 0..1 (negative = indeterminate)


class Worker(QRunnable):
    """Run one callable off the GUI thread.

    The callable may accept a ``progress`` keyword; if it does, it is handed a
    function that emits :attr:`WorkerSignals.progress`.
    """

    def __init__(self, fn: Callable[..., Any], *args,
                 wants_progress: bool = False, **kwargs) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._fn = fn
        self._args = args
        self._kwargs = dict(kwargs)
        if wants_progress:
            self._kwargs["progress"] = self._report

    def _report(self, message: str, fraction: float = -1.0) -> None:
        self.signals.progress.emit(str(message), float(fraction))

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - a worker crash must not kill the GUI
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            self.signals.finished.emit(result)


def run_async(
    pool: QThreadPool,
    fn: Callable[..., Any],
    *args,
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[str, str], None] | None = None,
    on_progress: Callable[[str, float], None] | None = None,
    wants_progress: bool = False,
    **kwargs,
) -> Worker:
    """Start *fn* on *pool* and wire its signals up. Returns the worker."""
    worker = Worker(fn, *args, wants_progress=wants_progress or on_progress is not None, **kwargs)
    if on_done:
        worker.signals.finished.connect(on_done)
    if on_error:
        worker.signals.failed.connect(on_error)
    if on_progress:
        worker.signals.progress.connect(on_progress)
    pool.start(worker)
    return worker
