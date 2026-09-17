"""How the client reaches the server.

Two implementations, chosen automatically:

:class:`~abcoder.client.transport.shared_fs.SharedFilesystemTransport`
    The client can see the cluster's storage directly. Jobs are written
    straight into the jobs root and the videos are never copied.
:class:`~abcoder.client.transport.ssh.SshTransport`
    The client is somewhere else. Files move over SFTP and commands run over
    SSH.

The pair of them is why the "delete videos from server" button exists on one
path and is disabled on the other: only an upload leaves something on the
server that is safe to delete.
"""

from __future__ import annotations

import abc as _abc
import json
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class TransportError(RuntimeError):
    """Anything that stops the client from reaching the server."""


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def json(self) -> Any:
        """Parse the JSON one of the ``abc`` subcommands printed.

        Tolerates a leading banner from a login shell -- ``.bashrc`` on a
        cluster prints motd, module output and quota warnings, and none of that
        is our JSON.
        """
        text = self.stdout.strip()
        if not text:
            raise TransportError(
                f"command produced no output (exit {self.returncode})"
                + (f": {self.stderr.strip()[:400]}" if self.stderr.strip() else "")
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            bracket = text.find("[")
            if bracket != -1 and (start == -1 or bracket < start):
                start = bracket
            if start != -1:
                try:
                    return json.loads(text[start:])
                except json.JSONDecodeError:
                    pass
        raise TransportError(
            f"expected JSON from the server, got: {text[:400]}"
            + (f" | stderr: {self.stderr.strip()[:200]}" if self.stderr.strip() else "")
        )


@dataclass
class Transport(_abc.ABC):
    """The operations the client needs, regardless of where the server is."""

    #: The ``abc`` command on the server.
    abc_command: str = "abc"
    #: Root under which job directories are created.
    jobs_root: str = "~/abc_jobs"

    # -- identity -----------------------------------------------------------
    @property
    @_abc.abstractmethod
    def kind(self) -> str:
        """``"shared"`` or ``"ssh"``; shown in the UI and stored in the job."""

    @property
    @_abc.abstractmethod
    def description(self) -> str:
        """One line describing where this transport points, for the status bar."""

    @property
    def uploads(self) -> bool:
        """True when media has to be copied to the server."""
        return self.kind != "shared"

    # -- commands -----------------------------------------------------------
    @_abc.abstractmethod
    def run(self, args: Sequence[str], timeout: float = 120.0) -> CommandResult:
        """Run a command on the server and capture its output."""

    def abc(self, args: Sequence[str], timeout: float = 120.0) -> Any:
        """Run an ``abc`` subcommand and return its parsed JSON.

        ``abc_command`` may be several words -- ``python3 -m abcoder.server.cli``
        is a perfectly good way to reach an install that is not on PATH -- so it
        is split like a shell would split it.
        """
        import shlex
        result = self.run([*shlex.split(self.abc_command), *args], timeout=timeout)
        payload = result.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise TransportError(str(payload["error"]))
        return payload

    # -- files --------------------------------------------------------------
    @_abc.abstractmethod
    def put_file(self, local: str | pathlib.Path, remote: str,
                 progress: Callable[[int, int], None] | None = None) -> None:
        """Copy one file to the server."""

    @_abc.abstractmethod
    def get_file(self, remote: str, local: str | pathlib.Path) -> None:
        """Copy one file from the server."""

    @_abc.abstractmethod
    def write_text(self, remote: str, text: str) -> None:
        """Create or replace a text file on the server."""

    @_abc.abstractmethod
    def read_text(self, remote: str) -> str:
        """Read a text file from the server."""

    @_abc.abstractmethod
    def exists(self, remote: str) -> bool:
        ...

    @_abc.abstractmethod
    def makedirs(self, remote: str) -> None:
        ...

    @_abc.abstractmethod
    def listdir(self, remote: str) -> list[str]:
        ...

    def close(self) -> None:  # noqa: B027 - an optional hook, not a requirement
        """Release connections. Safe to call more than once.

        Deliberately concrete and empty: a transport with nothing to release
        should not have to implement it.
        """

    # -- convenience --------------------------------------------------------
    def remote_join(self, *parts: str) -> str:
        """Join path components the way the *server* spells paths (POSIX)."""
        cleaned = [str(p).replace("\\", "/").rstrip("/") for p in parts if str(p)]
        if not cleaned:
            return ""
        head, *tail = cleaned
        return head + ("/" if tail else "") + "/".join(t.lstrip("/") for t in tail)

    def check(self, timeout: float = 60.0) -> dict[str, Any]:
        """``abc check`` on the server: versions, GPU, which engines work."""
        return self.abc(["check"], timeout=timeout)

    def engines(self, timeout: float = 60.0) -> dict[str, Any]:
        return self.abc(["engines"], timeout=timeout)
