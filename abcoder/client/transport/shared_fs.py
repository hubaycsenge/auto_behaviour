"""Transport for the case where client and server see the same filesystem.

This is the good case and the one to aim for: the videos stay exactly where the
researcher put them, nothing is copied, nothing has to be deleted afterwards,
and a 3 GB session costs no transfer time at all. On this cluster it applies
whenever the client runs on a machine with ``/nas/home`` mounted.

Commands still go through a shell, because submitting to SLURM needs a login
node even when the storage is local. When the client is *on* a login node that
shell is the local one.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

from .base import CommandResult, Transport, TransportError


@dataclass
class SharedFilesystemTransport(Transport):
    """Run commands locally and treat server paths as local paths.

    *ssh_host* is used only when this machine cannot submit to SLURM itself; in
    that case commands are forwarded to the login node while files continue to
    be written directly.
    """

    ssh_host: str = ""
    ssh_user: str = ""
    ssh_port: int = 22

    @property
    def kind(self) -> str:
        return "shared"

    @property
    def description(self) -> str:
        where = f" (commands via {self._ssh_target()})" if self.ssh_host else ""
        return f"Shared filesystem, jobs in {self.jobs_root}{where}"

    def _ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host

    # -- commands -----------------------------------------------------------
    def run(self, args: Sequence[str], timeout: float = 120.0) -> CommandResult:
        if self.ssh_host:
            command = " ".join(_quote(a) for a in args)
            args = ["ssh", "-p", str(self.ssh_port), "-o", "BatchMode=yes",
                    self._ssh_target(), command]
        try:
            proc = subprocess.run(list(args), capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except FileNotFoundError as exc:
            raise TransportError(f"command not found: {args[0]} ({exc})") from exc
        except subprocess.TimeoutExpired as exc:
            raise TransportError(f"command timed out after {timeout:.0f}s: {args[0]}") from exc
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)

    # -- files --------------------------------------------------------------
    def put_file(self, local, remote, progress=None) -> None:
        source = pathlib.Path(local).expanduser()
        target = pathlib.Path(remote).expanduser()
        if source.resolve() == target.resolve():
            return  # already in place: the entire point of this transport
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if progress:
            size = target.stat().st_size
            progress(size, size)

    def get_file(self, remote, local) -> None:
        source = pathlib.Path(remote).expanduser()
        target = pathlib.Path(local).expanduser()
        if source.resolve() == target.resolve():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def write_text(self, remote: str, text: str) -> None:
        path = pathlib.Path(remote).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def read_text(self, remote: str) -> str:
        return pathlib.Path(remote).expanduser().read_text(encoding="utf-8")

    def exists(self, remote: str) -> bool:
        return pathlib.Path(remote).expanduser().exists()

    def makedirs(self, remote: str) -> None:
        pathlib.Path(remote).expanduser().mkdir(parents=True, exist_ok=True)

    def listdir(self, remote: str) -> list[str]:
        path = pathlib.Path(remote).expanduser()
        return sorted(p.name for p in path.iterdir()) if path.is_dir() else []


def _quote(value: str) -> str:
    import shlex
    return shlex.quote(str(value))
