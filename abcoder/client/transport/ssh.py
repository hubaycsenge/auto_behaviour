"""Transport for a client that cannot see the cluster's filesystem.

Commands run over SSH and files move over SFTP. Two backends, because neither
is reliably present: paramiko is a pip install away but not standard, while
OpenSSH is on every Linux and Mac and on current Windows -- but gives no
transfer progress and needs key-based auth to be non-interactive.

paramiko is preferred when importable, since the upload progress bar matters a
great deal when the thing being uploaded is a folder of 50 MB videos.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import CommandResult, Transport, TransportError


@dataclass
class SshTransport(Transport):
    """Reach the server over SSH/SFTP."""

    host: str = ""
    user: str = ""
    port: int = 22
    key_filename: str = ""
    password: str = ""
    prefer_paramiko: bool = True

    _client: Any = field(default=None, init=False, repr=False)
    _sftp: Any = field(default=None, init=False, repr=False)
    _backend: str = field(default="", init=False, repr=False)

    # -- identity -----------------------------------------------------------
    @property
    def kind(self) -> str:
        return "ssh"

    @property
    def description(self) -> str:
        backend = self._backend or "not connected"
        return f"SSH to {self.target} ({backend}), jobs in {self.jobs_root}"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    # -- connection ---------------------------------------------------------
    def connect(self) -> None:
        if self._backend:
            return
        if not self.host:
            raise TransportError("no SSH host configured")
        if self.prefer_paramiko:
            try:
                self._connect_paramiko()
                self._backend = "paramiko"
                return
            except ImportError:
                pass
            except Exception as exc:  # noqa: BLE001 - auth/network, fall through to openssh
                raise TransportError(f"SSH connection to {self.target} failed: {exc}") from exc
        if not _which("ssh"):
            raise TransportError(
                "Neither paramiko nor the ssh command is available. "
                "Install paramiko (pip install paramiko) or OpenSSH."
            )
        self._backend = "openssh"
        # Fail fast on a bad host or missing key rather than at the first upload.
        probe = self._run_openssh(["true"], timeout=30)
        if not probe.ok:
            raise TransportError(
                f"cannot reach {self.target} over ssh: {probe.stderr.strip()[:300]}. "
                f"Set up key-based authentication (ssh-copy-id) -- ABC never prompts "
                f"for a password inside a command."
            )

    def _connect_paramiko(self) -> None:
        import paramiko

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict[str, Any] = {
            "hostname": self.host, "port": self.port,
            "username": self.user or None, "timeout": 30,
            "allow_agent": True, "look_for_keys": True,
        }
        if self.key_filename:
            kwargs["key_filename"] = str(pathlib.Path(self.key_filename).expanduser())
        if self.password:
            kwargs["password"] = self.password
        client.connect(**kwargs)
        self._client = client
        self._sftp = client.open_sftp()

    def close(self) -> None:
        for handle in (self._sftp, self._client):
            try:
                if handle is not None:
                    handle.close()
            except Exception:  # noqa: BLE001
                pass
        self._sftp = self._client = None
        self._backend = ""

    # -- commands -----------------------------------------------------------
    def run(self, args: Sequence[str], timeout: float = 120.0) -> CommandResult:
        self.connect()
        command = " ".join(shlex.quote(str(a)) for a in args)
        if self._backend == "paramiko":
            return self._run_paramiko(command, timeout)
        return self._run_openssh([command], timeout)

    def _run_paramiko(self, command: str, timeout: float) -> CommandResult:
        # A login shell is needed so the server's PATH includes the abc venv.
        _, stdout, stderr = self._client.exec_command(f"bash -lc {shlex.quote(command)}",
                                                      timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return CommandResult(stdout.channel.recv_exit_status(), out, err)

    def _run_openssh(self, command_parts: Sequence[str], timeout: float) -> CommandResult:
        command = " ".join(command_parts)
        argv = ["ssh", "-p", str(self.port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new"]
        if self.key_filename:
            argv += ["-i", str(pathlib.Path(self.key_filename).expanduser())]
        argv += [self.target, f"bash -lc {shlex.quote(command)}"]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise TransportError(f"ssh command timed out after {timeout:.0f}s") from exc
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)

    # -- files --------------------------------------------------------------
    def put_file(self, local, remote: str,
                 progress: Callable[[int, int], None] | None = None) -> None:
        self.connect()
        source = pathlib.Path(local).expanduser()
        self.makedirs(_parent(remote))
        if self._backend == "paramiko":
            # Upload to a temporary name and rename on success, so a task that
            # starts while the upload is in flight never opens a truncated file.
            staging = remote + ".part"
            callback = (lambda done, total: progress(done, total)) if progress else None
            self._sftp.put(str(source), staging, callback=callback)
            try:
                self._sftp.posix_rename(staging, remote)
            except (OSError, AttributeError):
                with contextlib.suppress(OSError):
                    self._sftp.remove(remote)
                self._sftp.rename(staging, remote)
            return
        self._scp(str(source), f"{self.target}:{remote}")
        if progress:
            size = source.stat().st_size
            progress(size, size)

    def get_file(self, remote: str, local) -> None:
        self.connect()
        target = pathlib.Path(local).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._backend == "paramiko":
            self._sftp.get(remote, str(target))
            return
        self._scp(f"{self.target}:{remote}", str(target))

    def _scp(self, source: str, target: str) -> None:
        argv = ["scp", "-P", str(self.port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new"]
        if self.key_filename:
            argv += ["-i", str(pathlib.Path(self.key_filename).expanduser())]
        argv += [source, target]
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise TransportError(f"scp failed: {proc.stderr.strip()[:300]}")

    def write_text(self, remote: str, text: str) -> None:
        self.connect()
        self.makedirs(_parent(remote))
        if self._backend == "paramiko":
            with self._sftp.open(remote, "w") as fh:
                fh.write(text)
            return
        with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False,
                                         encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_path = tmp.name
        try:
            self._scp(tmp_path, f"{self.target}:{remote}")
        finally:
            os.unlink(tmp_path)

    def read_text(self, remote: str) -> str:
        self.connect()
        if self._backend == "paramiko":
            with self._sftp.open(remote, "r") as fh:
                return fh.read().decode("utf-8", "replace")
        result = self.run(["cat", remote], timeout=120)
        if not result.ok:
            raise TransportError(f"cannot read {remote}: {result.stderr.strip()[:200]}")
        return result.stdout

    def exists(self, remote: str) -> bool:
        self.connect()
        if self._backend == "paramiko":
            try:
                self._sftp.stat(remote)
                return True
            except OSError:
                return False
        return self.run(["test", "-e", remote], timeout=30).ok

    def makedirs(self, remote: str) -> None:
        if not remote:
            return
        self.connect()
        if self._backend == "paramiko":
            parts = [p for p in remote.split("/") if p]
            path = "/" if remote.startswith("/") else ""
            for part in parts:
                path = f"{path}{part}" if path in ("", "/") else f"{path}/{part}"
                if path == "":
                    continue
                try:
                    self._sftp.stat(path)
                except OSError:
                    # A concurrent client may have created it first; fine.
                    with contextlib.suppress(OSError):
                        self._sftp.mkdir(path)
            return
        self.run(["mkdir", "-p", remote], timeout=60)

    def listdir(self, remote: str) -> list[str]:
        self.connect()
        if self._backend == "paramiko":
            try:
                return sorted(self._sftp.listdir(remote))
            except OSError:
                return []
        result = self.run(["ls", "-1", remote], timeout=60)
        if not result.ok:
            return []
        return sorted(line for line in result.stdout.splitlines() if line.strip())


def _parent(remote: str) -> str:
    return remote.rsplit("/", 1)[0] if "/" in remote else ""


def _which(name: str) -> bool:
    import shutil
    return bool(shutil.which(name))
