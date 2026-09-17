"""Deciding whether the client and the server can see the same files.

This is the one question that decides whether a job needs an upload step. ABC
answers it empirically rather than by configuration: the client writes a probe
file into a candidate shared directory and asks the server to read it back. If
the server sees the same bytes, the filesystem is shared and the videos never
move.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import uuid
from dataclasses import dataclass

#: Directories that are shared on a typical HPC deployment. Checked in order;
#: the first one the probe succeeds in is used.
DEFAULT_SHARED_ROOTS = ("~/", "/nas/home", "/home", "/shared", "/data")

PROBE_PREFIX = "_abc_probe_"


@dataclass
class ProbeToken:
    """A file the client wrote and the server is asked to verify."""

    path: str
    digest: str

    @classmethod
    def write(cls, directory: str | os.PathLike) -> ProbeToken:
        d = pathlib.Path(directory).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        payload = uuid.uuid4().hex.encode()
        p = d / f"{PROBE_PREFIX}{uuid.uuid4().hex[:8]}"
        p.write_bytes(payload)
        return cls(path=str(p), digest=hashlib.sha256(payload).hexdigest())

    def verify(self) -> bool:
        """Run on the *server*: does this path hold the bytes the client wrote?"""
        p = pathlib.Path(self.path)
        try:
            return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == self.digest
        except OSError:
            return False

    def cleanup(self) -> None:
        with contextlib.suppress(OSError):
            pathlib.Path(self.path).unlink(missing_ok=True)


def looks_shared(path: str | os.PathLike, shared_roots=DEFAULT_SHARED_ROOTS) -> bool:
    """Cheap pre-check: is *path* under a directory that is usually shared?

    Only a hint for the UI -- :class:`ProbeToken` is what actually decides.
    """
    try:
        resolved = pathlib.Path(path).expanduser().resolve()
    except OSError:
        return False
    for root in shared_roots:
        try:
            base = pathlib.Path(root).expanduser().resolve()
        except OSError:
            continue
        if resolved == base or base in resolved.parents:
            return True
    return False


def map_to_client(server_path: str, server_root: str, client_root: str) -> str:
    """Translate a server-side path into the client's view of the same file.

    Used when the two machines mount the same storage at different mount points
    (``/nas/home/x`` on the cluster, ``Z:\\x`` or ``/Volumes/nas/x`` on a
    workstation). Returns *server_path* unchanged when it is not under
    *server_root*, which is the caller's signal that the mapping does not apply.
    """
    if not server_root or not client_root:
        return server_path
    sp = pathlib.PurePosixPath(server_path.replace("\\", "/"))
    sr = pathlib.PurePosixPath(str(server_root).replace("\\", "/"))
    try:
        rel = sp.relative_to(sr)
    except ValueError:
        return server_path
    return str(pathlib.PurePath(client_root) / pathlib.PurePath(*rel.parts))


def map_to_server(client_path: str, client_root: str, server_root: str) -> str:
    """Inverse of :func:`map_to_client`."""
    if not client_root or not server_root:
        return client_path
    cp = pathlib.PurePath(client_path)
    cr = pathlib.PurePath(client_root)
    try:
        rel = cp.relative_to(cr)
    except ValueError:
        return client_path
    return str(pathlib.PurePosixPath(str(server_root).replace("\\", "/")).joinpath(*rel.parts))


def unique_dir(parent: str | os.PathLike, name: str) -> pathlib.Path:
    """``parent/name``, suffixed with -2, -3 ... until it does not exist."""
    parent = pathlib.Path(parent)
    candidate = parent / name
    n = 2
    while candidate.exists():
        candidate = parent / f"{name}-{n}"
        n += 1
    return candidate
