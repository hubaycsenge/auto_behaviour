"""Transport selection.

:func:`detect_transport` answers the question the whole staging design turns
on: can the cluster read the researcher's video folder where it already sits?
It answers it by experiment, not by configuration -- the client writes a probe
file and asks the server to read it back.
"""

from __future__ import annotations

import pathlib
from typing import Any

from ...common.paths import ProbeToken, looks_shared
from .base import CommandResult, Transport, TransportError
from .shared_fs import SharedFilesystemTransport
from .ssh import SshTransport

__all__ = [
    "Transport", "TransportError", "CommandResult",
    "SharedFilesystemTransport", "SshTransport",
    "detect_transport", "filesystem_is_shared",
]


def filesystem_is_shared(transport: Transport, directory: str | pathlib.Path) -> bool:
    """Write a probe file in *directory* and ask the server to verify it.

    Returns False on any error: a transport that cannot answer is treated as
    not shared, which costs an upload but is never wrong in the dangerous
    direction.
    """
    token: ProbeToken | None = None
    try:
        token = ProbeToken.write(directory)
        payload = transport.abc(["verify-probe", token.path, token.digest], timeout=60)
        return bool(payload.get("shared"))
    except Exception:  # noqa: BLE001
        return False
    finally:
        if token is not None:
            token.cleanup()


def detect_transport(config: dict[str, Any], source_dir: str | pathlib.Path | None = None
                     ) -> tuple[Transport, bool]:
    """Build the right transport for this machine, and say whether it uploads.

    The order is deliberate. A local ``sbatch`` means this *is* a cluster
    machine, so nothing needs to move. Otherwise the client connects over SSH
    and, if the source folder looks like it might be on shared storage, probes
    to find out -- a laptop with the NAS mounted over SMB is exactly the case
    that saves hours of uploading and would otherwise be missed.
    """
    client_cfg = config.get("client", {})
    server_cfg = config.get("server", {})
    jobs_root = str(server_cfg.get("jobs_root", "~/abc_jobs"))
    abc_command = str(client_cfg.get("server_abc", "abc"))

    import shutil
    if shutil.which("sbatch"):
        transport: Transport = SharedFilesystemTransport(
            abc_command=abc_command, jobs_root=jobs_root)
        return transport, False

    ssh = SshTransport(
        abc_command=abc_command,
        jobs_root=jobs_root,
        host=str(client_cfg.get("ssh_host", "")),
        user=str(client_cfg.get("ssh_user", "")),
        port=int(client_cfg.get("ssh_port", 22)),
        key_filename=str(client_cfg.get("ssh_key", "")),
    )

    roots = tuple(client_cfg.get("shared_roots", ()))
    if source_dir and looks_shared(source_dir, roots) and filesystem_is_shared(ssh, source_dir):
        shared = SharedFilesystemTransport(
            abc_command=abc_command, jobs_root=jobs_root,
            ssh_host=ssh.host, ssh_user=ssh.user, ssh_port=ssh.port)
        return shared, False
    return ssh, True
