"""
Tiny helpers that hide OS quirks - import anywhere.

WIN is True when running on Windows (native or Windows-container).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Tuple

WIN: bool = os.name == "nt"

# ──────────────────────────────── Conda helpers ──────────────────────────────
def conda_cmd() -> str:
    """Return the correct executable name for Conda on this OS."""
    return "conda.exe" if WIN else "conda"


def norm_env_path(path: str | Path) -> str:
    """
    Conda accepts forward-slashes on every platform *except* when the
    path begins with a Windows drive-letter.  This helper converts the
    env path so it is always understood by `conda run -p …`.
    """
    p = Path(path)
    return str(p) if WIN else p.as_posix()


# ───────────────────────── GPU memory helpers (Windows-friendly) ─────────────
def gpu_mem_info(gpu_id: int | None) -> Tuple[int, int]:
    """
    Return *(total_mb, free_mb)* for *gpu_id*.

    Uses **nvidia-smi**, which is present with NVIDIA Windows drivers.
    If the query fails or *gpu_id* is None, returns *(0, 0)*.
    """
    if gpu_id is None or gpu_id < 0:
        return (0, 0)

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if gpu_id >= len(lines):
            return (0, 0)

        total_str, free_str = re.split(r"\s*,\s*", lines[gpu_id])
        return (int(total_str), int(free_str))
    except Exception:          # noqa: BLE001
        return (0, 0)


# ───────────────────────────── local-IPC address helper ──────────────────────
def ipc_address(port: int) -> str | tuple[str, int]:
    """
    Return the address that both the worker and the replica use for
    **fast, intra-host IPC**:

    • **Windows**  →  Named-pipe path (AF_UNIX handset).  
    • **Linux/BSD**→  `/tmp` UNIX-domain socket.  
    • Fallback     →  (`"127.0.0.1"`, *port*) loop-back TCP.
    """
    # Windows 10 1803+ exposes Named Pipes through AF_UNIX sockets.
    if WIN and hasattr(socket, "AF_UNIX"):
        return fr"\\.\pipe\citadel_{port}"

    # POSIX - classic domain socket
    if hasattr(socket, "AF_UNIX"):
        return f"/tmp/citadel_{port}.sock"

    # Very old platforms - keep the TCP path
    return ("127.0.0.1", port)
