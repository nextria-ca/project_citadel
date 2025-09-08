"""
Light-weight .env loader that sits inside citadel_shared.

• Looks for  <citadel_shared>/<role>.env
• Ignores blank lines & #-comments
• Does **not** overwrite variables already defined in the host shell
  (so command-line/CI overrides still win).
• No external dependencies - pure std-lib.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict


def _parse_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue                         # skip malformed
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")   # trim simple quotes
        env[key] = value
    return env


def load_role_env(role: str) -> None:
    """
    Load master.env or worker.env (depending on *role*)
    **once, at process start-up**.

    Example:
        from citadel_shared.env_loader import load_role_env
        load_role_env("master")   # or "worker"
    """
    shared_root = Path(__file__).resolve().parent
    env_file = shared_root / f"{role}.env"

    if not env_file.exists():
        return                      # nothing to do

    for k, v in _parse_env_file(env_file).items():
        # Keep explicit host-level vars intact
        os.environ.setdefault(k, v)
