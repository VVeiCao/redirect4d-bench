"""Helpers for finding auxiliary Redirect4D-Bench conda environments."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _python_in_env(env_root: Path) -> Path | None:
    exe = env_root / ("python.exe" if os.name == "nt" else "bin/python")
    return exe if exe.exists() else None


def conda_env_python(env_name: str) -> Path | None:
    """Return the Python executable for a named conda env, if it is visible."""

    conda = shutil.which("conda")
    if conda:
        try:
            result = subprocess.run(
                [conda, "env", "list", "--json"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for env in json.loads(result.stdout).get("envs", []):
                env_root = Path(env)
                if env_root.name == env_name:
                    return _python_in_env(env_root)
        except Exception:
            pass

    current = Path(sys.executable).resolve()
    candidates: list[Path] = []
    if "envs" in current.parts:
        envs_idx = current.parts.index("envs")
        envs_root = Path(*current.parts[: envs_idx + 1])
        candidates.append(envs_root / env_name)
    candidates.append(current.parents[1] / "envs" / env_name)

    for candidate in candidates:
        exe = _python_in_env(candidate)
        if exe is not None:
            return exe
    return None


def resolve_python(
    explicit: str | None,
    *,
    env_var: str,
    conda_env: str,
    fallback: str | None = None,
) -> str:
    """Resolve a tool Python from explicit arg, env var, conda env, then fallback."""

    if explicit:
        return str(Path(explicit).expanduser())
    if os.environ.get(env_var):
        return os.environ[env_var]
    inferred = conda_env_python(conda_env)
    if inferred is not None:
        return str(inferred)
    return fallback or sys.executable
