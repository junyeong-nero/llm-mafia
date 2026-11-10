from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess

from src.providers.chat_client import ProviderError


def load_api_key(variable: str) -> str:
    load_project_env_file()
    api_key = os.getenv(variable, "").strip()
    if api_key:
        return api_key

    command = os.getenv(f"{variable}_COMMAND", "").strip()
    if command:
        # Explicit opt-in, like Pi's !command credentials, without invoking a shell.
        try:
            result = subprocess.run(
                shlex.split(command), capture_output=True, text=True, timeout=15, check=True,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            # Never include command output, which can contain credentials.
            raise ProviderError(f"{variable}_COMMAND failed") from None
        api_key = result.stdout.strip()
        if api_key:
            return api_key
        raise ProviderError(f"{variable}_COMMAND returned an empty key")
    raise ProviderError(f"{variable} is not set")


def load_project_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
