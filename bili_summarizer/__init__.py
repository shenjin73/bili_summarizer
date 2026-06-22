"""Bili Summarizer - Summarize Bilibili video content using AI."""

import os
from pathlib import Path

__version__ = "0.1.0"

# --- Load .env on package import ---
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _val = _line.partition("=")
        _key, _val = _key.strip(), _val.strip().strip('"').strip("'")
        if _key and _key not in os.environ:
            os.environ[_key] = _val
