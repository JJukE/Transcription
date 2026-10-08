"""Optional machine-local settings; explicit CLI values take precedence."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path.home() / ".cache" / "transcription" / "models"
KEYS = {"container", "container_root", "whisper_python", "qwen_python",
        "whisper_model_dir", "qwen_model_dir"}


def load_settings(path=None):
    selected = path or os.environ.get("TRANSCRIPTION_CONFIG")
    config = Path(selected).expanduser() if selected else ROOT / ".local" / "config.json"
    if not config.is_file():
        if selected:
            raise FileNotFoundError(f"Local configuration is missing: {config}")
        return {}
    data = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - KEYS:
        raise ValueError("Local configuration must be an object containing documented settings")
    for key, value in data.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Local setting {key} must be a nonempty string")
        if key == "container_root" or key.endswith("_python") or key.endswith("_model_dir"):
            if not Path(value).is_absolute():
                raise ValueError(f"Local setting {key} must be an absolute path")
    return data


def model_directory(backend, explicit=None, config_path=None):
    if explicit:
        return str(Path(explicit).expanduser())
    settings = load_settings(config_path)
    return settings.get(f"{backend}_model_dir", str(DEFAULT_MODEL_ROOT / backend))
