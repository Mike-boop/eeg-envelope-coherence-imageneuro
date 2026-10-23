from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = {
    "raw_transcripts_dir": "data/raw_transcripts",
    "whisperx": {
        "model": "large-v3",
        "language": "nl",
        "device": "cpu",
        "compute_type": "int8",
    },
    "elevenlabs": {
        "api_key": "",
        "model": "scribe_v1",
        "language_code": "nl",
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from JSON, falling back to defaults."""
    path = Path(config_path) if config_path else Path("src/package_data/config.json")
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        merged = DEFAULT_CONFIG.copy()
        if isinstance(loaded, dict):
            merged.update(loaded)
            for section in ("whisperx", "elevenlabs"):
                if section in loaded and isinstance(loaded[section], dict):
                    merged[section] = {**DEFAULT_CONFIG[section], **loaded[section]}
        if isinstance(merged.get("raw_transcripts_dir"), str):
            merged["raw_transcripts_dir"] = str((path.parent / merged["raw_transcripts_dir"]).resolve())
        if isinstance(merged.get("elevenlabs"), dict):
            api_key = merged["elevenlabs"].get("api_key")
            if not api_key:
                api_key = os.getenv("ELEVENLABS_API_KEY", "")
            merged["elevenlabs"]["api_key"] = api_key
        return merged

    return DEFAULT_CONFIG.copy()
