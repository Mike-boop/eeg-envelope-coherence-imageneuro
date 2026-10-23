import importlib.util
import json
from pathlib import Path

import numpy as np

from transcription.config import load_config
from transcription.textgrid import write_textgrid


def load_transcribe_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "transcriptions" / "transcribe.py"
    spec = importlib.util.spec_from_file_location("transcribe_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_config_reads_raw_transcripts_dir(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"raw_transcripts_dir": "data/raw_transcripts"})
    )

    config = load_config(config_path)

    assert config["raw_transcripts_dir"] == str((tmp_path / "data/raw_transcripts").resolve())
    assert config["elevenlabs"]["api_key"] == ""


def test_write_textgrid_creates_utterance_and_word_tiers(tmp_path):
    output_path = tmp_path / "example.TextGrid"
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Hallo wereld",
            "words": [
                {"word": "Hallo", "start": 0.0, "end": 0.5},
                {"word": "wereld", "start": 0.5, "end": 1.0},
            ],
        }
    ]

    write_textgrid(output_path, segments, sr=16000, duration=1.0)
    text = output_path.read_text()

    assert 'name = "utterance"' in text
    assert 'name = "words"' in text
    assert 'text = "Hallo"' in text
    assert 'text = "wereld"' in text


def test_load_config_uses_environment_token(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-token")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"raw_transcripts_dir": "data/raw_transcripts"}))

    config = load_config(config_path)

    assert config["elevenlabs"]["api_key"] == "test-token"


def test_discover_audio_files_uses_precomputed_wavs(tmp_path):
    module = load_transcribe_script_module()
    (tmp_path / "example.wav").write_bytes(b"RIFF")
    (tmp_path / "example.npz").write_bytes(b"npz")

    found_paths = module.discover_audio_files(tmp_path)

    assert [path.name for path in found_paths] == ["example.wav"]
