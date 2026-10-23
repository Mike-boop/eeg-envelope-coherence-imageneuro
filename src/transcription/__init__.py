"""Transcription pipeline package for generating TextGrid files and transcripts."""

from .config import load_config
from .textgrid import write_textgrid

__all__ = ["load_config", "write_textgrid"]
