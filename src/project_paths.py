"""Locations of repository resources and generated output."""

from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = SOURCE_DIR / "SO101"
OFFSETS_PATH = MODEL_DIR / "so101_offsets.json"
