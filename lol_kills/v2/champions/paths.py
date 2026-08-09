"""File-system paths for v2 champion ontology artifacts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "lol" / "v2" / "champions"

DEFAULT_ONTOLOGY_PATH = DATA_DIR / "champion-ontology-seed.json"
DEFAULT_SOURCE_PATH = DATA_DIR / "champion-ontology-sources.json"
DEFAULT_REVIEW_PATH = DATA_DIR / "champion-review-log.jsonl"
DEFAULT_FIXTURE_PATH = DATA_DIR / "evaluation-fixtures.json"
