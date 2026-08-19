from __future__ import annotations

import os
from pathlib import Path

# Keep source-code imports anchored to the checkout.  A hosted refresh worker
# can set SCRYGLASS_RUNTIME_ROOT to a writable /tmp overlay before importing
# the ETL modules.  Local runs keep the existing checkout paths.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("SCRYGLASS_RUNTIME_ROOT", SOURCE_ROOT)).resolve()
DATA = ROOT / "data" / "lol"
WAREHOUSE_DIR = DATA / "warehouse"

# Oracle's Elixir raw CSV staging directory.
#
# The Scryglass Worker downloads the annual OE exports into a macOS
# Application Support inbox.  Reading from that inbox directly means a Google
# Drive quota block - or a one-off manual browser download - is recovered
# without copying files into the checkout.  RAW_OE_DIR is both a read path
# (annual CSV globs) and a write path (Drive downloads, plus the `archive/`
# subdirectory), so every consumer moves together.
#
# Resolution order:
#   1. SCRYGLASS_OE_RAW_DIR - explicit override (hosted worker, CI, tests)
#   2. SCRYGLASS_OE_INBOX   - override for the Worker inbox location
#   3. the default Worker inbox, when it exists on this machine
#   4. <warehouse>/raw      - in-checkout fallback (Linux hosts, CI)
WAREHOUSE_RAW_DIR = WAREHOUSE_DIR / "raw"
DEFAULT_OE_INBOX = (
    Path.home() / "Library" / "Application Support" / "Scryglass Worker" / "oe-inbox"
)
OE_INBOX_DIR = Path(
    os.environ.get("SCRYGLASS_OE_INBOX", DEFAULT_OE_INBOX)
).expanduser()


def _resolve_raw_oe_dir() -> Path:
    """Return the directory holding the annual Oracle's Elixir CSV exports."""

    override = os.environ.get("SCRYGLASS_OE_RAW_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if OE_INBOX_DIR.is_dir():
        return OE_INBOX_DIR.resolve()
    return WAREHOUSE_RAW_DIR


RAW_OE_DIR = _resolve_raw_oe_dir()
OE_RECEIPT_DIR = WAREHOUSE_DIR / "receipts" / "oracles_elixir"
PARQUET_DIR = WAREHOUSE_DIR / "parquet"
FEATURES_DIR = DATA / "features"
MODELS_DIR = DATA / "models"
SCHEMA_PATH = WAREHOUSE_DIR / "schema.json"

# Existing Leaguepedia caches
LP_GAMES = DATA / "draft_games.json"
LP_PLAYERS = DATA / "draft_players.json"
KILL_MODELS = DATA / "kill_models.json"
DRAFT_MODEL = DATA / "draft_model.json"
MARKETS_MODEL = DATA / "markets_model.json"

# Known OE Google Drive file IDs (official OE folder mirror)
OE_DRIVE_IDS = {
    "2014": "12syQsRH2QnKrQZTQQ6G5zyVeTG2pAYvu",
    "2015": "1qyckLuw0-hJM8XqFhlV9l1xAbr3H78T_",
    "2016": "1muyfpaIqk8_0BFkgLCWXDGNgWSXoPBwG",
    "2017": "11fx3nNjSYB0X8vKxLAbYOrS2Bu6avm9A",
    "2018": "1GsNetJQOMx0QJ6_FN8M1kwGvU_GPPcPZ",
    "2019": "11eKtScnZcpfZcD3w3UrD7nnpfLHvj9_t",
    "2020": "1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d",
    "2021": "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
    "2022": "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
    "2023": "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    "2024": "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    "2025": "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    "2026": "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
}

OE_FOLDER = "https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"


def _print_path(name: str) -> int:
    """Expose one resolved path to shell callers.

    The launcher used to hardcode ``<runtime>/data/lol/warehouse/raw`` while the
    Python side resolved ``RAW_OE_DIR`` independently. When the inbox appeared,
    the two disagreed: the downloader accepted the fresh inbox CSV and wrote a
    receipt binding its bytes, then the importer was handed the stale warehouse
    copy and the receipt check refused the run. One resolver removes that split
    by construction.
    """

    known = {
        "--raw-oe-dir": RAW_OE_DIR,
        "--oe-inbox-dir": OE_INBOX_DIR,
        "--warehouse-raw-dir": WAREHOUSE_RAW_DIR,
    }
    value = known.get(name)
    if value is None:
        import sys

        print(f"unknown path request: {name}", file=sys.stderr)
        print(f"expected one of: {' '.join(sorted(known))}", file=sys.stderr)
        return 64
    print(value)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin shell entry point
    import sys

    raise SystemExit(
        _print_path(sys.argv[1]) if len(sys.argv) == 2 else _print_path("")
    )
