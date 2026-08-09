"""Stable source identifiers across Oracle's Elixir transport formats."""

from __future__ import annotations

from typing import Any


_TRANSPORT_PREFIXES = (
    "oe-api:",
    "oracle-elixir-api:",
)


def canonical_source_game_key(value: Any, fallback: Any = None) -> str:
    """Return one source game key across annual and API rows."""

    for candidate in (value, fallback):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text or text.casefold() in {"nan", "nat", "none", "<na>"}:
            continue
        changed = True
        while changed:
            changed = False
            lowered = text.casefold()
            for prefix in _TRANSPORT_PREFIXES:
                if lowered.startswith(prefix):
                    text = text[len(prefix) :].strip()
                    changed = True
                    break
        if text:
            return text
    return ""


__all__ = ["canonical_source_game_key"]
