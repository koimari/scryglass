"""Synthetic-only adapters for exercising the isolated 52-slot runner."""

from __future__ import annotations

from typing import Any, Mapping

from .contract import G1_PINS


MONTHS = ("2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05")


def synthetic_loaders(*, incremental: bool) -> dict[str, Any]:
    rows = []
    for index, month in enumerate(MONTHS):
        split = "train" if month.startswith("2025") else ("development" if month <= "2026-03" else "validation")
        rows.append({"game_id": f"synthetic-{month}", "split": split, "calendar_month": month, "y": index % 2, "m0": 0.45 if index % 2 else 0.55})
    ids = [row["game_id"] for row in rows]

    def load_features() -> Mapping[str, Any]:
        return {"game_ids": ids}

    def load_g1_lpl_subset_crosscheck() -> Mapping[str, str]:
        return dict(G1_PINS)

    def load_availability() -> Mapping[str, Any]:
        return {"game_ids": ids}

    def load_target_m0() -> list[dict[str, Any]]:
        return rows

    def fit(slot: Mapping[str, Any], selected: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        if not incremental:
            predictions = [float(row["m0"]) for row in selected]
            return {"predictions": predictions, "optimization_start_count": 3}
        if slot["stage"] == "development" and slot["width"] != 1:
            predictions = [float(row["m0"]) for row in selected]
            return {"predictions": predictions, "optimization_start_count": 3}
        if slot["stage"] == "validation" and slot["family"] == "M8_comparator":
            predictions = [float(row["m0"]) for row in selected]
            return {"predictions": predictions, "optimization_start_count": 3}
        predictions = [0.8 if row["y"] else 0.2 for row in selected]
        return {"predictions": predictions, "optimization_start_count": 3}

    return {"load_g1_lpl_subset_crosscheck": load_g1_lpl_subset_crosscheck, "load_features": load_features, "load_fit_availability": load_availability, "load_target_m0": load_target_m0, "fit": fit}
