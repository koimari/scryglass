"""Synthetic fixtures and test adapters for L2 build-time validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration import fit_calibration
from .types import CalibrationState, CandidateFit, EvalRow, MatchPrediction


def _owned_fixture_sha256(filename: str) -> str:
    return hashlib.sha256(
        Path(__file__).with_name(filename).read_bytes()
    ).hexdigest()


SYNTHETIC_CANDIDATE_ARTIFACT_SHA256 = _owned_fixture_sha256("pipeline.py")
SYNTHETIC_BASELINE_ARTIFACT_SHA256 = _owned_fixture_sha256("metrics.py")
SYNTHETIC_TRANSFORM_SHA256 = _owned_fixture_sha256("calibration.py")
SYNTHETIC_RUNTIME_MANIFEST_SHA256 = _owned_fixture_sha256("types.py")


def build_synthetic_rows() -> list[EvalRow]:
    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    default_roles = ("top", "jungle", "mid", "bot", "support")
    rows: list[EvalRow] = []

    # 12 resolved rows across unique short series so the fixture split remains
    # atomic under row-wise folding while still exercising all sealed holdout
    # classes and league/event variance.
    for idx, league in enumerate(["lpl", "lec", "lcs", "lpl", "lec", "lcs", "lpl", "lec", "lcs", "lpl", "lec", "lcs"]):
        series_id = f"series-{idx:02d}"
        patch = "26.13" if idx < 4 else ("26.14" if idx < 8 else "26.15")
        as_of = base + timedelta(days=idx * 2, hours=-2)
        event_start = base + timedelta(days=idx * 2)
        row = EvalRow(
            row_id=f"row-{idx:02d}",
            series_id=series_id,
            series_resolved=True,
            event_start=event_start,
            patch_id=patch,
            league_id=league,
            league_tier="tier1",
            region="emea" if league == "lec" else "na",
            as_of=as_of,
            label=idx % 2,
            feature_values={"feature_core": float((idx % 5) + 1) / 10.0, "future_signal": 0.0},
            feature_available_at={
                "feature_core": as_of + timedelta(minutes=10),
                "future_signal": as_of + timedelta(minutes=15),
            },
            roster_id=f"roster-{idx // 3}",
            roster_snapshot_id=f"snap-{idx // 3}",
            roster_snapshot_time=base + timedelta(days=idx * 2 - 1, hours=-1),
            roster_snapshot_stage="operational",
            is_international_event=idx in {5, 9},
            international_event_id="msi-2026" if idx in {5} else ("ewc-2026" if idx in {9} else None),
            is_roster_change=idx % 3 == 0,
            champion_ids=("Aatrox", "Ahri", "Leona", "Neeko", "Akali"),
            is_sparse_champion=idx in {9, 10},
            metadata={
                "baseline_probability": 0.42 + 0.05 * (idx % 4),
                "source_id": f"source-{idx:02d}",
                "roster_roles": list(default_roles),
                **({"masked_champion_residual": True} if idx == 2 else {}),
                **({"true_new_champion": True} if idx == 9 else {}),
                **({"archetype_transfer": True} if idx in {9, 10} else {}),
            },
        )
        rows.append(row)

    # A real second map from the same series. Series-level split construction
    # must keep both row identities atomic despite distinct event timestamps.
    rows.append(
        replace(
            rows[4],
            row_id="row-04-map-2",
            event_start=rows[4].event_start + timedelta(hours=1),
            as_of=rows[4].as_of + timedelta(hours=1),
            feature_available_at={
                name: available_at + timedelta(hours=1)
                for name, available_at in rows[4].feature_available_at.items()
            },
            label=1,
            metadata={
                **dict(rows[4].metadata),
                "source_id": "source-04-map-2",
                "baseline_probability": 0.57,
            },
        )
    )

    # Preserve independent sealed calibration support for roster-change suites.
    rows[11] = replace(
        rows[11],
        roster_id="roster-4",
        roster_snapshot_id="snap-4",
    )

    # One unresolved row to validate unresolved-safe primary inference gate.
    unresolved = EvalRow(
        row_id="row-unresolved-01",
        series_id="series-unresolved",
        series_resolved=False,
        event_start=base + timedelta(days=30),
        patch_id="26.15",
        league_id="lpl",
        league_tier="tier1",
        region="na",
        as_of=base + timedelta(days=30, hours=-1),
        label=1,
        feature_values={"feature_core": 0.95, "future_signal": 1.0},
        feature_available_at={
            "feature_core": base + timedelta(days=30, minutes=5),
            "future_signal": base + timedelta(days=29, hours=12),
        },
        roster_id="roster-unresolved",
        roster_snapshot_id="snap-u",
        roster_snapshot_time=base + timedelta(days=29, hours=-1),
        roster_snapshot_stage="operational",
        is_international_event=True,
        international_event_id="ewc-2026",
        is_roster_change=True,
        champion_ids=("Kai'Sa", "Neeko", "Neeko", "Nautilus", "Thresh"),
        is_sparse_champion=True,
        metadata={
            "baseline_probability": 0.51,
            "source_id": "source-u01",
            "roster_roles": list(default_roles),
        },
    )
    rows.append(unresolved)

    # A paired unresolved row to support invariant tests if needed.
    unresolved_partner = EvalRow(
        row_id="row-unresolved-02",
        series_id="series-unresolved",
        series_resolved=False,
        event_start=base + timedelta(days=31),
        patch_id="26.15",
        league_id="lpl",
        league_tier="tier1",
        region="na",
        as_of=base + timedelta(days=31, hours=-1),
        label=0,
        feature_values={"feature_core": 0.90, "future_signal": 0.5},
        feature_available_at={
            "feature_core": base + timedelta(days=31, minutes=5),
            "future_signal": base + timedelta(days=30, hours=6),
        },
        roster_id="roster-unresolved",
        roster_snapshot_id="snap-u",
        roster_snapshot_time=base + timedelta(days=30, hours=-1),
        roster_snapshot_stage="operational",
        is_international_event=False,
        is_roster_change=True,
        champion_ids=("Kai'Sa", "Neeko", "Neeko", "Nautilus", "Thresh"),
        is_sparse_champion=True,
        metadata={
            "baseline_probability": 0.49,
            "source_id": "source-u02",
            "roster_roles": list(default_roles),
        },
    )
    rows.append(unresolved_partner)

    # Additional rows with deterministic pair metadata for invariant checks.
    role_pair_a = EvalRow(
        row_id="row-role-a",
        series_id="series-role-a",
        series_resolved=True,
        event_start=base + timedelta(days=40),
        patch_id="26.15",
        league_id="lec",
        league_tier="tier1",
        region="emea",
        as_of=base + timedelta(days=40, hours=-1),
        label=1,
        feature_values={"feature_core": 0.77, "future_signal": 0.1},
        feature_available_at={
            "feature_core": base + timedelta(days=40, minutes=-1),
            "future_signal": base + timedelta(days=40, minutes=-1),
        },
        roster_id="roster-role",
        roster_snapshot_id="snap-role",
        roster_snapshot_time=base + timedelta(days=39, hours=-1),
        roster_snapshot_stage="operational",
        is_international_event=False,
        is_roster_change=False,
        champion_ids=("Ezreal", "Kai'Sa", "Azir", "Lux", "Jinx"),
        is_sparse_champion=False,
        metadata={
            "baseline_probability": 0.64,
            "role_invariance_pair": "row-role-b",
            "roster_roles": list(default_roles),
        },
    )
    role_pair_b = EvalRow(
        row_id="row-role-b",
        series_id="series-role-a",
        series_resolved=True,
        event_start=base + timedelta(days=42),
        patch_id="26.15",
        league_id="lec",
        league_tier="tier1",
        region="emea",
        as_of=base + timedelta(days=42, hours=-1),
        label=1,
        feature_values={"feature_core": 0.77, "future_signal": 0.1},
        feature_available_at={
            "feature_core": base + timedelta(days=42, minutes=-1),
            "future_signal": base + timedelta(days=42, minutes=-1),
        },
        roster_id="roster-role",
        roster_snapshot_id="snap-role",
        roster_snapshot_time=base + timedelta(days=41, hours=-1),
        roster_snapshot_stage="operational",
        is_international_event=False,
        is_roster_change=False,
        champion_ids=("Ezreal", "Kai'Sa", "Azir", "Lux", "Jinx"),
        is_sparse_champion=False,
        metadata={
            "baseline_probability": 0.61,
            "role_invariance_pair": "row-role-a",
            "roster_roles": list(default_roles),
        },
    )
    rows.extend([role_pair_a, role_pair_b])

    side_swap_a = EvalRow(
        row_id="row-side-a",
        series_id="series-side-a",
        series_resolved=True,
        event_start=base + timedelta(days=44),
        patch_id="26.15",
        league_id="lcs",
        league_tier="tier1",
        region="na",
        as_of=base + timedelta(days=44, hours=-1),
        label=0,
        feature_values={"feature_core": 0.66, "future_signal": 0.2},
        feature_available_at={
            "feature_core": base + timedelta(days=44, minutes=-1),
            "future_signal": base + timedelta(days=44, minutes=-1),
        },
        roster_id="roster-side",
        roster_snapshot_id="snap-side",
        roster_snapshot_time=base + timedelta(days=43, hours=-1),
        roster_snapshot_stage="operational",
        is_international_event=False,
        is_roster_change=False,
        champion_ids=("Azir", "Kai'Sa", "Sylas", "Kaisa", "Morgana"),
        is_sparse_champion=False,
        metadata={
            "baseline_probability": 0.52,
            "side_swap_pair": "row-side-b",
            "roster_roles": list(default_roles),
        },
    )
    side_swap_b = EvalRow(
        row_id="row-side-b",
        series_id="series-side-b",
        series_resolved=True,
        event_start=base + timedelta(days=46),
        patch_id="26.16",
        league_id="lcs",
        league_tier="tier1",
        region="na",
        as_of=base + timedelta(days=46, hours=-1),
        label=1,
        feature_values={"feature_core": 0.34, "future_signal": 0.2},
        feature_available_at={
            "feature_core": base + timedelta(days=46, minutes=-1),
            "future_signal": base + timedelta(days=46, minutes=-1),
        },
        roster_id="roster-side",
        roster_snapshot_id="snap-side",
        roster_snapshot_time=base + timedelta(days=45, hours=-1),
        roster_snapshot_stage="operational",
        is_international_event=False,
        is_roster_change=False,
        champion_ids=("Azir", "Kai'Sa", "Sylas", "Kaisa", "Morgana"),
        is_sparse_champion=False,
        metadata={
            "baseline_probability": 0.48,
            "side_swap_pair": "row-side-a",
            "roster_roles": list(default_roles),
        },
    )
    rows.extend([side_swap_a, side_swap_b])

    return rows


def make_model_snapshot(rows: Sequence[EvalRow]) -> dict[str, str]:
    return {row.row_id: row.fingerprint() for row in rows}


def _prediction_from_row(row: EvalRow, probability: float, model_version: str) -> MatchPrediction:
    p = float(max(0.0, min(1.0, probability)))
    import math

    logit: float | None = None
    if 0 < p < 1:
        logit = float(math.log(p / (1 - p)))
    return MatchPrediction(
        row_id=row.row_id,
        model_version=model_version,
        mode="terminal",
        raw_logit=logit,
        raw_probability=p,
        calibrated_probability=None,
        lower_95=max(0.0, p - 0.1),
        upper_95=min(1.0, p + 0.1),
        ledger={"base": 0.0, "champions": float(logit or 0.0)},
        context={"row_patch": row.patch_id, "baseline_probability": float(row.metadata.get("baseline_probability", 0.5))},
    )


@dataclass(frozen=True)
class ToyAdapter:
    adapter_id: str = "toy-v2"
    adapter_version: str = "0.0.1"
    source_tree_sha256: str = "source-tree-0000000000000000000000000000000000000000000000000000000000000000"
    runtime_artifact_sha256: str = SYNTHETIC_CANDIDATE_ARTIFACT_SHA256
    served_transform_sha256: str = SYNTHETIC_TRANSFORM_SHA256
    serialized_transform_sha256: str = SYNTHETIC_TRANSFORM_SHA256
    runtime_transform_manifest_sha256: str = SYNTHETIC_TRANSFORM_SHA256
    runtime_artifact_manifest_sha256: str = SYNTHETIC_CANDIDATE_ARTIFACT_SHA256
    terminal_probability_wording_approved: bool = True
    prefix_probability_wording_approved: Mapping[str, bool] = field(
        default_factory=lambda: {"slot_1": True, "slot_2": True}
    )

    def fit(self, rows: Sequence[EvalRow], *, split_name: str) -> CandidateFit:
        if not rows:
            return CandidateFit(
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                fit_digest="empty",
            )
        score = sum(row.label for row in rows) / len(rows)
        return CandidateFit(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            fit_digest=f"fit-{split_name}-{score:.5f}",
        )

    def _base_probability(self, row: EvalRow) -> float:
        return 0.5 + 0.06 * float(row.feature_values.get("feature_core", 0.0)) - 0.01 * float(row.metadata.get("roster_age", 0.0))

    def predict(
        self,
        state: CandidateFit,
        rows: Sequence[EvalRow],
        *,
        mode: str = "terminal",
        prefix: str | None = None,
    ) -> tuple[MatchPrediction, ...]:
        payload: list[MatchPrediction] = []
        for row in rows:
            p = max(0.0, min(1.0, self._base_probability(row)))
            if mode == "terminal":
                probability = p
            else:
                probability = min(0.95, max(0.05, p))
            if prefix is not None:
                context = {"prefix": prefix}
            else:
                context = {}
            prediction = _prediction_from_row(row, probability, self.adapter_version)
            prediction = MatchPrediction(
                row_id=prediction.row_id,
                model_version=prediction.model_version,
                mode=mode,
                raw_logit=prediction.raw_logit,
                raw_probability=prediction.raw_probability,
                calibrated_probability=prediction.calibrated_probability,
                lower_95=prediction.lower_95,
                upper_95=prediction.upper_95,
                ledger=dict(prediction.ledger),
                context={"row_patch": row.patch_id, **(prediction.context), **context},
            )
            payload.append(prediction)
        return tuple(payload)

    def fit_calibration(
        self,
        rows: Sequence[EvalRow],
        predictions: Sequence[MatchPrediction],
        *,
        mode: str = "terminal",
    ) -> CalibrationState:
        if not rows:
            return CalibrationState(
                kind="identity",
                intercept=0.0,
                slope=1.0,
                model_sha256=self.adapter_id,
            )
        return fit_calibration(predictions, tuple(int(row.label) for row in rows), self.adapter_id, kind="identity")

    def apply_calibration(
        self,
        state: CalibrationState,
        predictions: Sequence[MatchPrediction],
        *,
        mode: str = "terminal",
    ) -> tuple[MatchPrediction, ...]:
        from .calibration import calibrate_logits

        return calibrate_logits(tuple(predictions), state)

    def runtime_predict(
        self,
        state: CandidateFit,
        rows: Sequence[EvalRow],
        *,
        mode: str = "terminal",
        prefix: str | None = None,
    ) -> tuple[MatchPrediction, ...]:
        return self.predict(state, rows, mode=mode, prefix=prefix)


@dataclass(frozen=True)
class LeakyAdapter(ToyAdapter):
    """Deliberately leaked adapter used as a rejection sentinel."""

    def predict(
        self,
        state: CandidateFit,
        rows: Sequence[EvalRow],
        *,
        mode: str = "terminal",
        prefix: str | None = None,
    ) -> tuple[MatchPrediction, ...]:
        return tuple(
            _prediction_from_row(
                row,
                float(row.label),
                self.adapter_version,
            )
            for row in rows
        )


@dataclass(frozen=True)
class MismatchTransformAdapter(ToyAdapter):
    """Adapter with deliberately mismatched served-vs-serialized transform ids."""

    serialized_transform_sha256: str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


@dataclass(frozen=True)
class PrefixRejectAdapter(ToyAdapter):
    prefix_probability_wording_approved: Mapping[str, bool] = field(
        default_factory=lambda: {"slot_1": False, "slot_2": True}
    )


@dataclass(frozen=True)
class TerminalRejectAdapter(ToyAdapter):
    terminal_probability_wording_approved: bool = False


@dataclass(frozen=True)
class SnapshotRowsAdapter(ToyAdapter):
    """Adapter that exposes a pre-bundled row snapshot and snapshot_id."""

    snapshot_id: str = "snapshot-eval-v2"
    source_tree_sha256: str = "0" * 64
    rows_payload: tuple[EvalRow, ...] = ()

    def rows(self) -> Sequence[EvalRow]:
        return self.rows_payload


@dataclass(frozen=True)
class TransferPredictionsAdapter:
    """Synthetic holder for L6-provided ontology-free and transfer-ablation baselines."""

    adapter_id: str = "transfer-baseline-v1"
    adapter_version: str = "1.0.0"
    source_tree_sha256: str = "a" * 64
    snapshot_sha256: str = "107b9fc2ccef8db99e7eed3faca56d7024a9e8f13fd9c63fad997e7d6c2843fc"
    ontology_free_probabilities: Mapping[str, float] = field(default_factory=dict)
    transfer_ablation_probabilities: Mapping[str, float] = field(default_factory=dict)

    def predict_ontology_free(self, rows: Sequence[EvalRow]) -> Mapping[str, float]:
        return {
            row.row_id: self.ontology_free_probabilities[row.row_id]
            for row in rows
            if row.row_id in self.ontology_free_probabilities
        }

    def predict_transfer_ablation(self, rows: Sequence[EvalRow]) -> Mapping[str, float]:
        return {
            row.row_id: self.transfer_ablation_probabilities[row.row_id]
            for row in rows
            if row.row_id in self.transfer_ablation_probabilities
        }
