"""One connected, regularized player results rating.

The fit uses only completed map results and verified ten-player lineups. Each
player has one coefficient across every league, team, and competition tier.
Roster transfers and cross-circuit matches connect domestic pools. Competition
tier is never used as a rating bonus or penalty.

Map results alone cannot separate two players who never appear apart: their
design columns are identical, so a shrink-to-zero ridge hands them identical
coefficients. The ridge therefore shrinks toward a per-player performance
anchor instead of toward zero. The anchor is built from role-normalized
contribution metrics, is centered to exactly zero mean, and never adds a
competition-tier level bonus: metrics are z-scored *within* their own
(role, competition tier) pool, so tier labels only choose the comparison
group and cannot lift or lower a player's anchor on their own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from bisect import bisect_right, insort
import hashlib
import io
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - exercised by dependency failure tests
    threadpool_limits = None

from lol_kills.etl.source_keys import canonical_source_game_key


LOGIT_TO_ELO = 400.0 / math.log(10.0)
ROLE_ALIAS = {
    "top": "top",
    "jng": "jng",
    "jungle": "jng",
    "mid": "mid",
    "bot": "bot",
    "adc": "bot",
    "sup": "sup",
    "support": "sup",
    "utility": "sup",
}

# UNFITTED DEVELOPMENT DEFAULT.  These weights have never been fitted, tuned,
# or validated against any outcome; they are a deliberate equal-weight
# placeholder so the anchor stays inspectable while a weight study is pending.
# Do not describe a fit that uses them as a fitted contribution model.
PERFORMANCE_ANCHOR_METRIC_WEIGHTS: dict[str, float] = {
    "cs_per_min": 1.0,
    "gold_per_min": 1.0,
    "gold_share_pct": 1.0,
    "damage_per_min": 1.0,
    "damage_share_pct": 1.0,
    "kda_role_weighted": 1.0,
    "wards_per_min": 1.0,
    "wards_cleared_per_min": 1.0,
}
PERFORMANCE_ANCHOR_WEIGHTS_STATUS = "unfitted_development_default"

# Raw source columns `_contribution_metrics` reads. Any caller that builds the
# rating input by column projection MUST carry these, or the anchor is inert.
# `lol_kills.export.public_pack.PLAYER_CONTRIBUTION_COLUMNS` is pinned as a
# superset of this tuple by test_public_pack_projects_every_anchor_source_column.
PERFORMANCE_ANCHOR_SOURCE_COLUMNS: tuple[str, ...] = (
    "gamelength",
    "totalgold",
    "cspm",
    "dpm",
    "damageshare",
    "kills",
    "deaths",
    "assists",
    "wpm",
    "wcpm",
)
_ANCHOR_ZERO_MEAN_TOLERANCE = 1e-9
_PREFIX_CACHE_SCHEMA_VERSION = 1
_PREFIX_CACHE_SCHEMA_FINGERPRINT = (
    "robust-prefix-baseline:v1:median-mad-iqr:strict-prior-blocks:float64"
)
_PREFIX_CACHE_MISSING_DATE = np.iinfo(np.int64).min
_BASELINE_GROUP_SEPARATOR = "\x1f"
_GLOBAL_FIT_CACHE_SCHEMA_VERSION = 1
_GLOBAL_FIT_CACHE_SCHEMA_FINGERPRINT = "global-player-fit:v2:parquet-json:source-bound"


def _write_npz_level1(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a deterministic NumPy archive with fast level-one deflate."""

    with path.open("wb") as handle:
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload,
                    np.asarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.create_system = 3
                info.external_attr = 0o600 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload.getvalue())


class PrefixBaselineCacheError(RuntimeError):
    """Raised when a cached baseline cannot prove source equivalence."""


def _baseline_group(
    role: pd.Series,
    tier: pd.Series | None = None,
) -> pd.Series:
    """Build the global anchor's role/tier key.

    The global path has always kept an explicit string bucket for a missing
    tier. Preserve that bucket so an unlabelled row stays eligible for the
    global anchor without joining a labelled tier.
    """

    role_key = role.astype(str)
    if tier is None:
        return role_key
    return role_key + _BASELINE_GROUP_SEPARATOR + tier.astype(str).fillna("")


def _player_baseline_group(role: pd.Series, tier: pd.Series) -> pd.Series:
    """Build the player attribution key with nullable missing tiers.

    Player attribution has always dropped rows whose tier is unavailable.
    Nullable string concatenation preserves that fail-closed behavior.
    """

    return role.astype("string") + _BASELINE_GROUP_SEPARATOR + tier.astype("string")


@dataclass
class _PrefixBaselineEntry:
    metric_key: str
    min_obs: int
    row_ids: np.ndarray
    groups: np.ndarray
    dates: np.ndarray
    values: np.ndarray
    z: np.ndarray
    prior_count: np.ndarray


@dataclass
class _PrefixBaselineQuery:
    """Validated structural arrays shared by one normalization pass."""

    keys: tuple[tuple[str, ...], ...]
    row_ids: np.ndarray
    groups: np.ndarray
    dates: np.ndarray
    group_source_id: int
    date_source_id: int
    row_key_source_id: int
    catalog_generation: int


class PrefixBaselineCache:
    """Reuse exact robust baselines for immutable chronological prefixes.

    The cache stores outputs from the existing reference implementation. It
    never approximates the median, MAD, IQR, or z-score. A later request may
    reuse a complete prefix, or it may add rows before the cached latest date
    when every old row is unchanged and only the affected strict-prior suffix
    is rebuilt. Rows at a latest date may be partial because they cannot affect
    their own strict-prior baseline. Any source correction, deletion, or other
    unproven shape is a cache miss and is computed by the caller's reference
    path.

    When ``storage_path`` is set, the row catalog and every metric entry use a
    checksummed JSON manifest plus a NumPy payload. A source with later
    timestamp blocks extends the entry through the appended blocks. A source
    correction, deletion, schema change, or non-prefix census clears the
    loaded entries before the caller recomputes them.

    The row key must identify one player-map seat. A key normally contains the
    canonical game ID, side, player name, and normalized role. Values are
    compared by their float64 bytes, so changed source values cannot silently
    reuse an old baseline.
    """

    def __init__(
        self,
        *,
        storage_path: Path | str | None = None,
        source_identity: str | None = None,
        schema_fingerprint: str = _PREFIX_CACHE_SCHEMA_FINGERPRINT,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.manifest_path = (
            self.storage_path.with_suffix(".json")
            if self.storage_path is not None
            else None
        )
        self.payload_path = (
            self.storage_path.with_suffix(".npz")
            if self.storage_path is not None
            else None
        )
        self.source_identity = source_identity
        self.schema_fingerprint = str(schema_fingerprint)
        self._entries: dict[tuple[str, int], list[_PrefixBaselineEntry]] = {}
        self._row_ids: dict[tuple[str, ...], int] = {}
        self._catalog_generation = 0
        self._dirty = False
        self.invalidated = False
        self.invalidated_reason: str | None = None
        self.source_identity_changed = False
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.last_miss_reason: str | None = None
        if self.storage_path is not None:
            self._load()

    @property
    def persistent(self) -> bool:
        return self.storage_path is not None

    def _clear(self, reason: str) -> None:
        self._entries.clear()
        self._row_ids.clear()
        self._catalog_generation += 1
        self._dirty = True
        self.invalidated = True
        self.invalidated_reason = str(reason)

    @staticmethod
    def _array_sha256(array: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(array)
        return hashlib.sha256(contiguous.tobytes()).hexdigest()

    @staticmethod
    def _json_key(key: tuple[str, ...]) -> str:
        return json.dumps(key, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _unicode_array(values: list[str]) -> np.ndarray:
        width = max([len(value) for value in values] + [1])
        return np.asarray(values, dtype=f"<U{width}")

    @staticmethod
    def _manifest_spec(array: np.ndarray) -> dict[str, object]:
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": PrefixBaselineCache._array_sha256(array),
        }

    def _load(self) -> None:
        """Load a validated cache payload, leaving a failed cache empty."""

        assert self.manifest_path is not None
        assert self.payload_path is not None
        if not self.manifest_path.is_file() and not self.payload_path.is_file():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != _PREFIX_CACHE_SCHEMA_VERSION:
                raise PrefixBaselineCacheError("persistent cache schema version drift")
            if manifest.get("schema_fingerprint") != self.schema_fingerprint:
                raise PrefixBaselineCacheError("persistent cache schema fingerprint drift")
            saved_identity = manifest.get("source_identity")
            if (
                self.source_identity is not None
                and saved_identity is not None
                and self.source_identity != saved_identity
            ):
                # A changed source census can be an append. Row-level checks
                # below decide whether the change is append-only or a drift.
                self.source_identity_changed = True
                self._dirty = True
            if self.source_identity is None and saved_identity is not None:
                self.source_identity = str(saved_identity)
            payload_bytes = self.payload_path.read_bytes()
            expected_payload = manifest.get("payload_sha256")
            if hashlib.sha256(payload_bytes).hexdigest() != expected_payload:
                raise PrefixBaselineCacheError("persistent cache payload checksum drift")
            with np.load(self.payload_path, allow_pickle=False) as payload:
                specs = manifest.get("arrays")
                if not isinstance(specs, dict):
                    raise PrefixBaselineCacheError("persistent cache array manifest missing")
                arrays: dict[str, np.ndarray] = {}
                for name, spec in specs.items():
                    if name not in payload or not isinstance(spec, dict):
                        raise PrefixBaselineCacheError("persistent cache array missing")
                    array = np.asarray(payload[name])
                    if str(array.dtype) != spec.get("dtype") or list(array.shape) != spec.get("shape"):
                        raise PrefixBaselineCacheError("persistent cache array schema drift")
                    if self._array_sha256(array) != spec.get("sha256"):
                        raise PrefixBaselineCacheError("persistent cache array checksum drift")
                    arrays[str(name)] = array
            catalog = arrays.pop("row_catalog")
            self._row_ids = {}
            for row_id, encoded in enumerate(catalog.tolist()):
                decoded = json.loads(str(encoded))
                if not isinstance(decoded, list):
                    raise PrefixBaselineCacheError("persistent cache row catalog drift")
                self._row_ids[self._key(decoded)] = row_id
            entries = manifest.get("entries")
            if not isinstance(entries, list):
                raise PrefixBaselineCacheError("persistent cache entry manifest missing")
            loaded: dict[tuple[str, int], list[_PrefixBaselineEntry]] = {}
            for descriptor in entries:
                if not isinstance(descriptor, dict):
                    raise PrefixBaselineCacheError("persistent cache entry descriptor drift")
                metric_key = str(descriptor["metric_key"])
                min_obs = int(descriptor["min_obs"])
                prefix = str(descriptor["prefix"])
                entry = _PrefixBaselineEntry(
                    metric_key=metric_key,
                    min_obs=min_obs,
                    row_ids=arrays[f"{prefix}_row_ids"].astype(np.int64, copy=False),
                    groups=np.asarray(
                        [None if value == "" else str(value) for value in arrays[f"{prefix}_groups"].tolist()],
                        dtype=object,
                    ),
                    dates=arrays[f"{prefix}_dates"].astype(np.int64, copy=False),
                    values=arrays[f"{prefix}_values"].astype(float, copy=False),
                    z=arrays[f"{prefix}_z"].astype(float, copy=False),
                    prior_count=arrays[f"{prefix}_prior"].astype(float, copy=False),
                )
                loaded.setdefault((metric_key, min_obs), []).append(entry)
            self._entries = loaded
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, PrefixBaselineCacheError) as exc:
            self._clear(f"persistent cache load failed: {exc}")

    def flush(self) -> None:
        """Atomically serialize the validated cache for later processes."""

        if self.storage_path is None or not self._dirty:
            return
        assert self.manifest_path is not None
        assert self.payload_path is not None
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        catalog = [None] * len(self._row_ids)
        for key, row_id in self._row_ids.items():
            catalog[row_id] = self._json_key(key)
        arrays["row_catalog"] = self._unicode_array([str(value) for value in catalog])
        descriptors: list[dict[str, object]] = []
        counter = 0
        for entries in self._entries.values():
            for entry in entries:
                prefix = f"entry_{counter}"
                counter += 1
                arrays[f"{prefix}_row_ids"] = entry.row_ids.astype(np.int64, copy=False)
                arrays[f"{prefix}_groups"] = self._unicode_array(
                    ["" if value is None else str(value) for value in entry.groups.tolist()]
                )
                arrays[f"{prefix}_dates"] = entry.dates.astype(np.int64, copy=False)
                arrays[f"{prefix}_values"] = entry.values.astype(float, copy=False)
                arrays[f"{prefix}_z"] = entry.z.astype(float, copy=False)
                arrays[f"{prefix}_prior"] = entry.prior_count.astype(float, copy=False)
                descriptors.append(
                    {
                        "metric_key": entry.metric_key,
                        "min_obs": entry.min_obs,
                        "prefix": prefix,
                    }
                )
        payload_tmp = self.payload_path.with_name(self.payload_path.name + ".tmp")
        manifest_tmp = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        try:
            _write_npz_level1(payload_tmp, arrays)
            payload_bytes = payload_tmp.read_bytes()
            manifest = {
                "schema_version": _PREFIX_CACHE_SCHEMA_VERSION,
                "schema_fingerprint": self.schema_fingerprint,
                "source_identity": self.source_identity,
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "arrays": {
                    name: self._manifest_spec(array) for name, array in arrays.items()
                },
                "entries": descriptors,
            }
            manifest_tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(payload_tmp, self.payload_path)
            os.replace(manifest_tmp, self.manifest_path)
            self.invalidated = False
            self.invalidated_reason = None
            self.source_identity_changed = False
            self._dirty = False
        finally:
            for temporary in (payload_tmp, manifest_tmp):
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _key(value: object) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return tuple(sys.intern(str(part)) for part in value)
        if isinstance(value, list):
            return tuple(sys.intern(str(part)) for part in value)
        return (sys.intern(str(value)),)

    @staticmethod
    def _group(value: object) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _date(value: object) -> int | None:
        if value is None or pd.isna(value):
            return None
        return int(pd.Timestamp(value).value)

    @staticmethod
    def _same_float(left: object, right: object) -> bool:
        left_value = float(left)
        right_value = float(right)
        if np.isnan(left_value) and np.isnan(right_value):
            return True
        return np.float64(left_value).tobytes() == np.float64(right_value).tobytes()

    @staticmethod
    def _same_float_array(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compare float bytes while treating every NaN payload as equal."""

        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        same = (
            left_array.view(np.uint64) == right_array.view(np.uint64)
        )
        same |= np.isnan(left_array) & np.isnan(right_array)
        return same

    @staticmethod
    def _numeric(values: pd.Series) -> np.ndarray:
        """Convert one metric after its structural query is prepared."""

        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    def prepare_query(
        self,
        group: pd.Series,
        date: pd.Series,
        row_key: pd.Series,
    ) -> _PrefixBaselineQuery:
        """Prepare and validate structural arrays once per normalization pass."""

        if not (len(group) == len(date) == len(row_key)):
            raise PrefixBaselineCacheError(
                "baseline cache structural query length mismatch"
            )
        keys = tuple(
            PrefixBaselineCache._key(value)
            for value in row_key.to_numpy(dtype=object)
        )
        row_ids = np.empty(len(keys), dtype=np.int64)
        seen: set[tuple[str, ...]] = set()
        for key in keys:
            if key in seen:
                raise PrefixBaselineCacheError(
                    "baseline cache row key is duplicated"
                )
            seen.add(key)
        for position, key in enumerate(keys):
            row_ids[position] = self._row_ids.setdefault(key, len(self._row_ids))
        groups = np.asarray(
            [
                PrefixBaselineCache._group(value)
                for value in group.to_numpy(dtype=object)
            ],
            dtype=object,
        )
        dates = np.asarray(
            [
                _PREFIX_CACHE_MISSING_DATE
                if (value := PrefixBaselineCache._date(raw)) is None
                else value
                for raw in date.to_numpy(dtype=object)
            ],
            dtype=np.int64,
        )
        return _PrefixBaselineQuery(
            keys=keys,
            row_ids=row_ids,
            groups=groups,
            dates=dates,
            group_source_id=id(group),
            date_source_id=id(date),
            row_key_source_id=id(row_key),
            catalog_generation=self._catalog_generation,
        )

    def _refresh_query_catalog(
        self,
        prepared_query: _PrefixBaselineQuery,
        values: pd.Series,
        group: pd.Series,
        date: pd.Series,
        row_key: pd.Series,
    ) -> None:
        """Rebind a query after a fail-closed cache clear."""

        self._validate_query(prepared_query, values, group, date, row_key)
        if prepared_query.catalog_generation == self._catalog_generation:
            return
        refreshed = self.prepare_query(group, date, row_key)
        prepared_query.keys = refreshed.keys
        prepared_query.row_ids = refreshed.row_ids
        prepared_query.groups = refreshed.groups
        prepared_query.dates = refreshed.dates
        prepared_query.catalog_generation = refreshed.catalog_generation

    @staticmethod
    def _validate_query(
        prepared_query: _PrefixBaselineQuery,
        values: pd.Series,
        group: pd.Series,
        date: pd.Series,
        row_key: pd.Series,
    ) -> None:
        if not isinstance(prepared_query, _PrefixBaselineQuery):
            raise PrefixBaselineCacheError("baseline cache prepared query type drift")
        if (
            len(prepared_query.row_ids) != len(values)
            or prepared_query.group_source_id != id(group)
            or prepared_query.date_source_id != id(date)
            or prepared_query.row_key_source_id != id(row_key)
        ):
            raise PrefixBaselineCacheError("baseline cache prepared query source drift")

    def _arrays(
        self,
        values: pd.Series,
        group: pd.Series,
        date: pd.Series,
        row_key: pd.Series,
    ) -> tuple[
        tuple[tuple[str, ...], ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        prepared_query = self.prepare_query(group, date, row_key)
        return (
            prepared_query.keys,
            prepared_query.row_ids,
            prepared_query.groups,
            prepared_query.dates,
            self._numeric(values),
        )

    @staticmethod
    def _is_prefix(
        entry: _PrefixBaselineEntry,
        target_positions: np.ndarray,
        target_set: set[int],
    ) -> bool:
        if len(target_positions) == len(entry.row_ids):
            return True
        target_groups = entry.groups[target_positions]
        target_dates = entry.dates[target_positions]
        selected = np.zeros(len(entry.row_ids), dtype=bool)
        selected[target_positions] = True
        for current_group in set(target_groups.tolist()):
            if current_group is None:
                return False
            target_group_mask = target_groups == current_group
            group_target_dates = target_dates[target_group_mask]
            if (
                len(group_target_dates) == 0
                or (group_target_dates == _PREFIX_CACHE_MISSING_DATE).any()
            ):
                return False
            latest = int(group_target_dates.max())
            group_mask = entry.groups == current_group
            prior_mask = (
                group_mask
                & (entry.dates != _PREFIX_CACHE_MISSING_DATE)
                & (entry.dates < latest)
            )
            if np.any(prior_mask & ~selected):
                return False
        return True

    @staticmethod
    def _matched_positions(
        entry: _PrefixBaselineEntry,
        row_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.searchsorted(entry.row_ids, row_ids)
        known = positions < len(entry.row_ids)
        if known.any():
            known_indices = np.flatnonzero(known)
            known[known_indices] = entry.row_ids[positions[known_indices]] == row_ids[known_indices]
        return positions, known

    def _make_entry(
        self,
        values: pd.Series,
        group: pd.Series,
        date: pd.Series,
        min_obs: int,
        *,
        metric_key: str,
        row_key: pd.Series,
        z: pd.Series,
        prior_count: pd.Series,
        prepared_query: _PrefixBaselineQuery | None = None,
        numeric_values: np.ndarray | None = None,
        arrays: tuple[
            tuple[tuple[str, ...], ...],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None,
    ) -> _PrefixBaselineEntry | None:
        if arrays is not None:
            _keys, row_ids, groups, dates, numeric = arrays
        elif prepared_query is not None:
            self._refresh_query_catalog(prepared_query, values, group, date, row_key)
            _keys = prepared_query.keys
            row_ids = prepared_query.row_ids
            groups = prepared_query.groups
            dates = prepared_query.dates
            numeric = self._numeric(values) if numeric_values is None else numeric_values
        else:
            try:
                _keys, row_ids, groups, dates, numeric = self._arrays(
                    values, group, date, row_key
                )
            except PrefixBaselineCacheError:
                return None
        order = np.argsort(row_ids, kind="stable")
        return _PrefixBaselineEntry(
            metric_key=str(metric_key),
            min_obs=int(min_obs),
            row_ids=row_ids[order],
            groups=groups[order],
            dates=dates[order].astype(np.int64, copy=False),
            values=numeric[order],
            z=z.to_numpy(dtype=float)[order],
            prior_count=prior_count.to_numpy(dtype=float)[order],
        )

    def _try_append(
        self,
        values: pd.Series,
        group: pd.Series,
        date: pd.Series,
        min_obs: int,
        *,
        metric_key: str,
        row_key: pd.Series,
        block_baseline: Callable[
            [np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]
        ],
        prepared_query: _PrefixBaselineQuery | None = None,
        numeric_values: np.ndarray | None = None,
        arrays: tuple[
            tuple[tuple[str, ...], ...],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None,
    ) -> tuple[pd.Series, pd.Series] | None:
        """Extend one entry when every new row is a later timestamp block."""

        if arrays is not None:
            _keys, row_ids, groups, dates, numeric = arrays
        elif prepared_query is not None:
            try:
                self._refresh_query_catalog(prepared_query, values, group, date, row_key)
            except PrefixBaselineCacheError as exc:
                self.last_miss_reason = str(exc)
                return None
            _keys = prepared_query.keys
            row_ids = prepared_query.row_ids
            groups = prepared_query.groups
            dates = prepared_query.dates
            numeric = self._numeric(values) if numeric_values is None else numeric_values
        else:
            try:
                _keys, row_ids, groups, dates, numeric = self._arrays(
                    values, group, date, row_key
                )
            except PrefixBaselineCacheError as exc:
                self.last_miss_reason = str(exc)
                return None
        entries = self._entries.get((str(metric_key), int(min_obs)), [])
        for entry_index, entry in enumerate(entries):
            cached_positions, known = self._matched_positions(entry, row_ids)
            if int(known.sum()) != len(entry.row_ids):
                continue
            drift = False
            cached_known = cached_positions[known]
            drift = bool(
                np.any(entry.groups[cached_known] != groups[known])
                or np.any(entry.dates[cached_known] != dates[known])
                or not self._same_float_array(
                    entry.values[cached_known], numeric[known]
                ).all()
            )
            if drift:
                continue
            if known.all():
                continue
            if not known.all():
                old_dates = entry.dates[entry.dates != _PREFIX_CACHE_MISSING_DATE]
                new_dates = dates[~known]
                if len(old_dates) == 0 or (new_dates == _PREFIX_CACHE_MISSING_DATE).any():
                    continue
                latest_old = int(old_dates.max())
                if (new_dates <= latest_old).any():
                    continue

            output_z = np.full(len(values), np.nan, dtype=float)
            output_prior = np.zeros(len(values), dtype=float)
            output_z[known] = entry.z[cached_positions[known]]
            output_prior[known] = entry.prior_count[cached_positions[known]]

            for current_group in set(groups.tolist()):
                group_positions = np.flatnonzero(groups == current_group)
                if current_group is None or len(group_positions) == 0:
                    continue
                group_dates = dates[group_positions]
                if len(group_positions) > 1 and np.all(group_dates[1:] >= group_dates[:-1]):
                    order = group_positions
                else:
                    order = group_positions[np.argsort(group_dates, kind="stable")]
                group_dates = dates[order]
                starts = np.empty(len(order), dtype=bool)
                starts[0] = True
                starts[1:] = group_dates[1:] != group_dates[:-1]
                block_starts = np.flatnonzero(starts)
                present = np.isfinite(numeric[order])
                pool = numeric[order][present]
                available_all = np.concatenate(([0], np.cumsum(present)))[block_starts]
                new_block_mask = np.asarray(
                    [not known[position] for position in order[block_starts]],
                    dtype=bool,
                )
                new_blocks = np.flatnonzero(new_block_mask)
                if len(new_blocks) == 0:
                    continue
                available = available_all[new_blocks]
                block_location, block_scale = block_baseline(
                    pool, available, min_obs
                )
                block_lookup = {
                    int(block_index): offset
                    for offset, block_index in enumerate(new_blocks)
                }
                block_of_row = np.cumsum(starts) - 1
                new_local_positions = np.flatnonzero(~known[order])
                for local_position in new_local_positions:
                    target_position = order[local_position]
                    offset = block_lookup.get(int(block_of_row[local_position]))
                    if offset is None:
                        continue
                    count = int(available[offset])
                    output_prior[target_position] = float(count)
                    centre = float(block_location[offset])
                    spread = float(block_scale[offset])
                    if (
                        count < max(int(min_obs), 1)
                        or not np.isfinite(centre)
                        or not np.isfinite(spread)
                        or spread <= 0.0
                        or not np.isfinite(numeric[target_position])
                    ):
                        continue
                    output_z[target_position] = (
                        numeric[target_position] - centre
                    ) / spread

            replacement = self._make_entry(
                values,
                group,
                date,
                min_obs,
                metric_key=metric_key,
                row_key=row_key,
                z=pd.Series(output_z, index=values.index, dtype=float),
                prior_count=pd.Series(output_prior, index=values.index, dtype=float),
                arrays=(_keys, row_ids, groups, dates, numeric),
            )
            if replacement is None:
                return None
            self._entries[(str(metric_key), int(min_obs))] = [
                value for index, value in enumerate(entries) if index != entry_index
            ] + [replacement]
            self._dirty = True
            self.stores += 1
            return (
                pd.Series(output_z, index=values.index, dtype=float),
                pd.Series(output_prior, index=values.index, dtype=float),
            )
        return None

    def _try_insert(
        self,
        values: pd.Series,
        group: pd.Series,
        date: pd.Series,
        min_obs: int,
        *,
        metric_key: str,
        row_key: pd.Series,
        block_baseline: Callable[
            [np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]
        ],
        prepared_query: _PrefixBaselineQuery | None = None,
        numeric_values: np.ndarray | None = None,
        arrays: tuple[
            tuple[tuple[str, ...], ...],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None,
    ) -> tuple[pd.Series, pd.Series] | None:
        """Reuse an entry when new rows insert before its latest date.

        A source census can add an older map after a later map already exists.
        Such a change is not an append-only prefix, but it only changes strict
        prior baselines at the new map's date and later. Keep the proven rows
        before that boundary and recompute the affected suffix exactly.
        """

        if arrays is not None:
            _keys, row_ids, groups, dates, numeric = arrays
        elif prepared_query is not None:
            try:
                self._refresh_query_catalog(prepared_query, values, group, date, row_key)
            except PrefixBaselineCacheError as exc:
                self.last_miss_reason = str(exc)
                return None
            _keys = prepared_query.keys
            row_ids = prepared_query.row_ids
            groups = prepared_query.groups
            dates = prepared_query.dates
            numeric = self._numeric(values) if numeric_values is None else numeric_values
        else:
            try:
                _keys, row_ids, groups, dates, numeric = self._arrays(
                    values, group, date, row_key
                )
            except PrefixBaselineCacheError as exc:
                self.last_miss_reason = str(exc)
                return None

        entries = self._entries.get((str(metric_key), int(min_obs)), [])
        for entry_index, entry in enumerate(entries):
            cached_positions, known = self._matched_positions(entry, row_ids)
            # An insertion may add rows. It cannot delete an old row.
            if int(known.sum()) != len(entry.row_ids) or known.all():
                continue
            cached_known = cached_positions[known]
            if bool(
                np.any(entry.groups[cached_known] != groups[known])
                or np.any(entry.dates[cached_known] != dates[known])
                or not self._same_float_array(
                    entry.values[cached_known], numeric[known]
                ).all()
            ):
                continue

            new_mask = ~known
            if (
                np.any(groups[new_mask] == None)  # noqa: E711
                or np.any(dates[new_mask] == _PREFIX_CACHE_MISSING_DATE)
            ):
                continue

            output_z = np.full(len(values), np.nan, dtype=float)
            output_prior = np.zeros(len(values), dtype=float)
            output_z[known] = entry.z[cached_positions[known]]
            output_prior[known] = entry.prior_count[cached_positions[known]]

            for current_group in set(groups.tolist()):
                group_positions = np.flatnonzero(groups == current_group)
                if current_group is None or len(group_positions) == 0:
                    continue
                if not new_mask[group_positions].any():
                    continue
                group_dates = dates[group_positions]
                if len(group_positions) > 1 and np.all(
                    group_dates[1:] >= group_dates[:-1]
                ):
                    order = group_positions
                else:
                    order = group_positions[np.argsort(group_dates, kind="stable")]
                # Match _prior_baseline_z: rows without a usable date are
                # excluded from the baseline pool and retain their cached
                # neutral output.
                order = order[dates[order] != _PREFIX_CACHE_MISSING_DATE]
                if len(order) == 0:
                    continue
                ordered_dates = dates[order]
                starts = np.empty(len(order), dtype=bool)
                starts[0] = True
                starts[1:] = ordered_dates[1:] != ordered_dates[:-1]
                block_starts = np.flatnonzero(starts)
                block_of_row = np.cumsum(starts) - 1
                ordered_new = new_mask[order]
                block_has_new = np.zeros(len(block_starts), dtype=bool)
                for block_index, local_position in enumerate(block_starts):
                    end = (
                        block_starts[block_index + 1]
                        if block_index + 1 < len(block_starts)
                        else len(order)
                    )
                    block_has_new[block_index] = bool(
                        ordered_new[local_position:end].any()
                    )
                first_affected = int(np.flatnonzero(block_has_new)[0])

                present = np.isfinite(numeric[order])
                pool = numeric[order][present]
                available_all = np.concatenate(([0], np.cumsum(present)))[
                    block_starts
                ]
                available = available_all[first_affected:]
                block_location, block_scale = block_baseline(
                    pool, available, min_obs
                )
                affected_rows = np.flatnonzero(
                    block_of_row >= first_affected
                )
                for local_position in affected_rows:
                    target_position = order[local_position]
                    offset = int(block_of_row[local_position]) - first_affected
                    count = int(available[offset])
                    output_prior[target_position] = float(count)
                    centre = float(block_location[offset])
                    spread = float(block_scale[offset])
                    if (
                        count < max(int(min_obs), 1)
                        or not np.isfinite(centre)
                        or not np.isfinite(spread)
                        or spread <= 0.0
                        or not np.isfinite(numeric[target_position])
                    ):
                        continue
                    output_z[target_position] = (
                        numeric[target_position] - centre
                    ) / spread

            replacement = self._make_entry(
                values,
                group,
                date,
                min_obs,
                metric_key=metric_key,
                row_key=row_key,
                z=pd.Series(output_z, index=values.index, dtype=float),
                prior_count=pd.Series(output_prior, index=values.index, dtype=float),
                arrays=(_keys, row_ids, groups, dates, numeric),
            )
            if replacement is None:
                return None
            self._entries[(str(metric_key), int(min_obs))] = [
                value for index, value in enumerate(entries) if index != entry_index
            ] + [replacement]
            self._dirty = True
            self.stores += 1
            return (
                pd.Series(output_z, index=values.index, dtype=float),
                pd.Series(output_prior, index=values.index, dtype=float),
            )
        return None

    def lookup(
        self,
        values: pd.Series,
        group: pd.Series,
        date: pd.Series,
        min_obs: int,
        *,
        metric_key: str,
        row_key: pd.Series,
        block_baseline: Callable[
            [np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]
        ] | None = None,
        prepared_query: _PrefixBaselineQuery | None = None,
    ) -> tuple[pd.Series, pd.Series] | None:
        """Return cached z and prior counts when source equivalence is proven."""

        self.last_miss_reason = None
        try:
            if prepared_query is None:
                keys, row_ids, groups, dates, numeric = self._arrays(
                    values, group, date, row_key
                )
                query_for_append = None
            else:
                self._refresh_query_catalog(prepared_query, values, group, date, row_key)
                keys = prepared_query.keys
                row_ids = prepared_query.row_ids
                groups = prepared_query.groups
                dates = prepared_query.dates
                numeric = self._numeric(values)
                query_for_append = prepared_query
        except PrefixBaselineCacheError as exc:
            self.misses += 1
            self.last_miss_reason = str(exc)
            return None
        entries = self._entries.get((str(metric_key), int(min_obs)), [])
        if not entries:
            self.misses += 1
            self.last_miss_reason = "no_entry"
            return None
        for entry in entries:
            target_positions, known = self._matched_positions(entry, row_ids)
            missing = not bool(known.all())
            if missing:
                continue
            cached_targets = target_positions
            drift = bool(
                np.any(entry.groups[cached_targets] != groups)
                or np.any(entry.dates[cached_targets] != dates)
                or not self._same_float_array(
                    entry.values[cached_targets], numeric
                ).all()
            )
            if drift:
                continue
            target_array = np.asarray(target_positions, dtype=int)
            if not self._is_prefix(entry, target_array, set(target_positions)):
                continue
            self.hits += 1
            output_index = values.index
            return (
                pd.Series(entry.z[target_array], index=output_index, dtype=float),
                pd.Series(entry.prior_count[target_array], index=output_index, dtype=float),
            )
        if entries and block_baseline is not None:
            appended = self._try_append(
                values,
                group,
                date,
                min_obs,
                metric_key=metric_key,
                row_key=row_key,
                block_baseline=block_baseline,
                prepared_query=query_for_append,
                numeric_values=numeric,
                arrays=(keys, row_ids, groups, dates, numeric)
                if query_for_append is None
                else None,
            )
            if appended is not None:
                self.hits += 1
                return appended
            inserted = self._try_insert(
                values,
                group,
                date,
                min_obs,
                metric_key=metric_key,
                row_key=row_key,
                block_baseline=block_baseline,
                prepared_query=query_for_append,
                numeric_values=numeric,
                arrays=(keys, row_ids, groups, dates, numeric)
                if query_for_append is None
                else None,
            )
            if inserted is not None:
                self.hits += 1
                return inserted
        self.misses += 1
        self.last_miss_reason = "source_drift_or_non_prefix"
        if self.persistent:
            self._clear(self.last_miss_reason)
        return None

    def store(
        self,
        values: pd.Series,
        group: pd.Series,
        date: pd.Series,
        min_obs: int,
        *,
        metric_key: str,
        row_key: pd.Series,
        z: pd.Series,
        prior_count: pd.Series,
        prepared_query: _PrefixBaselineQuery | None = None,
    ) -> None:
        """Store reference results for a complete frame or safe prefix."""

        try:
            if prepared_query is None:
                _keys, row_ids, groups, dates, numeric = self._arrays(
                    values, group, date, row_key
                )
            else:
                self._refresh_query_catalog(prepared_query, values, group, date, row_key)
                _keys = prepared_query.keys
                row_ids = prepared_query.row_ids
                groups = prepared_query.groups
                dates = prepared_query.dates
                numeric = self._numeric(values)
        except PrefixBaselineCacheError:
            return
        order = np.argsort(row_ids, kind="stable")
        entry = _PrefixBaselineEntry(
            metric_key=str(metric_key),
            min_obs=int(min_obs),
            row_ids=row_ids[order],
            groups=groups[order],
            dates=dates[order].astype(np.int64, copy=False),
            values=numeric[order],
            z=z.to_numpy(dtype=float)[order],
            prior_count=prior_count.to_numpy(dtype=float)[order],
        )
        self._entries.setdefault((str(metric_key), int(min_obs)), []).append(entry)
        self._dirty = True
        self.stores += 1


class GlobalPlayerFitCache:
    """Persist exact global-player fit outputs by a source-bound key.

    The cache contains only private derived snapshots and fit metadata. The
    caller supplies a key that includes the cutoff-filtered source content,
    fit configuration, and implementation fingerprint. A changed source or
    implementation therefore selects a new entry. Each snapshot is stored as
    Parquet and each manifest records its schema, size, and checksum. A
    damaged entry is discarded and rebuilt by the caller. The cache keeps at
    most one entry per named slot, so shifting cutoffs cannot grow it without
    bound.
    """

    def __init__(
        self,
        *,
        storage_path: Path | str | None = None,
        schema_fingerprint: str = _GLOBAL_FIT_CACHE_SCHEMA_FINGERPRINT,
    ) -> None:
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.manifest_path = (
            self.storage_path.with_suffix(".json")
            if self.storage_path is not None
            else None
        )
        self.entries_path = (
            Path(str(self.storage_path) + ".entries")
            if self.storage_path is not None
            else None
        )
        self.schema_fingerprint = str(schema_fingerprint)
        self._entries: dict[tuple[str, str], dict[str, object]] = {}
        self._dirty = False
        self.invalidated = False
        self.invalidated_reason: str | None = None
        self.hits = 0
        self.misses = 0
        self.stores = 0
        if self.storage_path is not None:
            self._load()

    @property
    def persistent(self) -> bool:
        return self.storage_path is not None

    def _clear(self, reason: str) -> None:
        self._entries.clear()
        self._dirty = True
        self.invalidated = True
        self.invalidated_reason = str(reason)

    @staticmethod
    def _metadata_copy(meta: dict[str, Any]) -> dict[str, Any]:
        """Validate that fit metadata stays JSON-serializable."""

        try:
            return json.loads(json.dumps(meta, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise PrefixBaselineCacheError("global fit cache metadata is not JSON-safe") from exc

    @staticmethod
    def _file_sha256(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        return size, digest.hexdigest()

    @staticmethod
    def _safe_file_name(slot: str, key: str) -> str:
        digest = hashlib.sha256(f"{slot}\x1f{key}".encode("utf-8")).hexdigest()
        return f"entry_{digest}.parquet"

    def _load(self) -> None:
        assert self.manifest_path is not None
        assert self.entries_path is not None
        if not self.manifest_path.is_file() and not self.entries_path.exists():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != _GLOBAL_FIT_CACHE_SCHEMA_VERSION:
                raise PrefixBaselineCacheError("global fit cache schema version drift")
            if manifest.get("schema_fingerprint") != self.schema_fingerprint:
                raise PrefixBaselineCacheError("global fit cache schema fingerprint drift")
            descriptors = manifest.get("entries")
            if not isinstance(descriptors, list):
                raise PrefixBaselineCacheError("global fit cache entry manifest missing")
            loaded: dict[tuple[str, str], dict[str, object]] = {}
            for descriptor in descriptors:
                if not isinstance(descriptor, dict):
                    raise PrefixBaselineCacheError("global fit cache entry descriptor drift")
                slot = str(descriptor.get("slot") or "")
                key = str(descriptor.get("key") or "")
                file_name = str(descriptor.get("file") or "")
                if not slot or not key or file_name != self._safe_file_name(slot, key):
                    raise PrefixBaselineCacheError("global fit cache entry identity drift")
                path = self.entries_path / file_name
                if not path.is_file():
                    raise PrefixBaselineCacheError("global fit cache entry file missing")
                size, checksum = self._file_sha256(path)
                if size != int(descriptor.get("bytes", -1)) or checksum != descriptor.get("sha256"):
                    raise PrefixBaselineCacheError("global fit cache entry checksum drift")
                snapshot = pd.read_parquet(path)
                columns = descriptor.get("columns")
                if columns != [str(column) for column in snapshot.columns]:
                    raise PrefixBaselineCacheError("global fit cache entry columns drift")
                dtypes = descriptor.get("dtypes")
                if dtypes != [str(dtype) for dtype in snapshot.dtypes]:
                    raise PrefixBaselineCacheError("global fit cache entry dtypes drift")
                meta = descriptor.get("meta")
                validated = descriptor.get("validated")
                if not isinstance(meta, dict) or not isinstance(validated, bool):
                    raise PrefixBaselineCacheError("global fit cache entry metadata drift")
                loaded[(slot, key)] = {
                    "snapshot": snapshot,
                    "meta": self._metadata_copy(meta),
                    "validated": validated,
                    "file": file_name,
                }
            self._entries = loaded
        except Exception as exc:
            self._clear(f"global fit cache load failed: {exc}")

    def lookup(
        self,
        key: str,
        *,
        require_validated: bool = False,
        slot: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        record = None
        if slot is not None:
            record = self._entries.get((str(slot), str(key)))
        else:
            for (_slot, saved_key), candidate in self._entries.items():
                if saved_key == str(key):
                    record = candidate
                    break
        if record is None:
            self.misses += 1
            return None
        if require_validated and not bool(record.get("validated")):
            self.misses += 1
            return None
        snapshot = record.get("snapshot")
        meta = record.get("meta")
        if not isinstance(snapshot, pd.DataFrame) or not isinstance(meta, dict):
            self.misses += 1
            self._clear("global fit cache entry type drift")
            return None
        self.hits += 1
        return snapshot.copy(deep=True), self._metadata_copy(meta)

    def store(
        self,
        key: str,
        snapshot: pd.DataFrame,
        meta: dict[str, Any],
        *,
        validated: bool,
        slot: str | None = None,
    ) -> None:
        if not isinstance(snapshot, pd.DataFrame) or not isinstance(meta, dict):
            return
        saved_slot = str(slot or f"key-{str(key)[:16]}")
        saved_meta = self._metadata_copy(meta)
        self._entries[(saved_slot, str(key))] = {
            "snapshot": snapshot.copy(deep=True),
            "meta": saved_meta,
            "validated": bool(validated),
        }
        # A slot represents one semantic cutoff. Replacing it prevents an
        # append cycle from retaining an unbounded sequence of historical keys.
        self._entries = {
            identity: record
            for identity, record in self._entries.items()
            if identity[0] != saved_slot or identity[1] == str(key)
        }
        if len(self._entries) > 5:
            for identity in sorted(self._entries)[:-5]:
                self._entries.pop(identity, None)
        self._dirty = True
        self.stores += 1

    def flush(self) -> None:
        if self.storage_path is None or not self._dirty:
            return
        assert self.manifest_path is not None
        assert self.entries_path is not None
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries_path.mkdir(parents=True, exist_ok=True)
        manifest_tmp = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        descriptors: list[dict[str, object]] = []
        active_files: set[str] = set()
        try:
            for (slot, key), record in sorted(self._entries.items()):
                snapshot = record["snapshot"]
                meta = record["meta"]
                if not isinstance(snapshot, pd.DataFrame) or not isinstance(meta, dict):
                    raise PrefixBaselineCacheError("global fit cache entry type drift")
                file_name = self._safe_file_name(slot, key)
                path = self.entries_path / file_name
                temporary = path.with_name(path.name + ".tmp.parquet")
                snapshot.to_parquet(temporary, index=False)
                os.replace(temporary, path)
                size, checksum = self._file_sha256(path)
                active_files.add(file_name)
                descriptors.append(
                    {
                        "slot": slot,
                        "key": key,
                        "file": file_name,
                        "bytes": size,
                        "sha256": checksum,
                        "columns": [str(column) for column in snapshot.columns],
                        "dtypes": [str(dtype) for dtype in snapshot.dtypes],
                        "validated": bool(record.get("validated")),
                        "meta": self._metadata_copy(meta),
                    }
                )
                record["file"] = file_name
            manifest = {
                "schema_version": _GLOBAL_FIT_CACHE_SCHEMA_VERSION,
                "schema_fingerprint": self.schema_fingerprint,
                "entries": descriptors,
            }
            manifest_tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(manifest_tmp, self.manifest_path)
            for stale in self.entries_path.glob("entry_*.parquet"):
                if stale.name not in active_files:
                    stale.unlink()
            self._dirty = False
            self.invalidated = False
            self.invalidated_reason = None
        finally:
            for temporary in (manifest_tmp,):
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

# Minimum number of STRICTLY EARLIER observations a (role, competition tier)
# baseline needs before it may normalize a metric.  Below the floor the metric
# stays unavailable (NaN) for that player-map; it is never imputed.
#
# This mirrors ``lol_kills.ratings.player_elo.ATTRIBUTION_MIN_BASELINE_OBS`` and
# must stay equal to it.  It is duplicated rather than imported because
# ``player_elo`` imports this module, so importing back would be circular;
# ``test_anchor_baseline_floor_matches_the_elo_constant`` pins the two together.
ANCHOR_MIN_BASELINE_OBS = 20

# A player needs this many of their own scored maps before the anchor speaks for
# them. Guards the case where one malformed row is a large share of a player's
# record; below the floor the player stays exactly neutral and is counted.
ANCHOR_MIN_PLAYER_MAPS = 5

# Upper bound on the magnitude of any single normalized metric before it enters
# the composite.  The upstream completeness gate at
# ``lol_kills/etl/oe_database.py:548-559`` only checks finiteness, nonnegativity
# and a handful of ratio bounds, so an implausibly large but finite statistic
# survives ingestion.  Without this clip one such value would dominate the
# composite and then the player-level standardization, moving a low-sample
# player by hundreds of Elo.
ANCHOR_METRIC_Z_CLIP = 3.0

# Consistency constants for the ROBUST baseline used by ``_prior_baseline_z``.
#
# The baseline is a median and a median absolute deviation, not a mean and a
# standard deviation.  A mean/std baseline has a breakdown point of 1/n: one
# malformed but ingestible statistic (``cspm = 1e12`` clears the completeness
# gate at ``lol_kills/etl/oe_database.py:548-559``) inflates the pool's mean and
# std enough that every LATER row in that pool reads as roughly -1/sqrt(n)
# standard deviations instead of 0.  That is the classical masking failure, and
# clipping the resulting z cannot repair it because the contamination is in the
# statistic the z is measured AGAINST, not in the z.  The median and the MAD
# both have a 50% breakdown point, so a single row cannot move either.
#
# ``_MAD_TO_SIGMA`` makes the MAD a consistent estimator of sigma under
# normality; ``_IQR_TO_SIGMA`` does the same for the interquartile range.
_MAD_TO_SIGMA = 1.4826
_IQR_TO_SIGMA = 1.349


class GlobalPlayerRatingError(RuntimeError):
    """Raised when the shared player scale cannot pass its release checks."""


@dataclass(frozen=True)
class GlobalPlayerBTConfig:
    l2: float = 2.0
    side_l2: float = 0.01
    prior_rating: float = 1500.0
    holdout_fraction: float = 0.20
    minimum_maps: int = 100
    minimum_connected_share: float = 0.95
    minimum_holdout_gain: float = 0.005
    max_iterations: int = 400
    performance_anchor_scale: float = 0.15
    performance_anchor_enabled: bool = True


class _Components:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _role(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text in ROLE_ALIAS:
        return ROLE_ALIAS[text]
    for prefix, role in ROLE_ALIAS.items():
        if text.startswith(prefix):
            return role
    return None


def _canonical_game_ids(frame: pd.DataFrame) -> pd.Series:
    if "game_uid" in frame.columns:
        fallback = frame["gameid"] if "gameid" in frame.columns else None
        return pd.Series(
            [
                canonical_source_game_key(
                    value,
                    fallback.loc[index] if fallback is not None else None,
                )
                for index, value in frame["game_uid"].items()
            ],
            index=frame.index,
            dtype="string",
        )
    if "gameid" in frame.columns:
        return frame["gameid"].map(canonical_source_game_key).astype("string")
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def _complete_lineups(players: pd.DataFrame) -> dict[str, dict[str, list[tuple[str, str]]]]:
    required = {"side", "position", "playername"}
    if players is None or players.empty or not required.issubset(players.columns):
        return {}
    identity_columns = [
        column
        for column in ("game_uid", "gameid", "side", "position", "playername")
        if column in players.columns
    ]
    frame = players.loc[:, identity_columns].copy()
    frame["_game_id"] = _canonical_game_ids(frame)
    frame["_side"] = frame["side"].astype(str).str.title()
    position = frame["position"].astype("string").str.strip().str.casefold()
    frame["_role"] = position.map(ROLE_ALIAS)
    unknown_role = frame["_role"].isna()
    for prefix, role in ROLE_ALIAS.items():
        frame["_role"] = frame["_role"].where(
            ~unknown_role | ~position.str.startswith(prefix, na=False),
            role,
        )
    frame["_player"] = frame["playername"].astype("string").str.strip()
    frame = frame[
        frame["_game_id"].notna()
        & frame["_game_id"].str.strip().ne("")
        & frame["_side"].isin({"Blue", "Red"})
        & frame["_role"].notna()
        & frame["_player"].notna()
        & frame["_player"].ne("")
        & frame["_player"].str.casefold().ne("nan")
    ]
    if frame.empty:
        return {}
    order = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
    # Keep the first-seen group order while letting pandas perform the row
    # ordering and duplicate selection in one pass. This matches the old
    # per-group sort and ``setdefault`` behavior, including repeated identical
    # rows and ambiguous player aliases.
    frame["_group_order"] = frame.groupby(
        ["_game_id", "_side"], sort=False, observed=True
    ).ngroup()
    frame["_role_order"] = frame["_role"].map(order)
    frame["_player_text"] = frame["_player"].astype(str)
    frame["_player_casefold"] = frame["_player"].str.casefold()
    ordered = frame.sort_values(
        [
            "_group_order",
            "_role_order",
            "_player_casefold",
            "_player_text",
        ],
        kind="stable",
    )
    selected = ordered.drop_duplicates(
        ["_game_id", "_side", "_role"], keep="first"
    )
    group_stats = selected.groupby(
        ["_game_id", "_side"], sort=False, observed=True
    ).agg(
        role_count=("_role", "nunique"),
        player_count=("_player", "nunique"),
    )
    valid_groups = group_stats.index[
        group_stats["role_count"].eq(5) & group_stats["player_count"].eq(5)
    ]
    selected = selected.set_index(["_game_id", "_side"])
    selected = selected.loc[selected.index.isin(valid_groups)].reset_index()
    grouped = selected.groupby(
        ["_game_id", "_side"], sort=False, observed=True
    )[["_player", "_role"]].agg(list)
    output: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for (game_id, side), player_rows, role_rows in grouped.itertuples(
        index=True, name=None
    ):
        output.setdefault(str(game_id), {})[str(side)] = list(
            zip(
                (str(player) for player in player_rows),
                (str(role) for role in role_rows),
            )
        )
    return {
        game_id: sides
        for game_id, sides in output.items()
        if len(sides.get("Blue", [])) == 5 and len(sides.get("Red", [])) == 5
    }


def _model_rows(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    through: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[tuple[str, str]]]]]:
    if maps is None or maps.empty:
        return pd.DataFrame(), {}
    frame = maps.copy()
    frame["game_id"] = _canonical_game_ids(frame)
    frame["date"] = pd.to_datetime(frame.get("date"), utc=True, errors="coerce").dt.tz_localize(None)
    frame["result"] = pd.to_numeric(frame.get("y_blue_win"), errors="coerce")
    frame = frame[
        frame["game_id"].notna()
        & frame["game_id"].str.strip().ne("")
        & frame["date"].notna()
        & frame["result"].isin({0, 1})
    ]
    if through is not None:
        cutoff = pd.Timestamp(through)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        frame = frame[frame["date"].le(cutoff)]
    frame = frame.sort_values(["date", "game_id"], kind="stable").drop_duplicates("game_id", keep="last")
    lineups = _complete_lineups(players)
    frame = frame[frame["game_id"].isin(lineups)].reset_index(drop=True)
    return frame, lineups


def _design(
    frame: pd.DataFrame,
    lineups: dict[str, dict[str, list[tuple[str, str]]]],
    names: list[str],
) -> csr_matrix:
    index = {name: position for position, name in enumerate(names)}
    row_index: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_number, game_id in enumerate(frame["game_id"].astype(str)):
        for side, sign in (("Blue", 1.0), ("Red", -1.0)):
            lineup = lineups[game_id][side]
            for player, _role_name in lineup:
                row_index.append(row_number)
                columns.append(index[player])
                values.append(sign / len(lineup))
    return csr_matrix((values, (row_index, columns)), shape=(len(frame), len(names)))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Finite numeric view of one column; an absent column reads as all-NaN."""

    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.where(np.isfinite(values))


def _robust_block_baseline(
    pool: np.ndarray,
    available: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding median and robust scale for one group's blocks.

    ``pool`` holds a single group's PRESENT values in date order and
    ``available[k]`` is how many of them exist strictly before block ``k``
    begins, so ``pool[:available[k]]`` is exactly that block's baseline sample.

    The scale is the MAD rescaled by ``_MAD_TO_SIGMA``.  Its one known failure
    mode is MAD == 0, which happens whenever more than half the sample ties --
    entirely plausible for wards-per-minute pools where many rows are exactly
    0.  The fallback is the interquartile range rescaled by ``_IQR_TO_SIGMA``,
    which is still robust (25% breakdown) and survives ties the MAD cannot.

    FAIL CLOSED: a sample under the floor, or one where BOTH robust scales are
    zero or non-finite, returns NaN.  Nothing is imputed, no group mean is
    substituted, and nothing is ever divided by zero.
    """

    location = np.full(len(available), np.nan, dtype=float)
    scale = np.full(len(available), np.nan, dtype=float)
    if len(pool) == 0:
        return location, scale
    # ``expanding().median()`` is pandas' C skiplist, and it is bit-identical
    # to ``np.median(pool[:n])`` for every n (pinned by
    # ``test_expanding_median_is_bit_identical_to_numpy_median``).  Only the
    # MAD needs the per-block pass, because its deviations are taken against
    # that block's own median and so cannot be accumulated.
    running_median = pd.Series(pool).expanding().median().to_numpy(dtype=float)
    floor = max(int(min_obs), 1)
    previous = -1
    for position in range(len(available)):
        count = int(available[position])
        if count < floor:
            continue
        if count == previous:
            # A block with no present rows leaves the sample untouched.
            location[position] = location[position - 1]
            scale[position] = scale[position - 1]
            continue
        previous = count
        prefix = pool[:count]
        centre = float(running_median[count - 1])
        if not np.isfinite(centre):
            continue
        deviation = np.abs(prefix - centre)
        spread = _MAD_TO_SIGMA * float(np.median(deviation, overwrite_input=True))
        if not np.isfinite(spread) or spread <= 0.0:
            low, high = np.quantile(prefix, (0.25, 0.75))
            spread = float(high - low) / _IQR_TO_SIGMA
        if not np.isfinite(spread) or spread <= 0.0:
            continue
        location[position] = centre
        scale[position] = spread
    return location, scale


def _kth_abs_distance(
    sorted_values: list[np.float64],
    centre: np.float64,
    rank: int,
) -> np.float64:
    """Return one exact order statistic of distances from ``centre``.

    Values below the centre produce one sorted distance stream when read from
    right to left. Values at or above the centre produce the other stream when
    read from left to right. A partition search selects the requested element
    without materialising either stream.
    """

    split = bisect_right(sorted_values, centre)
    left_count = split
    right_count = len(sorted_values) - split
    lower = max(0, rank - right_count)
    upper = min(rank, left_count)
    negative_infinity = -np.inf
    positive_infinity = np.inf
    while lower <= upper:
        left_taken = (lower + upper) // 2
        right_taken = rank - left_taken
        left_previous = (
            centre - sorted_values[split - left_taken]
            if left_taken
            else negative_infinity
        )
        right_previous = (
            sorted_values[split + right_taken - 1] - centre
            if right_taken
            else negative_infinity
        )
        left_next = (
            centre - sorted_values[split - left_taken - 1]
            if left_taken < left_count
            else positive_infinity
        )
        right_next = (
            sorted_values[split + right_taken] - centre
            if right_taken < right_count
            else positive_infinity
        )
        if left_previous > right_next:
            upper = left_taken - 1
        elif right_previous > left_next:
            lower = left_taken + 1
        else:
            return min(left_next, right_next)
    raise RuntimeError("absolute-distance order statistic partition failed")


def _linear_quantile_sorted(
    sorted_values: list[np.float64],
    quantile: float,
) -> np.float64:
    """Match NumPy's default linear quantile on an already sorted prefix."""

    index = (len(sorted_values) - 1) * quantile
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def _robust_block_baseline_fast(
    pool: np.ndarray,
    available: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact robust prefixes with one sorted append-only sample.

    The reference implementation scans every prefix for each block. This
    path inserts each present value once. Raw medians use the maintained order,
    and MAD order statistics use two monotone distance streams. The IQR
    fallback uses the same linear interpolation as ``np.quantile``.

    Strictly prior blocks remain the caller's responsibility. ``available``
    contains the count before each block, so the sorted sample is updated only
    after a prior baseline has been selected for that block.
    """

    location = np.full(len(available), np.nan, dtype=float)
    scale = np.full(len(available), np.nan, dtype=float)
    if len(pool) == 0:
        return location, scale
    pool_array = np.asarray(pool, dtype=np.float64)
    if not np.isfinite(pool_array).all():
        return _robust_block_baseline(pool_array, available, min_obs)
    try:
        counts = np.asarray([int(value) for value in available], dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return _robust_block_baseline(pool_array, available, min_obs)
    if (
        (counts < 0).any()
        or (counts > len(pool_array)).any()
        or (len(counts) > 1 and (np.diff(counts) < 0).any())
    ):
        return _robust_block_baseline(pool_array, available, min_obs)
    if len(counts) <= 16 and int(counts[-1]) > 1024:
        # Append-only lookups usually ask for a handful of new blocks after a
        # large history. The reference path builds its expanding median once
        # and scans only those few MAD prefixes. That is cheaper than sorting
        # the entire history for each sparse request.
        return _robust_block_baseline(pool_array, available, min_obs)

    sorted_values: list[np.float64] = []
    inserted = 0
    previous = -1
    floor = max(int(min_obs), 1)
    for position, count_value in enumerate(counts):
        count = int(count_value)
        if count != inserted:
            # Append-only lookups pass only the new blocks. Their first
            # available count can be near the end of a large history, so
            # initialise that prefix with one C-level sort. Cold builds still
            # start at zero and retain the cheaper incremental inserts.
            if inserted == 0 and count > 1024:
                sorted_values = [
                    np.float64(value)
                    for value in np.sort(pool_array[:count], kind="stable")
                ]
            else:
                for value in pool_array[inserted:count]:
                    insort(sorted_values, np.float64(value))
            inserted = count
        if count < floor:
            continue
        if count == previous:
            location[position] = location[position - 1]
            scale[position] = scale[position - 1]
            continue
        previous = count
        if count & 1:
            centre = sorted_values[count // 2]
        else:
            centre = (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2.0
        lower_rank = (count - 1) // 2
        mad = _kth_abs_distance(sorted_values, centre, lower_rank)
        if not count & 1:
            upper_rank = count // 2
            mad = (mad + _kth_abs_distance(sorted_values, centre, upper_rank)) / 2.0
        spread = _MAD_TO_SIGMA * float(mad)
        if not np.isfinite(spread) or spread <= 0.0:
            low = _linear_quantile_sorted(sorted_values, 0.25)
            high = _linear_quantile_sorted(sorted_values, 0.75)
            spread = float(high - low) / _IQR_TO_SIGMA
        if not np.isfinite(spread) or spread <= 0.0:
            continue
        location[position] = float(centre)
        scale[position] = spread
    return location, scale


def _prior_baseline_z(
    values: pd.Series,
    group: pd.Series,
    date: pd.Series,
    min_obs: int,
    *,
    baseline_cache: PrefixBaselineCache | None = None,
    metric_key: str | None = None,
    row_key: pd.Series | None = None,
    prepared_query: object | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Robust z-score against a baseline built only from strictly earlier dates.

    Mirrors ``lol_kills.ratings.player_elo._prior_baseline_z`` (that module
    imports this one, so it cannot be imported back).  The mirror is pinned by
    ``test_prior_baseline_z_matches_the_elo_implementation``; the only
    difference is that this copy also returns the prior observation count so
    the caller can report how many cells the floor withheld.

    The baseline for a row is the expanding MEDIAN and MAD over every row in
    the same ``group`` whose date is strictly before the row's own date, so map
    ``t`` never contributes to its own baseline.  Rows sharing a timestamp form
    one block and cannot see each other.  A baseline thinner than ``min_obs``
    or with a degenerate robust spread yields an unavailable (NaN) z-score.

    Median/MAD rather than mean/std is the whole point: see ``_MAD_TO_SIGMA``.
    A single malformed row cannot move a 50%-breakdown estimator, so it cannot
    poison the baseline that every LATER row in the pool is measured against.
    """

    if baseline_cache is not None and metric_key is not None and row_key is not None:
        cached = baseline_cache.lookup(
            values,
            group,
            date,
            min_obs,
            metric_key=metric_key,
            row_key=row_key,
            block_baseline=_robust_block_baseline_fast,
            prepared_query=prepared_query,
        )
        if cached is not None:
            return cached

    index = values.index
    x = values.to_numpy(dtype=float)
    present = np.isfinite(x)

    location = np.full(len(index), np.nan, dtype=float)
    scale = np.full(len(index), np.nan, dtype=float)
    prior_count = np.zeros(len(index), dtype=float)

    work = pd.DataFrame(
        {"_g": group.to_numpy(), "_d": date.to_numpy()},
        index=pd.RangeIndex(len(index)),
    )
    work["_x"] = x
    work["_p"] = present
    # A row with no group or no date cannot be placed in the prior ordering, so
    # it is left unavailable rather than scored against a baseline it might
    # belong inside.  This is what ``dropna=True`` did for the block groupby.
    placed = work["_g"].notna().to_numpy() & work["_d"].notna().to_numpy()

    for _key, sub in work[placed].groupby("_g", sort=False):
        if sub.empty:
            continue
        sub = sub.sort_values("_d", kind="mergesort")
        positions = sub.index.to_numpy()
        dates = sub["_d"].to_numpy()
        rows_present = sub["_p"].to_numpy(dtype=bool)
        pool = sub["_x"].to_numpy(dtype=float)[rows_present]

        # One block per distinct timestamp; every row in a block shares the
        # baseline taken as of the last row STRICTLY BEFORE the block starts.
        starts = np.empty(len(sub), dtype=bool)
        starts[0] = True
        starts[1:] = dates[1:] != dates[:-1]
        block_of_row = np.cumsum(starts) - 1
        available = np.concatenate(([0], np.cumsum(rows_present)))[
            np.flatnonzero(starts)
        ]

        block_location, block_scale = _robust_block_baseline_fast(
            pool, available, min_obs
        )
        location[positions] = block_location[block_of_row]
        scale[positions] = block_scale[block_of_row]
        prior_count[positions] = available[block_of_row].astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        usable = (
            present
            & (prior_count >= float(min_obs))
            & np.isfinite(location)
            & np.isfinite(scale)
            & (scale > 0.0)
        )
        z = np.where(usable, (x - location) / scale, np.nan)
    output_z = pd.Series(z, index=index, dtype=float)
    output_prior = pd.Series(prior_count, index=index, dtype=float)
    if baseline_cache is not None and metric_key is not None and row_key is not None:
        baseline_cache.store(
            values,
            group,
            date,
            min_obs,
            metric_key=metric_key,
            row_key=row_key,
            z=output_z,
            prior_count=output_prior,
            prepared_query=prepared_query,
        )
    return output_z, output_prior


def _map_dates(frame: pd.DataFrame) -> pd.Series:
    """Authoritative map date per canonical game id, taken from the map rows."""

    return pd.Series(
        frame["date"].to_numpy(),
        index=pd.Index(frame["game_id"].astype(str), name="game_id"),
    )


def _contribution_metrics(
    players: pd.DataFrame,
    map_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Per player-map contribution metrics used to build the ridge anchor.

    Every metric is fail-closed: a missing column, a missing value, or an
    impossible denominator yields NaN for that metric on that map. Nothing is
    imputed and no league or role mean is ever substituted.

    ``map_dates`` carries the authoritative map date per canonical game id and
    is what orders the shifted/expanding baselines.  A row whose map has no
    usable date cannot be placed in that order, so it is dropped rather than
    scored against a baseline it might belong inside.
    """

    required = {"side", "position", "playername"}
    if players is None or players.empty or not required.issubset(players.columns):
        return pd.DataFrame()

    frame = pd.DataFrame(index=players.index)
    frame["_game_id"] = _canonical_game_ids(players)
    if map_dates is not None:
        dates = frame["_game_id"].astype(str).map(map_dates)
    elif "date" in players.columns:
        dates = players["date"]
    else:
        dates = pd.Series(pd.NaT, index=players.index)
    frame["_date"] = pd.to_datetime(
        pd.Series(dates, index=players.index), utc=True, errors="coerce"
    ).dt.tz_localize(None)
    frame["_side"] = players["side"].astype(str).str.title()
    frame["_role"] = players["position"].map(_role)
    frame["_player"] = players["playername"].astype("string").str.strip()
    if "competition_tier" in players.columns:
        frame["_tier"] = players["competition_tier"].astype("string").str.strip().str.casefold()
    else:
        frame["_tier"] = pd.Series(pd.NA, index=players.index, dtype="string")
    for source in PERFORMANCE_ANCHOR_SOURCE_COLUMNS:
        frame[f"_raw_{source}"] = _numeric(players, source)

    frame = frame[
        frame["_game_id"].notna()
        & frame["_game_id"].astype(str).str.strip().ne("")
        & frame["_date"].notna()
        & frame["_side"].isin({"Blue", "Red"})
        & frame["_role"].notna()
        & frame["_player"].notna()
        & frame["_player"].ne("")
        & frame["_player"].str.casefold().ne("nan")
    ]
    # One row per player and map so a duplicated feed row cannot double-weight
    # a single performance, and so team totals stay a five-player sum.
    frame = frame.drop_duplicates(["_game_id", "_player"], keep="first")
    if frame.empty:
        return pd.DataFrame()

    minutes = frame["_raw_gamelength"].where(frame["_raw_gamelength"] > 0) / 60.0
    total_gold = frame["_raw_totalgold"].where(frame["_raw_totalgold"] >= 0)
    # A share needs a complete denominator. If any seat on the team is missing
    # its gold, the team total is short and every teammate's share would be
    # silently inflated, so the whole side's share is withheld instead.
    side_keys = [frame["_game_id"], frame["_side"]]
    gold_by_side = total_gold.groupby(side_keys, dropna=False)
    team_gold = gold_by_side.transform("sum", min_count=1)
    team_gold = team_gold.where(
        gold_by_side.transform("count").eq(gold_by_side.transform("size"))
    )
    deaths = frame["_raw_deaths"].where(frame["_raw_deaths"] >= 0)

    frame["cs_per_min"] = frame["_raw_cspm"]
    frame["gold_per_min"] = total_gold / minutes
    frame["gold_share_pct"] = 100.0 * total_gold / team_gold.where(team_gold > 0)
    frame["damage_per_min"] = frame["_raw_dpm"]
    frame["damage_share_pct"] = frame["_raw_damageshare"]
    frame["kda_role_weighted"] = (
        frame["_raw_kills"] + frame["_raw_assists"]
    ) / deaths.clip(lower=1.0)
    frame["wards_per_min"] = frame["_raw_wpm"]
    frame["wards_cleared_per_min"] = frame["_raw_wcpm"]

    keep = ["_game_id", "_date", "_side", "_role", "_player", "_tier", *PERFORMANCE_ANCHOR_METRIC_WEIGHTS]
    metrics = frame[keep].copy()
    for metric in PERFORMANCE_ANCHOR_METRIC_WEIGHTS:
        values = metrics[metric].astype(float)
        metrics[metric] = values.where(np.isfinite(values))
    return metrics


def _role_normalized_composite(
    metrics: pd.DataFrame,
    *,
    baseline_cache: PrefixBaselineCache | None = None,
    _return_components: bool = False,
    _group_mode: str | None = None,
) -> tuple[Any, ...]:
    """Weighted mean of within-(role, tier) z-scores for each player-map row.

    Role normalization is what makes the anchor fair: a support's 0.87 cs/min
    is normal for a support and must not read as a bad performance.

    Each metric is normalized against a shifted/expanding baseline over
    STRICTLY EARLIER maps in the same (role, competition tier) pool, so a map
    never contributes to its own baseline and never sees a later map.  A
    baseline with fewer than ``ANCHOR_MIN_BASELINE_OBS`` prior observations
    withholds the metric entirely, and every surviving z-score is clipped to
    +/-``ANCHOR_METRIC_Z_CLIP`` before it enters the composite so no single
    malformed statistic can dominate.
    """

    group_keys = ["_role"]
    normalization = "role"
    if _group_mode == "role+competition_tier":
        group_keys = ["_role", "_tier"]
        normalization = "role+competition_tier"
    elif _group_mode == "role":
        group_keys = ["_role"]
    elif "_tier" in metrics.columns and metrics["_tier"].notna().any():
        group_keys = ["_role", "_tier"]
        normalization = "role+competition_tier"
    # Missing tier stays an explicit bucket instead of collapsing into another
    # pool or being silently discarded.
    group = _baseline_group(
        metrics[group_keys[0]],
        metrics[group_keys[1]] if len(group_keys) > 1 else None,
    )
    date = metrics["_date"]
    row_key = pd.Series(
        list(
            zip(
                metrics["_game_id"].astype(str),
                metrics["_side"].astype(str),
                metrics["_player"].astype(str),
                metrics["_role"].astype(str),
            )
        ),
        index=metrics.index,
    )
    prepared_query = (
        baseline_cache.prepare_query(group, date, row_key)
        if baseline_cache is not None
        else None
    )

    diagnostics: dict[str, Any] = {
        "baseline_min_prior_observations": int(ANCHOR_MIN_BASELINE_OBS),
        "normalized_metric_clip": float(ANCHOR_METRIC_Z_CLIP),
        "metric_cells_present": 0,
        "metric_cells_observed": 0,
        "metric_cells_withheld_below_baseline_floor": 0,
        "metric_cells_withheld_degenerate_baseline": 0,
        "metric_cells_clipped": 0,
        "normalized_metric_min": None,
        "normalized_metric_max": None,
    }

    weighted_sum = pd.Series(0.0, index=metrics.index)
    weight_total = pd.Series(0.0, index=metrics.index)
    normalized_z = pd.DataFrame(index=metrics.index)
    prior_counts = pd.DataFrame(index=metrics.index)
    observed_low: float | None = None
    observed_high: float | None = None
    for metric, weight in PERFORMANCE_ANCHOR_METRIC_WEIGHTS.items():
        if weight <= 0.0 or metric not in metrics.columns:
            continue
        values = metrics[metric]
        # The player Elo attribution path shares this cache object, but its
        # nullable-tier grouping and source scope are different. A bare metric
        # name would let one path inspect the other's entry and clear the
        # whole cache on a legitimate source difference.
        cache_metric_key = f"global:{normalization}:{metric}"
        raw_z, prior_obs = _prior_baseline_z(
            values,
            group,
            date,
            ANCHOR_MIN_BASELINE_OBS,
            baseline_cache=baseline_cache,
            metric_key=cache_metric_key,
            row_key=row_key,
            prepared_query=prepared_query,
        )
        present = values.notna() & np.isfinite(values.astype(float))
        below_floor = present & prior_obs.lt(float(ANCHOR_MIN_BASELINE_OBS))
        withheld = present & raw_z.isna()
        z = raw_z.clip(lower=-ANCHOR_METRIC_Z_CLIP, upper=ANCHOR_METRIC_Z_CLIP)
        observed = z.notna()
        normalized_z[metric] = raw_z
        prior_counts[metric] = prior_obs

        diagnostics["metric_cells_present"] += int(present.sum())
        diagnostics["metric_cells_observed"] += int(observed.sum())
        diagnostics["metric_cells_withheld_below_baseline_floor"] += int(below_floor.sum())
        diagnostics["metric_cells_withheld_degenerate_baseline"] += int(
            (withheld & ~below_floor).sum()
        )
        diagnostics["metric_cells_clipped"] += int(
            (raw_z.abs() > ANCHOR_METRIC_Z_CLIP).sum()
        )
        if observed.any():
            low = float(z[observed].min())
            high = float(z[observed].max())
            observed_low = low if observed_low is None else min(observed_low, low)
            observed_high = high if observed_high is None else max(observed_high, high)

        weighted_sum = weighted_sum + z.where(observed, 0.0) * weight
        weight_total = weight_total + observed.astype(float) * weight

    diagnostics["normalized_metric_min"] = observed_low
    diagnostics["normalized_metric_max"] = observed_high
    composite = weighted_sum / weight_total.where(weight_total > 0)
    composite = composite.where(np.isfinite(composite))
    if _return_components:
        return composite, normalization, diagnostics, normalized_z, prior_counts
    return composite, normalization, diagnostics


@dataclass
class GlobalPlayerFitWorkspace:
    """Shared immutable work for current and historical global fits.

    The lineups, contribution metrics, and strict-prior normalized values are
    independent of the fit cutoff. A cutoff only selects complete model rows
    and the already computed rows before that cutoff. The source digests keep
    the object fail-closed when a caller passes a different frame.
    """

    model_frame: pd.DataFrame
    lineups: dict[str, dict[str, list[tuple[str, str]]]]
    metrics: pd.DataFrame
    composite: pd.Series
    normalization: str
    normalized_z: pd.DataFrame
    prior_counts: pd.DataFrame
    components_by_mode: dict[str, tuple[pd.Series, str, pd.DataFrame, pd.DataFrame]]
    source_maps_digest: str
    source_players_digest: str

    @classmethod
    def build(
        cls,
        maps: pd.DataFrame,
        players: pd.DataFrame,
        *,
        baseline_cache: PrefixBaselineCache | None = None,
    ) -> "GlobalPlayerFitWorkspace":
        model_frame, lineups = _model_rows(maps, players)
        if model_frame.empty:
            metrics = pd.DataFrame()
            composite = pd.Series(dtype=float)
            normalization = "role"
            normalized_z = pd.DataFrame()
            prior_counts = pd.DataFrame()
            components_by_mode: dict[str, tuple[pd.Series, str, pd.DataFrame, pd.DataFrame]] = {}
        else:
            metrics = _contribution_metrics(players, _map_dates(model_frame))
            if metrics.empty:
                composite = pd.Series(np.nan, index=metrics.index, dtype=float)
                normalization = "role"
                normalized_z = pd.DataFrame(index=metrics.index)
                prior_counts = pd.DataFrame(index=metrics.index)
                components_by_mode = {}
            else:
                full_mode = (
                    "role+competition_tier"
                    if "_tier" in metrics.columns and metrics["_tier"].notna().any()
                    else "role"
                )
                (
                    composite,
                    normalization,
                    _diagnostics,
                    normalized_z,
                    prior_counts,
                ) = _role_normalized_composite(
                    metrics,
                    baseline_cache=baseline_cache,
                    _return_components=True,
                    _group_mode=full_mode,
                )
                components_by_mode = {
                    full_mode: (
                        composite,
                        normalization,
                        normalized_z,
                        prior_counts,
                    )
                }
        return cls(
            model_frame=model_frame,
            lineups=lineups,
            metrics=metrics,
            composite=composite,
            normalization=normalization,
            normalized_z=normalized_z,
            prior_counts=prior_counts,
            components_by_mode=components_by_mode,
            source_maps_digest=_workspace_source_digest(maps),
            source_players_digest=_workspace_source_digest(players),
        )

    def matches_source(self, maps: pd.DataFrame, players: pd.DataFrame) -> bool:
        """Prove that this workspace belongs to the requested source frame."""

        return bool(
            self.source_maps_digest == _workspace_source_digest(maps)
            and self.source_players_digest == _workspace_source_digest(players)
        )

    def frame_for(self, through: pd.Timestamp | None) -> pd.DataFrame:
        """Select a cutoff without rebuilding canonical lineups."""

        if through is None:
            return self.model_frame
        cutoff = pd.Timestamp(through)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        return self.model_frame[self.model_frame["date"].le(cutoff)].reset_index(drop=True)

    def anchor_inputs(
        self,
        names: list[str],
        game_ids: set[str],
    ) -> tuple[pd.Series, str, dict[str, Any], pd.DataFrame] | None:
        """Return exact cutoff-scoped composite values and diagnostics."""

        anchor_mask = (
            self.metrics["_game_id"].astype(str).isin(game_ids)
            & self.metrics["_player"].astype(str).isin(set(names))
        )
        scoped = self.metrics.loc[anchor_mask]
        mode = "role"
        if "_tier" in scoped.columns and scoped["_tier"].notna().any():
            mode = "role+competition_tier"
        components = self.components_by_mode.get(mode)
        if components is None:
            # A full frame can contain tier labels that are absent from an
            # earlier prefix. Reusing the full-frame mode would change the
            # historical baseline groups. The caller must run the exact
            # prefix-scoped path in that case.
            return None
        composite_source, normalization, normalized_z, prior_counts = components
        diagnostics: dict[str, Any] = {
            "baseline_min_prior_observations": int(ANCHOR_MIN_BASELINE_OBS),
            "normalized_metric_clip": float(ANCHOR_METRIC_Z_CLIP),
            "metric_cells_present": 0,
            "metric_cells_observed": 0,
            "metric_cells_withheld_below_baseline_floor": 0,
            "metric_cells_withheld_degenerate_baseline": 0,
            "metric_cells_clipped": 0,
            "normalized_metric_min": None,
            "normalized_metric_max": None,
        }
        if scoped.empty:
            return (
                pd.Series(np.nan, index=scoped.index, dtype=float),
                self.normalization,
                diagnostics,
                scoped,
            )
        observed_low: float | None = None
        observed_high: float | None = None
        for metric, weight in PERFORMANCE_ANCHOR_METRIC_WEIGHTS.items():
            if weight <= 0.0 or metric not in scoped.columns or metric not in self.normalized_z.columns:
                continue
            values = scoped[metric]
            raw_z = normalized_z.loc[scoped.index, metric]
            prior_obs = prior_counts.loc[scoped.index, metric]
            present = values.notna() & np.isfinite(values.astype(float))
            below_floor = present & prior_obs.lt(float(ANCHOR_MIN_BASELINE_OBS))
            withheld = present & raw_z.isna()
            z = raw_z.clip(lower=-ANCHOR_METRIC_Z_CLIP, upper=ANCHOR_METRIC_Z_CLIP)
            observed = z.notna()
            diagnostics["metric_cells_present"] += int(present.sum())
            diagnostics["metric_cells_observed"] += int(observed.sum())
            diagnostics["metric_cells_withheld_below_baseline_floor"] += int(below_floor.sum())
            diagnostics["metric_cells_withheld_degenerate_baseline"] += int(
                (withheld & ~below_floor).sum()
            )
            diagnostics["metric_cells_clipped"] += int(
                (raw_z.abs() > ANCHOR_METRIC_Z_CLIP).sum()
            )
            if observed.any():
                low = float(z[observed].min())
                high = float(z[observed].max())
                observed_low = low if observed_low is None else min(observed_low, low)
                observed_high = high if observed_high is None else max(observed_high, high)
        diagnostics["normalized_metric_min"] = observed_low
        diagnostics["normalized_metric_max"] = observed_high
        return composite_source.loc[scoped.index], normalization, diagnostics, scoped


def _performance_anchor(
    metrics: pd.DataFrame,
    names: list[str],
    game_ids: set[str],
    cfg: GlobalPlayerBTConfig,
    *,
    baseline_cache: PrefixBaselineCache | None = None,
    workspace: GlobalPlayerFitWorkspace | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Zero-mean ridge anchor in logit units, one entry per fitted player.

    Returns the anchor, a boolean mask of players that actually received one,
    and the release evidence for the anchor itself.
    """

    anchor = np.zeros(len(names), dtype=float)
    anchored = np.zeros(len(names), dtype=bool)
    evidence: dict[str, Any] = {
        "enabled": bool(cfg.performance_anchor_enabled),
        "scale_logit": float(cfg.performance_anchor_scale),
        "elo_per_contribution_sd": float(LOGIT_TO_ELO * cfg.performance_anchor_scale),
        "metric_weights": dict(PERFORMANCE_ANCHOR_METRIC_WEIGHTS),
        "weights_status": PERFORMANCE_ANCHOR_WEIGHTS_STATUS,
        "normalization": None,
        "player_map_rows_used": 0,
        "players_anchored": 0,
        "players_without_metrics": len(names),
        "anchor_mean_logit": 0.0,
        "anchor_sd_logit": 0.0,
        "baseline_min_prior_observations": int(ANCHOR_MIN_BASELINE_OBS),
        "normalized_metric_clip": float(ANCHOR_METRIC_Z_CLIP),
        "metric_cells_present": 0,
        "metric_cells_observed": 0,
        "metric_cells_withheld_below_baseline_floor": 0,
        "metric_cells_withheld_degenerate_baseline": 0,
        "metric_cells_clipped": 0,
        "normalized_metric_min": None,
        "normalized_metric_max": None,
    }
    if not cfg.performance_anchor_enabled or metrics is None or metrics.empty or not names:
        return anchor, anchored, evidence

    wanted = set(names)
    scoped = metrics[
        metrics["_game_id"].astype(str).isin(game_ids)
        & metrics["_player"].astype(str).isin(wanted)
    ]
    if scoped.empty:
        return anchor, anchored, evidence

    workspace_inputs = (
        workspace.anchor_inputs(names, game_ids) if workspace is not None else None
    )
    if workspace_inputs is not None:
        composite, normalization, diagnostics, scoped = workspace_inputs
    else:
        composite, normalization, diagnostics = _role_normalized_composite(
            scoped,
            baseline_cache=baseline_cache,
        )
    evidence["normalization"] = normalization
    evidence.update(diagnostics)
    evidence["player_map_rows_used"] = int(composite.notna().sum())
    # Median, not mean: clipping bounds a single row to +/-ANCHOR_METRIC_Z_CLIP,
    # but the mean of a low-sample player is still dominated by one extreme row,
    # so a malformed feed value could move that player the full anchor range.
    # The median makes a lone outlier unable to carry the player's composite.
    grouped = composite.groupby(scoped["_player"].astype(str))
    per_player = grouped.median()
    # A player must also have enough of their OWN maps before the anchor speaks
    # for them. Below the floor the player stays exactly neutral and is counted.
    per_player_maps = grouped.count()
    thin = per_player_maps < ANCHOR_MIN_PLAYER_MAPS
    evidence["players_withheld_below_player_map_floor"] = int(thin.sum())
    per_player = per_player.where(~thin)
    aligned = per_player.reindex(names).to_numpy(dtype=float)

    observed = np.isfinite(aligned)
    # Fewer than two anchored players leaves the spread undefined, so the whole
    # anchor stays at zero rather than inventing a scale.
    if int(observed.sum()) < 2:
        return anchor, anchored, evidence
    sample = aligned[observed]
    spread = float(np.std(sample, ddof=0))
    if not np.isfinite(spread) or spread <= 0.0:
        return anchor, anchored, evidence

    standardized = (sample - float(sample.mean())) / spread
    # Two centering passes so the residual float drift of the mean lands at
    # machine zero: the global rating scale must not move at all.
    standardized = standardized - float(standardized.mean())
    standardized = standardized - float(standardized.mean())

    anchor[observed] = cfg.performance_anchor_scale * standardized
    anchored = observed
    drift = float(anchor.sum())
    if abs(drift) > _ANCHOR_ZERO_MEAN_TOLERANCE * max(len(names), 1):
        raise GlobalPlayerRatingError(
            f"performance anchor is not zero-mean: total drift {drift:.3e}"
        )
    evidence["players_anchored"] = int(observed.sum())
    evidence["players_without_metrics"] = int(len(names) - observed.sum())
    evidence["anchor_mean_logit"] = float(anchor.mean())
    evidence["anchor_sd_logit"] = float(np.std(anchor[observed], ddof=0))
    return anchor, anchored, evidence


def _fit(
    design: csr_matrix,
    outcome: np.ndarray,
    cfg: GlobalPlayerBTConfig,
    *,
    anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    side = csr_matrix(np.ones((design.shape[0], 1), dtype=float))
    matrix = hstack([design, side], format="csr")
    penalty = np.concatenate(
        [
            np.full(design.shape[1], cfg.l2, dtype=float),
            np.asarray([cfg.side_l2], dtype=float),
        ]
    )
    # The side term is always anchored at zero; a zero player anchor reproduces
    # the plain shrink-to-zero ridge exactly.
    if anchor is None:
        player_anchor = np.zeros(design.shape[1], dtype=float)
    else:
        player_anchor = np.asarray(anchor, dtype=float).reshape(-1)
        if player_anchor.shape[0] != design.shape[1]:
            raise GlobalPlayerRatingError(
                f"anchor has {player_anchor.shape[0]} entries for {design.shape[1]} players"
            )
        if not np.isfinite(player_anchor).all():
            raise GlobalPlayerRatingError("anchor contains non-finite entries")
    anchor_vector = np.concatenate([player_anchor, np.zeros(1, dtype=float)])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = np.asarray(matrix @ parameters).reshape(-1)
        residual = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35))) - outcome
        loss = float(np.logaddexp(0.0, logits).sum() - np.dot(outcome, logits))
        delta = parameters - anchor_vector
        loss += 0.5 * float(np.dot(penalty, delta**2))
        gradient = np.asarray(matrix.T @ residual).reshape(-1) + penalty * delta
        return loss, gradient

    if threadpool_limits is None:
        raise GlobalPlayerRatingError(
            "deterministic global player fit requires the threadpoolctl dependency"
        )
    with threadpool_limits(limits=1):
        fitted = minimize(
            objective,
            np.zeros(matrix.shape[1], dtype=float),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": cfg.max_iterations, "ftol": 1e-10, "gtol": 1e-6},
        )
    if not fitted.success:
        raise GlobalPlayerRatingError(f"global player fit failed: {fitted.message}")
    return fitted.x[:-1], float(fitted.x[-1])


def _log_loss(outcome: np.ndarray, logits: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - outcome * logits))


def _frame_digest(frame: pd.DataFrame) -> str:
    """Hash frame values and schema without depending on row-index labels."""

    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode("utf-8"))
    digest.update(
        json.dumps(
            [(str(column), str(frame[column].dtype)) for column in frame.columns],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for column in frame.columns:
        try:
            hashed = pd.util.hash_pandas_object(frame[column], index=False)
            digest.update(np.asarray(hashed, dtype=np.uint64).tobytes())
        except (TypeError, ValueError):
            digest.update(
                "\x1e".join(repr(value) for value in frame[column].tolist()).encode("utf-8")
            )
    return digest.hexdigest()


def _workspace_source_digest(frame: pd.DataFrame | None) -> str:
    """Hash the source columns after normalizing the common date view.

    The weekly builder copies the map frame and converts ``date`` to a
    timezone-naive datetime before applying its cutoff. The rating builder
    receives the same rows with the source date column often still as strings.
    Normalize that representation so an object-local workspace can move
    between the two builders while still detecting row, value, and schema
    drift.
    """

    if frame is None:
        return _frame_digest(pd.DataFrame())
    normalized = frame.reset_index(drop=True).copy()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(
            normalized["date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    return _frame_digest(normalized)


def _global_fit_schema_fingerprint() -> str:
    """Fingerprint the implementation that can change a fitted snapshot."""

    implementation = (
        _model_rows,
        _design,
        _contribution_metrics,
        _role_normalized_composite,
        GlobalPlayerFitWorkspace,
        _performance_anchor,
        _fit,
        fit_global_player_bt,
    )
    source_parts: list[str] = []
    for function in implementation:
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = repr(function)
        source_parts.append(source)
    return "global-player-fit:v2:" + hashlib.sha256(
        "\n".join(source_parts).encode("utf-8")
    ).hexdigest()


def _global_fit_cache_key(
    frame: pd.DataFrame,
    lineups: dict[str, dict[str, list[tuple[str, str]]]],
    players: pd.DataFrame,
    cfg: GlobalPlayerBTConfig,
) -> str:
    """Build a key from the exact cutoff census and fit contract.

    The caller has already applied ``through`` in ``frame``. That filtered
    frame, its lineups, and its scoped player rows are the complete fit input,
    so the timestamp itself is redundant. This permits a later equivalent
    cutoff between the same map blocks to reuse the exact fit.
    """

    player_ids = _canonical_game_ids(players)
    frame_ids = set(frame["game_id"].astype(str))
    if len(player_ids) == len(players):
        scoped_players = players.loc[player_ids.astype(str).isin(frame_ids)].reset_index(drop=True)
    else:
        scoped_players = players.reset_index(drop=True)
    lineup_rows: list[object] = []
    for game_id in frame["game_id"].astype(str):
        lineup_rows.append(
            [
                game_id,
                [list(value) for value in lineups[game_id]["Blue"]],
                [list(value) for value in lineups[game_id]["Red"]],
            ]
        )
    contract = {
        "schema": _global_fit_schema_fingerprint(),
        "config": asdict(cfg),
        "frame": _frame_digest(frame),
        "players": _frame_digest(scoped_players),
        "lineups": lineup_rows,
    }
    return hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _component_summary(
    frame: pd.DataFrame,
    lineups: dict[str, dict[str, list[tuple[str, str]]]],
) -> tuple[dict[str, str], dict[str, int], str, float]:
    components = _Components()
    for game_id in frame["game_id"].astype(str):
        names = [player for side in ("Blue", "Red") for player, _ in lineups[game_id][side]]
        for player in names:
            components.find(player)
        for player in names[1:]:
            components.union(names[0], player)
    roots = {player: components.find(player) for player in components.parent}
    sizes: dict[str, int] = {}
    for root in roots.values():
        sizes[root] = sizes.get(root, 0) + 1
    largest = max(sizes, key=sizes.get)
    return roots, sizes, largest, sizes[largest] / max(len(roots), 1)


def fit_global_player_bt(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: GlobalPlayerBTConfig | None = None,
    *,
    through: pd.Timestamp | None = None,
    validate: bool = True,
    baseline_cache: PrefixBaselineCache | None = None,
    fit_cache: GlobalPlayerFitCache | None = None,
    fit_cache_slot: str | None = None,
    workspace: GlobalPlayerFitWorkspace | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one player-results scale and return its release evidence."""

    cfg = cfg or GlobalPlayerBTConfig()
    workspace_used = bool(
        workspace is not None and workspace.matches_source(maps, players)
    )
    if workspace_used:
        assert workspace is not None
        frame = workspace.frame_for(through)
        lineups = workspace.lineups
    else:
        frame, lineups = _model_rows(maps, players, through=through)
    if len(frame) < cfg.minimum_maps:
        raise GlobalPlayerRatingError(
            f"global player fit has {len(frame)} complete maps; {cfg.minimum_maps} required"
        )
    cache_key = None
    if fit_cache is not None:
        cache_key = _global_fit_cache_key(frame, lineups, players, cfg)
        cached = fit_cache.lookup(
            cache_key,
            require_validated=validate,
            slot=fit_cache_slot,
        )
        if cached is not None:
            return cached
    names = sorted(
        {
            player
            for game_id in frame["game_id"].astype(str)
            for side in ("Blue", "Red")
            for player, _ in lineups[game_id][side]
        },
        key=lambda value: (value.casefold(), value),
    )
    design = _design(frame, lineups, names)
    outcome = frame["result"].to_numpy(dtype=float)
    game_ids = frame["game_id"].astype(str)
    metrics = (
        workspace.metrics
        if workspace_used and workspace is not None
        else _contribution_metrics(players, _map_dates(frame))
        if cfg.performance_anchor_enabled
        else pd.DataFrame()
    )
    anchor, anchored, anchor_evidence = _performance_anchor(
        metrics,
        names,
        set(game_ids),
        cfg,
        baseline_cache=baseline_cache,
        workspace=workspace if workspace_used else None,
    )
    # FAIL CLOSED.  An anchor that reaches zero players is not a neutral
    # anchor, it is a silently inert one: the published ladder would go back to
    # handing byte-identical ratings to every player who never appears apart
    # from a teammate.  This is exactly what happened when the release
    # projection at lol_kills/export/public_pack.py:1546 dropped the
    # contribution columns, so a release-grade fit must refuse to publish.
    if validate and cfg.performance_anchor_enabled and not anchor_evidence["players_anchored"]:
        raise GlobalPlayerRatingError(
            "performance anchor is enabled but anchored 0 of "
            f"{len(names)} players: contribution statistics are absent from the "
            "rating input, so the published ladder would keep every teammate "
            "tie. Check that the caller's column projection carries "
            + ", ".join(PERFORMANCE_ANCHOR_SOURCE_COLUMNS)
        )
    roots, component_sizes, largest, connected_share = _component_summary(frame, lineups)
    if connected_share < cfg.minimum_connected_share:
        raise GlobalPlayerRatingError(
            f"largest player component covers {connected_share:.1%}; "
            f"{cfg.minimum_connected_share:.1%} required"
        )

    holdout: dict[str, float | int | None] = {
        "train_maps": None,
        "test_maps": None,
        "model_log_loss": None,
        "side_only_log_loss": None,
        "gain": None,
    }
    if validate:
        split = min(max(int(len(frame) * (1.0 - cfg.holdout_fraction)), 1), len(frame) - 1)
        train_x = design[:split]
        test_x = design[split:]
        train_y = outcome[:split]
        test_y = outcome[split:]
        # The holdout anchor sees train maps only. Contribution metrics are
        # measured on the same maps as the outcome, so a full-census anchor
        # would leak test-window performance into the gate.
        train_anchor, _train_anchored, _train_evidence = _performance_anchor(
            metrics,
            names,
            set(game_ids.iloc[:split]),
            cfg,
            baseline_cache=baseline_cache,
            workspace=workspace if workspace_used else None,
        )
        train_coefficients, train_side = _fit(train_x, train_y, cfg, anchor=train_anchor)
        model_loss = _log_loss(test_y, np.asarray(test_x @ train_coefficients).reshape(-1) + train_side)
        blue_rate = min(max(float(train_y.mean()), 1e-6), 1.0 - 1e-6)
        side_only = math.log(blue_rate / (1.0 - blue_rate))
        baseline_loss = _log_loss(test_y, np.full(len(test_y), side_only))
        gain = baseline_loss - model_loss
        holdout = {
            "train_maps": int(split),
            "test_maps": int(len(frame) - split),
            "model_log_loss": model_loss,
            "side_only_log_loss": baseline_loss,
            "gain": gain,
        }
        if gain < cfg.minimum_holdout_gain:
            raise GlobalPlayerRatingError(
                f"holdout log-loss gain is {gain:.6f}; {cfg.minimum_holdout_gain:.6f} required"
            )

    coefficients, side_advantage = _fit(design, outcome, cfg, anchor=anchor)
    appearances: dict[str, int] = {name: 0 for name in names}
    for game_id in frame["game_id"].astype(str):
        for side in ("Blue", "Red"):
            for player, _ in lineups[game_id][side]:
                appearances[player] += 1
    rows = []
    for position, (name, coefficient) in enumerate(zip(names, coefficients)):
        root = roots[name]
        row = {
            "player": name,
            "global_rating": cfg.prior_rating + LOGIT_TO_ELO * float(coefficient),
            "global_logit": float(coefficient),
            "global_connected": int(root == largest),
            "global_component_id": str(root),
            "global_component_size": int(component_sizes[root]),
            "global_model_maps": int(appearances[name]),
        }
        if cfg.performance_anchor_enabled:
            row["global_performance_anchor_logit"] = float(anchor[position])
            row["global_performance_anchored"] = int(bool(anchored[position]))
        rows.append(row)
    snapshot = pd.DataFrame(rows).sort_values(
        ["global_connected", "global_rating", "player"],
        ascending=[False, False, True],
        kind="stable",
    )
    meta: dict[str, Any] = {
        "model": "regularized_global_player_bt",
        "claim": "One descriptive results scale across all accepted competition tiers.",
        "n_maps": int(len(frame)),
        "n_players": int(len(names)),
        "n_components": int(len(component_sizes)),
        "largest_component_players": int(component_sizes[largest]),
        "connected_share": float(connected_share),
        "side_advantage_logit": float(side_advantage),
        "config": asdict(cfg),
        "holdout": holdout,
        "tier_adjustments": False,
        "player_statistics_used": bool(anchor_evidence["players_anchored"] > 0),
        "performance_anchor": anchor_evidence,
    }
    snapshot = snapshot.reset_index(drop=True)
    if fit_cache is not None and cache_key is not None:
        fit_cache.store(
            cache_key,
            snapshot,
            meta,
            validated=validate,
            slot=fit_cache_slot,
        )
    return snapshot, meta
