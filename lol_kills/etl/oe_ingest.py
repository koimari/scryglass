"""Download / parse Oracle's Elixir annual CSVs into team-game + player-game parquet.

Keeps the **full** OE schema on both team and player rows (no column allowlist).
Downstream models can select what they need; missing fields across years are NaN."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.paths import (
    OE_DRIVE_IDS,
    OE_FOLDER,
    OE_RECEIPT_DIR,
    PARQUET_DIR,
    RAW_OE_DIR,
    ROOT,
)
from lol_kills.net import require_https_url
from lol_kills.etl.riot_patch_receipts import RiotPatchReceiptError, validate_patch_receipt
from lol_kills.v2.patch_identity import (
    PatchIdentityError,
    corrected_oe_patch_token,
    public_patch,
)


OE_MIN_DOWNLOAD_BYTES = 10_000
OE_REQUIRED_COLUMNS = frozenset(
    {"gameid", "league", "date", "side", "position", "teamname", "kills"}
)


class OeDownloadError(RuntimeError):
    """A requested OE source refresh could not be safely committed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise OeDownloadError(f"duplicate JSON key in OE receipt: {key}")
        value[key] = item
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_oe_csv(path: Path, year: str) -> dict[str, Any]:
    """Validate source identity and temporal coverage before replacement."""

    if not path.is_file():
        raise OeDownloadError(f"OE candidate is not a regular file: {path}")
    size = path.stat().st_size
    if size < OE_MIN_DOWNLOAD_BYTES:
        raise OeDownloadError(
            f"OE candidate for {year} is too small ({size} bytes)"
        )
    head = path.read_bytes()[:512].lower()
    if b"<!doctype html" in head or b"<html" in head:
        raise OeDownloadError(f"OE candidate for {year} is HTML, not CSV")

    try:
        header = [str(column).strip() for column in pd.read_csv(path, nrows=0).columns]
    except Exception as exc:  # noqa: BLE001
        raise OeDownloadError(f"OE candidate header could not be parsed: {exc}") from exc
    missing = sorted(OE_REQUIRED_COLUMNS.difference(header))
    if missing:
        raise OeDownloadError(
            f"OE candidate for {year} is missing required columns: {missing}"
        )

    selected = [column for column in header if column in OE_REQUIRED_COLUMNS]
    if "patch" in header and "patch" not in selected:
        selected.append("patch")
    try:
        frame = pd.read_csv(
            path,
            usecols=selected,
            low_memory=False,
            dtype={"date": str, "patch": str},
        )
    except Exception as exc:  # noqa: BLE001
        raise OeDownloadError(f"OE candidate body could not be parsed: {exc}") from exc
    if frame.empty:
        raise OeDownloadError(f"OE candidate for {year} has no rows")

    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise OeDownloadError(f"OE candidate for {year} has unparseable dates")
    observed_years = sorted({str(value) for value in dates.dt.year.unique()})
    if observed_years != [year]:
        raise OeDownloadError(
            f"OE candidate year mismatch: expected {year}, observed {observed_years}"
        )

    positions = frame["position"].astype(str).str.strip().str.lower()
    team_rows = int(positions.eq("team").sum())
    player_rows = int((~positions.eq("team")).sum())
    game_count = int(frame["gameid"].nunique(dropna=True))
    if team_rows == 0 or player_rows == 0 or game_count == 0:
        raise OeDownloadError(
            f"OE candidate for {year} lacks team, player, or game rows"
        )

    patch_token_counts: dict[str, int] = {}
    patch_game_counts: dict[str, int] = {}
    public_patch_by_source: dict[str, str] = {}
    if "patch" in frame.columns:
        patch = frame["patch"].astype("string").str.strip()
        patch = patch.mask(
            patch.isna()
            | patch.str.casefold().isin({"", "nan", "nat", "none", "<na>"})
        )
        present = frame.loc[patch.notna(), ["gameid"]].copy()
        present["patch"] = patch.loc[patch.notna()]
        patch_token_counts = {
            str(token): int(count)
            for token, count in present["patch"].value_counts(sort=True).items()
        }
        patch_game_counts = {
            str(token): int(count)
            for token, count in present.groupby("patch", sort=True)["gameid"].nunique().items()
        }
        for token in sorted(patch_token_counts):
            try:
                public_patch_by_source[token] = public_patch(token)
            except PatchIdentityError:
                continue

    return {
        "raw_sha256": _sha256_file(path),
        "bytes": int(size),
        "row_count": int(len(frame)),
        "game_count": game_count,
        "team_row_count": team_rows,
        "player_row_count": player_rows,
        "league_count": int(frame["league"].nunique(dropna=True)),
        "date_min_utc": dates.min().isoformat(),
        "date_max_utc": dates.max().isoformat(),
        "columns_sha256": _canonical_sha256(header),
        "patch_token_counts": patch_token_counts,
        "patch_game_counts": patch_game_counts,
        "public_patch_by_source": public_patch_by_source,
    }


def _archive_existing(path: Path, raw_sha256: str) -> Path:
    archive_dir = RAW_OE_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{path.stem}.{raw_sha256}{path.suffix}"
    if archive.exists():
        if not archive.is_file() or _sha256_file(archive) != raw_sha256:
            raise OeDownloadError(f"OE archive path conflict: {archive}")
        return archive
    try:
        os.link(path, archive)
    except OSError:
        shutil.copy2(path, archive)
    if _sha256_file(archive) != raw_sha256:
        raise OeDownloadError(f"OE archive verification failed: {archive}")
    return archive


def _write_refresh_receipt(receipt: dict[str, Any]) -> Path:
    unsigned = dict(receipt)
    unsigned["receipt_canonical_sha256"] = _canonical_sha256(unsigned)
    payload = (json.dumps(unsigned, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    retrieved = datetime.fromisoformat(str(receipt["retrieved_at_utc"]).replace("Z", "+00:00"))
    stamp = retrieved.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = str(receipt["candidate"]["raw_sha256"])
    destination = OE_RECEIPT_DIR / f"oe-{receipt['year']}-{stamp}-{digest[:16]}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            if destination.read_bytes() != payload:
                raise OeDownloadError(f"OE receipt path conflict: {destination}")
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_refresh_receipt(path: Path) -> dict[str, Any]:
    """Load and verify one hash-bound OE transport receipt."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OeDownloadError(f"OE refresh receipt could not be parsed: {path}") from exc
    if not isinstance(payload, dict):
        raise OeDownloadError("OE refresh receipt root must be an object")
    if payload.get("schema") != "scryglass:oe-source-refresh:v1":
        raise OeDownloadError("OE refresh receipt schema is not supported")
    claimed = payload.get("receipt_canonical_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_canonical_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise OeDownloadError("OE refresh receipt canonical hash mismatch")
    candidate = payload.get("candidate")
    if (
        not isinstance(candidate, dict)
        or not isinstance(candidate.get("raw_sha256"), str)
        or len(candidate["raw_sha256"]) != 64
    ):
        raise OeDownloadError("OE refresh receipt candidate binding is invalid")
    authority = payload.get("authority")
    if not isinstance(authority, dict) or any(
        authority.get(name) is not False
        for name in (
            "model_validation_authority",
            "probability_authority",
            "recommendation_authority",
            "betting_authority",
        )
    ):
        raise OeDownloadError("OE refresh receipt exceeds its authority ceiling")
    return payload


def validate_accepted_source_receipt(
    receipt_path: Path,
    csv_path: Path,
    year: str | int,
) -> dict[str, Any]:
    """Verify that one transport receipt authorizes the exact annual CSV bytes."""

    receipt = load_refresh_receipt(receipt_path.expanduser().resolve())
    expected_year = str(year)
    candidate = receipt.get("candidate")
    if receipt.get("year") != expected_year or not isinstance(candidate, dict):
        raise OeDownloadError("OE source receipt year does not match the requested import")
    source = csv_path.expanduser().resolve()
    validated = _validate_oe_csv(source, expected_year)
    if candidate.get("raw_sha256") != validated["raw_sha256"]:
        raise OeDownloadError("OE source receipt does not bind the requested CSV bytes")
    for field in (
        "bytes",
        "row_count",
        "game_count",
        "columns_sha256",
        "date_max_utc",
        "patch_token_counts",
        "patch_game_counts",
        "public_patch_by_source",
    ):
        if field not in candidate:
            continue
        if candidate.get(field) != validated[field]:
            raise OeDownloadError(f"OE source receipt field does not match the CSV: {field}")
    return {
        "receipt_path": str(receipt_path.expanduser().resolve()),
        "receipt_canonical_sha256": receipt["receipt_canonical_sha256"],
        "year": expected_year,
        "status": receipt.get("status"),
        **validated,
    }


def oe_csv_path(year: str | int) -> Path:
    y = str(year)
    return RAW_OE_DIR / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"


def list_local_oe_csvs() -> list[Path]:
    RAW_OE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))


def _remote_state_path() -> Path:
    return OE_RECEIPT_DIR / "remote-state.json"


def _load_remote_state() -> dict[str, dict[str, Any]]:
    path = _remote_state_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(year): value
        for year, value in payload.items()
        if isinstance(value, dict)
    }


def _write_remote_state(state: dict[str, dict[str, Any]]) -> None:
    path = _remote_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remote_file_signature(url: str) -> dict[str, Any]:
    """Read public Drive metadata without downloading the annual CSV."""

    url = require_https_url(url, hosts={"drive.google.com"})
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/csv",
            "Range": "bytes=0-0",
            "User-Agent": "scryglass/oe-csv-refresh/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            headers = response.headers
            content_range = str(headers.get("Content-Range") or "")
            match = re.search(r"bytes\s+\d+-\d+/(\d+)", content_range)
            total_bytes = int(match.group(1)) if match else int(headers.get("Content-Length") or 0)
            if total_bytes <= 0:
                raise OeDownloadError(f"OE remote metadata has no file size: {url}")
            return {
                "bytes": total_bytes,
                "last_modified": str(headers.get("Last-Modified") or "").strip(),
            }
    except urllib.error.HTTPError as exc:
        raise OeDownloadError(f"OE remote metadata returned HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise OeDownloadError(f"OE remote metadata request failed: {url}") from exc


def _remote_signature_matches(cached: dict[str, Any], current: dict[str, Any]) -> bool:
    try:
        cached_bytes = int(cached.get("bytes", -1))
        current_bytes = int(current.get("bytes", -2))
    except (TypeError, ValueError):
        return False
    if cached_bytes != current_bytes:
        return False
    cached_modified = str(cached.get("last_modified") or "")
    current_modified = str(current.get("last_modified") or "")
    return bool(cached_modified and current_modified and cached_modified == current_modified) or (
        not cached_modified and not current_modified
    )


def load_cached_oe(
    years: Iterable[str | int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the last successful normalized OE cache, if one exists.

    CI workers do not retain the raw annual CSVs between runs.  The normalized
    parquet cache is the durable hand-off between the fast GRID refresh and a
    slower OE reconciliation.  An empty pair is returned when no cache exists.
    """
    team_path = PARQUET_DIR / "oe_team_games.parquet"
    player_path = PARQUET_DIR / "oe_player_games.parquet"
    if not team_path.exists() or not player_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    try:
        team = pd.read_parquet(team_path)
        players = pd.read_parquet(player_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[oe] cached parquet could not be read: {exc}")
        return pd.DataFrame(), pd.DataFrame()

    if years:
        wanted = {int(y) for y in years}

        def keep_years(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            year_col = next(
                (c for c in ("oe_year", "year") if c in frame.columns),
                None,
            )
            if year_col is None:
                return frame
            return frame[pd.to_numeric(frame[year_col], errors="coerce").isin(wanted)].copy()

        team = keep_years(team)
        players = keep_years(players)
    return team, players


def download_oe_years(years: Iterable[str | int], force: bool = False) -> list[Path]:
    """
    Download missing or changed annual files, or safely refresh them when ``force`` is true.

    A one-byte range request checks the public Drive file size and modification
    time before a full download. Every downloaded candidate is staged and
    structurally validated. A refresh refuses a temporal regression, archives
    the prior exact bytes, atomically replaces the annual file, and emits a
    hash-bound source receipt.
    """
    RAW_OE_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    prepared: list[dict[str, Any]] = []
    remote_state = _load_remote_state()
    downloader: Any | None = None
    for year in years:
        y = str(year)
        dest = oe_csv_path(y)
        previous: dict[str, Any] | None = None
        previous_error: str | None = None
        if dest.exists():
            try:
                previous = _validate_oe_csv(dest, y)
            except OeDownloadError as exc:
                previous_error = str(exc)
        fid = OE_DRIVE_IDS.get(y)
        if not fid:
            message = f"no known Drive id for {y}"
            if force:
                for item in prepared:
                    Path(item["temporary"]).unlink(missing_ok=True)
                raise OeDownloadError(message)
            print(f"[oe] {message}")
            if dest.exists():
                out_paths.append(dest)
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        remote_signature: dict[str, Any] | None = None
        if previous is not None and not force:
            try:
                remote_signature = _remote_file_signature(url)
            except OeDownloadError as exc:
                print(f"[oe] remote check failed for {y}: {exc}")
                print(f"     Keeping validated cache {dest.name}")
                out_paths.append(dest)
                continue
            cached_signature = remote_state.get(y)
            if isinstance(cached_signature, dict) and _remote_signature_matches(
                cached_signature, remote_signature
            ):
                remote_state[y] = {
                    **cached_signature,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                print(f"[oe] remote file unchanged; using cache {dest.name}")
                out_paths.append(dest)
                continue
        if downloader is None:
            try:
                import gdown
            except ImportError as exc:
                raise SystemExit(
                    "gdown required for OE download: pip install gdown\n"
                    f"Or manually download from {OE_FOLDER} into {RAW_OE_DIR}"
                ) from exc
            downloader = gdown
        action = "refreshing" if dest.exists() else "downloading"
        print(f"[oe] {action} {y} → staged candidate for {dest.name}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{dest.name}.", suffix=".download", dir=RAW_OE_DIR
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            downloader.download(url, str(temporary), quiet=False)
            candidate = _validate_oe_csv(temporary, y)
            if remote_signature is not None and int(candidate["bytes"]) != int(remote_signature["bytes"]):
                raise OeDownloadError(
                    f"OE remote file changed during download for {y}: "
                    f"expected {remote_signature['bytes']} bytes, got {candidate['bytes']}"
                )
            if (
                previous is not None
                and str(candidate["date_max_utc"]) < str(previous["date_max_utc"])
            ):
                raise OeDownloadError(
                    "OE candidate date_max regressed "
                    f"from {previous['date_max_utc']} to {candidate['date_max_utc']}"
                )
        except Exception as exc:  # noqa: BLE001
            temporary.unlink(missing_ok=True)
            message = f"OE download validation failed for {y}: {exc}"
            if force:
                for item in prepared:
                    Path(item["temporary"]).unlink(missing_ok=True)
                raise OeDownloadError(message) from exc
            print(f"[oe] {message}")
            print(f"     Existing bytes were not changed; place a valid file manually: {dest}")
            if dest.exists():
                out_paths.append(dest)
            continue

        prepared.append(
            {
                "year": y,
                "destination": dest,
                "temporary": temporary,
                "source_url": url,
                "drive_file_id": fid,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate": candidate,
                "previous": previous,
                "previous_validation_error": previous_error,
                "remote_signature": remote_signature,
            }
        )

    try:
        for item in prepared:
            y = str(item["year"])
            dest = Path(item["destination"])
            temporary = Path(item["temporary"])
            candidate = dict(item["candidate"])
            previous = item["previous"]
            archive: Path | None = None
            status = "downloaded"

            if previous is not None and previous["raw_sha256"] == candidate["raw_sha256"]:
                status = "unchanged"
                temporary.unlink(missing_ok=True)
            else:
                if dest.exists():
                    old_sha = _sha256_file(dest)
                    archive = _archive_existing(dest, old_sha)
                    status = "refreshed"
                os.replace(temporary, dest)
                if _sha256_file(dest) != candidate["raw_sha256"]:
                    raise OeDownloadError(f"OE committed hash verification failed for {y}")

            previous_summary: dict[str, Any] | None = None
            if dest.exists() and previous is not None:
                previous_summary = dict(previous)
                previous_summary["archive_locator"] = (
                    _display_path(archive) if archive is not None else None
                )
            elif item["previous_validation_error"] is not None:
                previous_summary = {
                    "raw_sha256": _sha256_file(archive if archive is not None else dest),
                    "validation_status": "invalid_replaced",
                    "validation_error": item["previous_validation_error"],
                    "archive_locator": _display_path(archive) if archive is not None else None,
                }

            receipt = {
                "schema": "scryglass:oe-source-refresh:v1",
                "year": y,
                "retrieved_at_utc": item["retrieved_at_utc"],
                "status": status,
                "source": {
                    "provider": "Oracle's Elixir",
                    "transport": "public_google_drive_file",
                    "locator": item["source_url"],
                    "drive_file_id": item["drive_file_id"],
                    "folder_locator": OE_FOLDER,
                    "remote_signature": item["remote_signature"],
                },
                "destination_locator": _display_path(dest),
                "candidate": candidate,
                "previous": previous_summary,
                "authority": {
                    "descriptive_source_freshness_evidence": True,
                    "model_validation_authority": False,
                    "probability_authority": False,
                    "recommendation_authority": False,
                    "betting_authority": False,
                },
                "claim_ceiling": (
                    "This receipt proves transport, structural validation, temporal coverage, "
                    "and exact source bytes only. It does not validate a model, probability, "
                    "odds, recommendation, or wager."
                ),
            }
            receipt_path = _write_refresh_receipt(receipt)
            print(
                f"[oe] {status} {dest.name}; max={candidate['date_max_utc']} "
                f"sha256={candidate['raw_sha256']} receipt={receipt_path.name}"
            )
            out_paths.append(dest)
            remote_signature = item["remote_signature"]
            if isinstance(remote_signature, dict):
                remote_state[y] = {
                    **remote_signature,
                    "checked_at_utc": item["retrieved_at_utc"],
                }
    finally:
        for item in prepared:
            Path(item["temporary"]).unlink(missing_ok=True)
        if remote_state:
            _write_remote_state(remote_state)
    return out_paths


def install_browser_download(candidate_path: Path, year: str | int) -> Path:
    """Validate and atomically install an annual CSV downloaded by Brave Origin."""

    y = str(year)
    # The transport is whatever acquisition path actually won in the launcher
    # ladder, exported as SCRYGLASS_OE_TRANSPORT. The receipt must record what
    # HAPPENED, not what this function was historically named for: after the
    # headless-first change the common path is anonymous HTTPS, and a receipt
    # asserting browser transport for those bytes would be false provenance.
    # An unset variable proves nothing, so it is recorded as exactly that.
    transport = os.environ.get("SCRYGLASS_OE_TRANSPORT") or "local_candidate_unverified_transport"
    source = candidate_path.expanduser().resolve()
    destination = oe_csv_path(y)
    RAW_OE_DIR.mkdir(parents=True, exist_ok=True)
    candidate = _validate_oe_csv(source, y)

    previous: dict[str, Any] | None = None
    previous_error: str | None = None
    if destination.exists():
        try:
            previous = _validate_oe_csv(destination, y)
        except OeDownloadError as exc:
            previous_error = str(exc)
    if (
        previous is not None
        and str(candidate["date_max_utc"]) < str(previous["date_max_utc"])
    ):
        raise OeDownloadError(
            "OE browser candidate date_max regressed "
            f"from {previous['date_max_utc']} to {candidate['date_max_utc']}"
        )

    archive: Path | None = None
    status = "downloaded"
    if previous is not None and previous["raw_sha256"] == candidate["raw_sha256"]:
        status = "unchanged"
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".browser", dir=RAW_OE_DIR
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            if _sha256_file(temporary) != candidate["raw_sha256"]:
                raise OeDownloadError(f"OE browser copy verification failed for {y}")
            if destination.exists():
                old_sha = _sha256_file(destination)
                archive = _archive_existing(destination, old_sha)
                status = "refreshed"
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    previous_summary: dict[str, Any] | None = None
    if previous is not None:
        previous_summary = {
            **previous,
            "archive_locator": _display_path(archive) if archive is not None else None,
        }
    elif previous_error is not None:
        previous_summary = {
            "validation_status": "invalid_replaced",
            "validation_error": previous_error,
            "archive_locator": _display_path(archive) if archive is not None else None,
        }

    retrieved_at = datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema": "scryglass:oe-source-refresh:v1",
        "year": y,
        "retrieved_at_utc": retrieved_at,
        "status": status,
        "source": {
            "provider": "Oracle's Elixir",
            "transport": transport,
            # The pinned id can rotate. When the headless fetcher re-resolved
            # the folder, the launcher exports the identity that ACTUALLY
            # served the bytes; hardcoding the pinned id here would attribute
            # the accepted bytes to an object that failed.
            "locator": (
                os.environ.get("SCRYGLASS_OE_LOCATOR")
                or f"https://drive.google.com/uc?export=download&id={OE_DRIVE_IDS.get(y, '')}"
            ),
            "drive_file_id": (
                os.environ.get("SCRYGLASS_OE_DRIVE_FILE_ID") or OE_DRIVE_IDS.get(y)
            ),
            "folder_locator": OE_FOLDER,
            "remote_signature": None,
        },
        "destination_locator": _display_path(destination),
        "candidate": candidate,
        "previous": previous_summary,
        "authority": {
            "descriptive_source_freshness_evidence": True,
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "This receipt proves the recorded transport, structural validation, temporal "
            "coverage, and exact source bytes only. It does not validate a model, "
            "probability, odds, recommendation, or wager."
        ),
    }
    receipt_path = _write_refresh_receipt(receipt)
    print(
        f"[oe] {status} {transport} {destination.name}; "
        f"max={candidate['date_max_utc']} sha256={candidate['raw_sha256']} "
        f"receipt={receipt_path.name}"
    )
    return destination


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve OE column names (stripped); drop accidental duplicate headers."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def _normalize_patch_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep source patch tokens nullable and string-typed across cache versions."""

    if "patch" not in frame.columns:
        return frame
    out = frame.copy()
    patch = out["patch"].astype("string").str.strip()
    out["patch"] = patch.mask(
        patch.isna()
        | patch.str.casefold().isin({"", "nan", "nat", "none", "<na>"})
    )
    return out


def _normalize_identity(
    frame: pd.DataFrame,
    *,
    players: bool,
    patch_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Team/champ aliases + dates only — do not drop or invent columns."""
    out = _normalize_patch_column(frame)
    if "patch" in out.columns:
        out["oe_patch_token"] = out["patch"]
    if "teamname" in out.columns:
        out["teamname"] = out["teamname"].map(
            lambda x: normalize_team(str(x)) if pd.notna(x) else x
        )
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "patch" in out.columns and "date" in out.columns:
        realm_patch_column = next(
            (
                column
                for column in (
                    "server_patch",
                    "game_patch",
                    "realm_patch",
                    "authoritative_patch",
                )
                if column in out.columns
            ),
            None,
        )
        realm_kind_column = next(
            (
                column
                for column in ("patch_realm", "realm_kind", "server_kind")
                if column in out.columns
            ),
            None,
        )
        corrected_tokens: list[str] = []
        corrections: list[str | None] = []
        patch_values = out[realm_patch_column].tolist() if realm_patch_column else [None] * len(out)
        realm_values = out[realm_kind_column].tolist() if realm_kind_column else [None] * len(out)
        for token, event_time, realm_patch, realm_kind in zip(
            out["patch"].tolist(),
            out["date"].tolist(),
            patch_values,
            realm_values,
        ):
            if token is None or bool(pd.isna(token)):
                corrected_tokens.append(pd.NA)
                corrections.append(None)
                continue
            corrected, rule = corrected_oe_patch_token(
                token,
                event_time,
                realm_patch=realm_patch,
                realm_kind=realm_kind,
            )
            corrected_tokens.append(corrected)
            corrections.append(rule["corrected_token"] if rule else None)
        out["patch"] = pd.Series(corrected_tokens, index=out.index, dtype="string")
        out["patch_correction"] = pd.Series(corrections, index=out.index, dtype="string")
    if patch_receipts and "gameid" in out.columns:
        receipt_patches: list[str | None] = []
        receipt_hashes: list[str | None] = []
        for game_id in out["gameid"].tolist():
            receipt = patch_receipts.get(str(game_id))
            if receipt is None:
                receipt_patches.append(None)
                receipt_hashes.append(None)
                continue
            try:
                validated = validate_patch_receipt(receipt)
            except RiotPatchReceiptError as exc:
                raise ValueError(f"invalid Riot patch receipt for game {game_id!r}") from exc
            observed = validated["observed"]
            receipt_patches.append(str(observed["client_patch"]))
            receipt_hashes.append(str(validated["receipt_canonical_sha256"]))
        receipt_patch_series = pd.Series(receipt_patches, index=out.index, dtype="string")
        out["patch_receipt_sha256"] = pd.Series(receipt_hashes, index=out.index, dtype="string")
        has_receipt = receipt_patch_series.notna()
        if has_receipt.any():
            out.loc[has_receipt, "patch"] = receipt_patch_series.loc[has_receipt]
            out.loc[has_receipt, "patch_source"] = "riot_live_feed"
    if "playoffs" in out.columns:
        numeric = pd.to_numeric(out["playoffs"], errors="coerce")
        invalid = numeric.notna() & ~numeric.isin((0, 1))
        if invalid.any():
            raise ValueError("OE playoffs contains a value other than 0, 1, or missing")
        out["playoffs"] = numeric.astype("Int64").astype("boolean")
    if players and "champion" in out.columns:
        out["champion"] = out["champion"].map(
            lambda x: normalize_champ(str(x)) if pd.notna(x) else x
        )
    return out


def parse_oe_csv(
    path: Path,
    *,
    patch_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (team_games, player_games) with the full OE schema on each."""
    print(f"[oe] parsing {path.name}")
    try:
        # Keep date and patch as source text. Patch labels such as 16.10 and
        # 16.1 are different Riot patches, so numeric inference is unsafe.
        df = _strip_columns(
            pd.read_csv(path, low_memory=False, dtype={"date": str, "patch": str})
        )
    except (ValueError, TypeError):
        df = _strip_columns(pd.read_csv(path, low_memory=False, dtype={"patch": str}))
    # position/participant: team rows have position == 'team'
    pos_col = next((c for c in df.columns if c.lower() == "position"), None)
    if pos_col is None:
        raise ValueError(f"No position column in {path}")

    team = _normalize_identity(
        df[df[pos_col].astype(str).str.lower() == "team"].copy(),
        players=False,
        patch_receipts=patch_receipts,
    )
    players = _normalize_identity(
        df[df[pos_col].astype(str).str.lower() != "team"].copy(),
        players=True,
        patch_receipts=patch_receipts,
    )

    year = _year_from_name(path.name)
    for frame in (team, players):
        frame["source"] = "oe"
        frame["oe_year"] = year
    print(
        f"[oe]   {path.name}: team={len(team)} player={len(players)} "
        f"cols={len(df.columns)}"
    )
    return team, players


def _year_from_name(name: str) -> int | None:
    m = re.match(r"(\d{4})_", name)
    return int(m.group(1)) if m else None


# How many days a local annual OE CSV may lag before the refresh is
# considered stale.  A Drive quota block leaves the previous download in
# place, so age - not absence - is the signal that the upstream feed stopped
# reaching us.
OE_SOURCE_MAX_AGE_DAYS = 3


def check_oe_source_freshness(
    paths: list[Path],
    *,
    now: datetime | None = None,
    max_age_days: int = OE_SOURCE_MAX_AGE_DAYS,
) -> None:
    """Decide how a stale local OE export should affect a refresh.

    ``paths`` are the annual CSVs the ingest is about to parse.  The freshest
    file's modification time is the cheapest available staleness signal: a
    quota-blocked download leaves the prior CSV untouched, so its mtime stops
    advancing while the pipeline keeps reporting success.

    Policy: warn loudly, do not refuse.  A hard failure here would also stop
    the ``oe_api`` completed-game bridge, and that bridge is what keeps result
    based ratings current while detail statistics are missing - refusing to
    build would turn a partial outage into a total one.  Publishing correctness
    is already enforced downstream, where absent statistics produce an explicit
    ``unavailable`` grade rather than a wrong number.

    Set ``SCRYGLASS_OE_STRICT_FRESHNESS=1`` to raise instead, which is the
    right setting for a scheduled job that should page someone.

    Caveat: modification time answers "when did we last download", not "how
    fresh is the data inside".  A quota-blocked download leaves the previous
    file untouched, which is exactly the case this catches, but a re-download
    of an upstream file that itself stopped advancing would still look fresh.
    ``_validate_oe_csv`` reports ``date_max_utc`` per file and is the stronger
    signal once the CSVs are parsed.
    """

    if not paths:
        return None

    now = now or datetime.now(timezone.utc)
    newest = max(paths, key=lambda path: path.stat().st_mtime)
    fetched_at = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc)
    age_days = (now - fetched_at).total_seconds() / 86400.0
    if age_days <= max_age_days:
        return None

    message = (
        f"stale Oracle's Elixir source: newest local CSV {newest.name} was "
        f"last written {age_days:.1f} days ago "
        f"({fetched_at.isoformat(timespec='seconds')}), over the "
        f"{max_age_days}-day limit. Detail statistics will be missing for "
        f"games after that date and grades will report unavailable. "
        f"Refresh {RAW_OE_DIR} from {OE_FOLDER} "
        f"(a Google Drive quota block leaves the previous file in place)."
    )
    if os.environ.get("SCRYGLASS_OE_STRICT_FRESHNESS") == "1":
        raise OeDownloadError(message)
    print(f"[oe] WARNING: {message}")
    return None


def ingest_oe(
    years: list[str] | None = None,
    download: bool = False,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all local OE CSVs (optionally download first).
    Writes parquet caches under warehouse/parquet/.
    """
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    if download:
        ys = years or list(OE_DRIVE_IDS.keys())[-4:]  # recent by default
        download_oe_years(ys, force=force_download)

    paths = list_local_oe_csvs()
    if years:
        want = {str(y) for y in years}
        paths = [p for p in paths if any(p.name.startswith(y) for y in want)]

    check_oe_source_freshness(paths)

    if not paths:
        cached_team, cached_players = load_cached_oe(years)
        if not cached_team.empty or not cached_players.empty:
            print(
                "[oe] no annual CSV available; preserving the last normalized "
                f"cache (team={len(cached_team)} player={len(cached_players)})"
            )
            return cached_team, cached_players
        print(
            "[oe] no CSVs found. Put annual files in "
            f"{RAW_OE_DIR} (from {OE_FOLDER}) or pass --download-oe"
        )
        empty_t = pd.DataFrame()
        empty_p = pd.DataFrame()
        empty_t.to_parquet(PARQUET_DIR / "oe_team_games.parquet", index=False)
        empty_p.to_parquet(PARQUET_DIR / "oe_player_games.parquet", index=False)
        return empty_t, empty_p

    teams, players = [], []
    source_files: list[dict[str, Any]] = []
    for p in paths:
        source_files.append(
            {
                "locator": _display_path(p),
                "year": _year_from_name(p.name),
                **_validate_oe_csv(p, str(_year_from_name(p.name))),
            }
        )
        t, pl = parse_oe_csv(p)
        teams.append(t)
        players.append(pl)

    # A partial Google Drive recovery must not erase years whose annual file
    # was still quota-blocked. Keep cached rows only for those missing years;
    # newly parsed OE rows remain authoritative for years we did download.
    cached_team, cached_players = load_cached_oe()
    parsed_years = {_year_from_name(p.name) for p in paths}
    parsed_years.discard(None)
    if parsed_years:
        for cached, target in (
            (cached_team, teams),
            (cached_players, players),
        ):
            if cached.empty:
                continue
            year_col = "oe_year" if "oe_year" in cached.columns else "year"
            if year_col not in cached.columns:
                continue
            missing = cached[
                ~pd.to_numeric(cached[year_col], errors="coerce").isin(parsed_years)
            ]
            if not missing.empty:
                target.insert(0, missing)

    # Outer-union columns across years (older dumps may lack newer fields).
    team_df = pd.concat(teams, ignore_index=True, sort=False)
    player_df = pd.concat(players, ignore_index=True, sort=False)

    # Dedup by gameid+side (later years may overlap)
    if not team_df.empty and "gameid" in team_df.columns:
        team_df = team_df.sort_values("date").drop_duplicates(["gameid", "side"], keep="last")
    if not player_df.empty and "gameid" in player_df.columns:
        player_df = player_df.sort_values("date").drop_duplicates(
            ["gameid", "side", "position"], keep="last"
        )

    team_path = PARQUET_DIR / "oe_team_games.parquet"
    player_path = PARQUET_DIR / "oe_player_games.parquet"
    team_df.to_parquet(team_path, index=False)
    player_df.to_parquet(player_path, index=False)
    meta = {
        "schema_version": "scryglass:oe-normalized-cache:v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_team_rows": int(len(team_df)),
        "n_player_rows": int(len(player_df)),
        "n_team_cols": int(len(team_df.columns)),
        "n_player_cols": int(len(player_df.columns)),
        "n_games": int(team_df["gameid"].nunique()) if len(team_df) and "gameid" in team_df else 0,
        "files": [p.name for p in paths],
        "source_files": source_files,
        "schema": "full_oe",
        "team_columns": list(team_df.columns),
        "player_columns": list(player_df.columns),
    }
    (PARQUET_DIR / "oe_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[oe] wrote {team_path.name} rows={len(team_df)} cols={meta['n_team_cols']} "
        f"games={meta['n_games']}"
    )
    print(
        f"[oe] wrote {player_path.name} rows={len(player_df)} cols={meta['n_player_cols']}"
    )
    return team_df, player_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-browser-candidate", type=Path)
    parser.add_argument("--year")
    parser.add_argument("--receipt-output", type=Path)
    arguments = parser.parse_args()
    if arguments.install_browser_candidate is not None:
        if not arguments.year:
            parser.error("--year is required with --install-browser-candidate")
        installed = install_browser_download(arguments.install_browser_candidate, arguments.year)
        if arguments.receipt_output is not None:
            source = _validate_oe_csv(installed, str(arguments.year))
            matches = sorted(
                OE_RECEIPT_DIR.glob(
                    f"oe-{arguments.year}-*-{source['raw_sha256'][:16]}.json"
                ),
                key=lambda path: path.stat().st_mtime_ns,
            )
            if not matches:
                raise OeDownloadError("installed browser source has no transport receipt")
            validate_accepted_source_receipt(matches[-1], installed, arguments.year)
            arguments.receipt_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matches[-1], arguments.receipt_output)
    else:
        ingest_oe(download=False)
