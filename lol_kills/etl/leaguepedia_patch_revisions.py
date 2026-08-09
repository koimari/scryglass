"""Recover pre-event patch authority from Leaguepedia Data-page revisions.

The current ScoreboardGames patch field is useful for reconciliation, but its
capture is retrospective.  Leaguepedia's MatchSchedule rows identify the
backing ``Data:`` page and the match's tab/ordinal.  This module captures that
schedule metadata, selects the latest revision of the Data page strictly
before each fixture cutoff, and extracts only the patch directive applicable
to that match.  Result fields are never requested or emitted in receipts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "scryglass:leaguepedia-patch-revisions:v1"
DEFAULT_RUN = Path("data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31")
DEFAULT_OUTPUT = DEFAULT_RUN / "leaguepedia-patch-revisions-v1"
DEFAULT_RETROSPECTIVE_MANIFEST = DEFAULT_RUN / "leaguepedia-patch-receipts-v1" / "receipt-manifest.json"
API_ROOT = "https://lol.fandom.com/api.php"
CARGO_ROOT = "https://lol.fandom.com/wiki/Special:CargoExport"
USER_AGENT = "Scryglass-historical-patch-recovery/1.0"
SCHEDULE_FIELDS = (
    "MatchSchedule.MatchId",
    "MatchSchedule.OverviewPage",
    "MatchSchedule.Patch",
    "MatchSchedule.Tab",
    "MatchSchedule.N_MatchInTab",
    "MatchSchedule.N_Page",
    "MatchSchedule.N_MatchInPage",
    "MatchSchedule.Team1",
    "MatchSchedule.Team2",
    "MatchSchedule.DateTime_UTC",
    "_pageName",
)
OUTCOME_FIELDS = frozenset(
    {
        "Winner",
        "Team1Score",
        "Team2Score",
        "winner_team_id",
        "complete",
        "won",
        "WinTeam",
        "LossTeam",
    }
)
SET_PATCH_RE = re.compile(
    r"\{\{\s*SetPatch\s*\|(?P<body>[^}]*)\}\}", re.IGNORECASE | re.DOTALL
)
SCHEDULE_START_RE = re.compile(
    r"\{\{\s*MatchSchedule/Start\s*\|(?P<body>[^}]*)\}\}", re.IGNORECASE | re.DOTALL
)
MATCH_RE = re.compile(r"\{\{\s*MatchSchedule\s*\|", re.IGNORECASE)


class PatchRevisionError(ValueError):
    """Raised when historical patch evidence is malformed."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise PatchRevisionError("timestamp is empty")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rfc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_object(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _write_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _fetch(url: str, *, timeout: float) -> tuple[bytes, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    try:
        return raw, json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PatchRevisionError(f"invalid JSON from {url}") from exc


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, Mapping) and isinstance(row.get("pregame"), Mapping) for row in rows):
        raise PatchRevisionError("frozen ledger contains an invalid pregame row")
    return [dict(row) for row in rows]


def _batches(values: list[str], size: int = 35) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _cargo_url(match_ids: list[str]) -> str:
    where = "(" + " OR ".join(
        f'MatchSchedule.MatchId="{match_id.replace(chr(34), chr(92) + chr(34))}"'
        for match_id in match_ids
    ) + ")"
    params = {
        "tables": "MatchSchedule",
        "fields": ",".join(SCHEDULE_FIELDS),
        "where": where,
        "order_by": "MatchSchedule.DateTime_UTC ASC",
        "limit": "500",
        "format": "json",
    }
    return CARGO_ROOT + "?" + urllib.parse.urlencode(params)


def _api_url(params: Mapping[str, Any]) -> str:
    return API_ROOT + "?" + urllib.parse.urlencode(params)


def _history_params(page: str, *, newest: datetime, oldest: datetime) -> dict[str, str]:
    return {
        "action": "query",
        "prop": "revisions",
        "titles": page,
        "rvprop": "ids|timestamp",
        "rvlimit": "max",
        "rvstart": _rfc(newest),
        "rvend": _rfc(oldest),
        "rvdir": "older",
        "format": "json",
        "formatversion": "2",
    }


def _revision_content_params(page: str, revision_id: int) -> dict[str, str]:
    return {
        "action": "query",
        "prop": "revisions",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "revids": str(revision_id),
        "format": "json",
        "formatversion": "2",
    }


def _page_revisions(payload: Any) -> tuple[str | None, list[dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        return None, []
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list) or not pages or not isinstance(pages[0], Mapping):
        return None, []
    page = pages[0]
    title = str(page.get("title")) if page.get("title") else None
    raw_revisions = page.get("revisions", [])
    if not isinstance(raw_revisions, list):
        return title, []
    revisions: list[dict[str, Any]] = []
    for raw in raw_revisions:
        if not isinstance(raw, Mapping) or raw.get("revid") is None or raw.get("timestamp") is None:
            continue
        revisions.append(
            {
                "revision_id": int(raw["revid"]),
                "revision_timestamp": str(raw["timestamp"]),
            }
        )
    return title, revisions


def _content_from_payload(payload: Any) -> tuple[int, str, str] | None:
    if not isinstance(payload, Mapping):
        return None
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list) or not pages or not isinstance(pages[0], Mapping):
        return None
    revisions = pages[0].get("revisions", [])
    if not isinstance(revisions, list) or not revisions or not isinstance(revisions[0], Mapping):
        return None
    revision = revisions[0]
    slots = revision.get("slots", {})
    main = slots.get("main", {}) if isinstance(slots, Mapping) else {}
    content = main.get("content") if isinstance(main, Mapping) else None
    if not isinstance(content, str):
        content = revision.get("content")
    if not isinstance(content, str) or revision.get("revid") is None or revision.get("timestamp") is None:
        return None
    return int(revision["revid"]), str(revision["timestamp"]), content


def _patch_label(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return None


def _arg(body: str, name: str) -> str | None:
    match = re.search(rf"(?:^|\|)\s*{re.escape(name)}\s*=\s*([^|\n}}]+)", body, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _norm_tab(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def extract_match_patch(
    content: str,
    *,
    tab: str | None,
    match_ordinal: int | None,
) -> dict[str, Any]:
    """Extract the SetPatch that governs one MatchSchedule block."""

    patches = [
        {
            "patch": _patch_label(_arg(match.group("body"), "patch")),
            "offset": match.start(),
        }
        for match in SET_PATCH_RE.finditer(content)
    ]
    patches = [row for row in patches if row["patch"]]
    starts = list(SCHEDULE_START_RE.finditer(content))
    matches = list(MATCH_RE.finditer(content))
    target_tab = _norm_tab(tab)
    try:
        target_ordinal = int(match_ordinal) if match_ordinal is not None else None
    except (TypeError, ValueError):
        target_ordinal = None
    counts: defaultdict[str, int] = defaultdict(int)
    candidates: list[dict[str, Any]] = []
    for match in matches:
        preceding_starts = [start for start in starts if start.start() < match.start()]
        start = preceding_starts[-1] if preceding_starts else None
        current_tab = _arg(start.group("body"), "tab") if start else None
        tab_key = _norm_tab(current_tab)
        counts[tab_key] += 1
        governed = [row["patch"] for row in patches if row["offset"] < match.start()]
        candidates.append(
            {
                "tab": current_tab,
                "tab_key": tab_key,
                "ordinal": counts[tab_key],
                "patch": governed[-1] if governed else None,
                "offset": match.start(),
            }
        )
    selected = [
        row
        for row in candidates
        if row["tab_key"] == target_tab and target_ordinal is not None and row["ordinal"] == target_ordinal
    ]
    if len(selected) == 1 and selected[0].get("patch"):
        return {
            "status": "exact",
            "patch": selected[0]["patch"],
            "tab": selected[0]["tab"],
            "match_ordinal": selected[0]["ordinal"],
            "candidate_count": len(selected),
            "page_patch_values": sorted({row["patch"] for row in patches}),
        }
    unique_page_patches = sorted({row["patch"] for row in patches})
    if len(unique_page_patches) == 1:
        return {
            "status": "page_single_patch_fallback",
            "patch": unique_page_patches[0],
            "tab": tab,
            "match_ordinal": target_ordinal,
            "candidate_count": len(selected),
            "page_patch_values": unique_page_patches,
        }
    return {
        "status": "blocked",
        "patch": None,
        "tab": tab,
        "match_ordinal": target_ordinal,
        "candidate_count": len(selected),
        "page_patch_values": unique_page_patches,
        "blocker": "historical_patch_match_block_not_resolved",
    }


def _select_revision(revisions: Iterable[Mapping[str, Any]], cutoff: datetime) -> Mapping[str, Any] | None:
    eligible = []
    for revision in revisions:
        try:
            timestamp = _parse_time(revision.get("revision_timestamp"))
        except Exception:
            continue
        if timestamp < cutoff:
            eligible.append((timestamp, int(revision.get("revision_id", 0)), revision))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1]))
    return eligible[-1][2]


def _load_retrospective_receipts(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    receipt_path = Path(str(manifest.get("receipt_file", "")))
    if not receipt_path.is_absolute() and not receipt_path.exists():
        receipt_path = path.parent / receipt_path
    if not receipt_path.exists():
        return {}
    rows = {}
    for line in receipt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, Mapping) and row.get("fixture_id"):
            rows[str(row["fixture_id"])] = row
    return rows


def _capture_schedule_rows(
    match_ids: list[str],
    *,
    output_dir: Path,
    delay: float,
    timeout: float,
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    schedule_dir = output_dir / "raw" / "schedule"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pages: list[dict[str, Any]] = []
    for index, batch in enumerate(_batches(match_ids)):
        url = _cargo_url(batch)
        raw, payload = _fetch(url, timeout=timeout)
        if not isinstance(payload, list):
            raise PatchRevisionError(f"schedule response is not an array: page {index}")
        filename = f"schedule-{index:03d}.json"
        _write_atomic(schedule_dir / filename, raw)
        pages.append(
            {
                "raw_file": str(Path("raw") / "schedule" / filename),
                "source_url": url,
                "payload_sha256": _sha_bytes(raw),
                "row_count": len(payload),
                "requested_fields": list(SCHEDULE_FIELDS),
            }
        )
        for row in payload:
            if isinstance(row, Mapping) and row.get("MatchId"):
                by_match[str(row["MatchId"])].append(dict(row))
        if delay:
            import time

            time.sleep(delay)
    selected: dict[str, Mapping[str, Any]] = {}
    for match_id, rows in by_match.items():
        if len(rows) == 1:
            selected[match_id] = rows[0]
    return selected, pages


def _capture_page_history(
    page: str,
    *,
    newest: datetime,
    oldest: datetime,
    output_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    page_dir = output_dir / "history" / re.sub(r"[^A-Za-z0-9._-]+", "-", page).strip("-")
    page_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = page_dir / "history-manifest.json"
    params = _history_params(page, newest=newest, oldest=oldest)
    raw_pages: list[dict[str, Any]] = []
    revisions: dict[int, dict[str, Any]] = {}
    title: str | None = None
    page_index = 0
    while True:
        url = _api_url(params)
        raw, payload = _fetch(url, timeout=timeout)
        filename = f"history-{page_index:03d}.json"
        _write_atomic(page_dir / filename, raw)
        page_title, page_revisions = _page_revisions(payload)
        title = title or page_title
        for revision in page_revisions:
            revisions[int(revision["revision_id"])] = revision
        raw_pages.append(
            {
                "raw_file": str(Path("history") / page_dir.name / filename),
                "source_url": url,
                "payload_sha256": _sha_bytes(raw),
                "revision_count": len(page_revisions),
            }
        )
        continuation = payload.get("continue") if isinstance(payload, Mapping) else None
        if not isinstance(continuation, Mapping) or not continuation.get("rvcontinue"):
            break
        params = dict(params)
        params["rvcontinue"] = str(continuation["rvcontinue"])
        params["continue"] = str(continuation.get("continue", "||"))
        page_index += 1
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "page": page,
        "resolved_title": title,
        "newest": _rfc(newest),
        "oldest": _rfc(oldest),
        "captured_at": _now(),
        "pages": raw_pages,
        "revisions": sorted(revisions.values(), key=lambda row: (row["revision_timestamp"], row["revision_id"])),
        "status": "ok" if revisions else "no_revision_in_window",
    }
    manifest = {**unsigned, "manifest_sha256": _sha_object(unsigned)}
    _write_json(manifest_path, manifest)
    return manifest


def _capture_revision(
    page: str,
    revision: Mapping[str, Any],
    *,
    output_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    page_dir = output_dir / "revisions" / re.sub(r"[^A-Za-z0-9._-]+", "-", page).strip("-")
    page_dir.mkdir(parents=True, exist_ok=True)
    revision_id = int(revision["revision_id"])
    url = _api_url(_revision_content_params(page, revision_id))
    raw, payload = _fetch(url, timeout=timeout)
    content = _content_from_payload(payload)
    if content is None:
        raise PatchRevisionError(f"revision has no wikitext: {page} rev={revision_id}")
    raw_path = page_dir / f"revision-{revision_id}.json"
    _write_atomic(raw_path, raw)
    payload_path = page_dir / f"payload-{revision_id}.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "page": page,
        "revision_id": content[0],
        "revision_timestamp": content[1],
        "source_url": url,
        "raw_file": str(Path("revisions") / page_dir.name / raw_path.name),
        "raw_payload_sha256": _sha_bytes(raw),
        "content_sha256": _sha_bytes(content[2].encode("utf-8")),
        "content": content[2],
    }
    _write_json(payload_path, result)
    return result


def build_receipts(
    run_dir: Path,
    output_dir: Path,
    *,
    retrospective_manifest: Path = DEFAULT_RETROSPECTIVE_MANIFEST,
    workers: int = 4,
    delay: float = 0.15,
    timeout: float = 60.0,
) -> dict[str, Any]:
    ledger = _load_ledger(run_dir / "frozen-ledger.jsonl")
    fixture_rows: list[dict[str, Any]] = []
    match_ids: list[str] = []
    for row in ledger:
        pregame = row["pregame"]
        fixture_id = str(pregame.get("fixture_id", ""))
        if not fixture_id:
            raise PatchRevisionError("frozen row has no fixture_id")
        match_id = fixture_id.rsplit("_", 1)[0]
        match_ids.append(match_id)
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "match_id": match_id,
                "event_start": str(pregame.get("event_start", "")),
                "as_of": str(pregame.get("as_of", "")),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_rows, schedule_pages = _capture_schedule_rows(
        sorted(set(match_ids)), output_dir=output_dir, delay=delay, timeout=timeout
    )
    if not schedule_rows:
        raise PatchRevisionError("no MatchSchedule rows matched the frozen fixtures")
    page_for_match: dict[str, str] = {}
    for match_id, row in schedule_rows.items():
        page = str(row.get("_pageName", "")).strip()
        if page:
            page_for_match[match_id] = page
    pages = sorted(set(page_for_match.values()))
    event_times = [_parse_time(row["event_start"]) for row in fixture_rows]
    newest = max(event_times) + timedelta(days=1)
    oldest = min(event_times) - timedelta(days=370)
    history_manifests: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _capture_page_history,
                page,
                newest=newest,
                oldest=oldest,
                output_dir=output_dir,
                timeout=timeout,
            ): page
            for page in pages
        }
        for future in concurrent.futures.as_completed(futures):
            page = futures[future]
            history_manifests[page] = future.result()
    selected_by_fixture: dict[str, Mapping[str, Any]] = {}
    for row in fixture_rows:
        schedule = schedule_rows.get(row["match_id"])
        if not schedule:
            continue
        page = page_for_match.get(row["match_id"])
        history = history_manifests.get(page or "", {})
        selected = _select_revision(history.get("revisions", []), _parse_time(row["as_of"]))
        if selected is not None:
            selected_by_fixture[row["fixture_id"]] = {
                **selected,
                "page": page,
                "schedule": schedule,
            }
    unique_revisions = {
        (str(selected["page"]), int(selected["revision_id"])): selected
        for selected in selected_by_fixture.values()
    }
    revision_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _capture_revision,
                page,
                selected,
                output_dir=output_dir,
                timeout=timeout,
            ): (page, revision_id)
            for (page, revision_id), selected in unique_revisions.items()
        }
        for future in concurrent.futures.as_completed(futures):
            revision_payloads[futures[future]] = future.result()
    retrospective = _load_retrospective_receipts(retrospective_manifest)
    receipts: list[dict[str, Any]] = []
    for row in fixture_rows:
        fixture_id = row["fixture_id"]
        schedule = schedule_rows.get(row["match_id"])
        selected = selected_by_fixture.get(fixture_id)
        blockers: list[str] = []
        evidence: dict[str, Any] = {
            "source_kind": "leaguepedia_data_page_revision",
            "schedule_row": {
                key: value
                for key, value in (schedule or {}).items()
                if key not in OUTCOME_FIELDS
            },
            "retrospective_patch": retrospective.get(fixture_id, {}).get("patch"),
        }
        patch: str | None = None
        if schedule is None:
            blockers.append("historical_patch_schedule_row_missing")
        elif selected is None:
            blockers.append("no_data_page_revision_strictly_before_cutoff")
        else:
            key = (str(selected["page"]), int(selected["revision_id"]))
            payload = revision_payloads.get(key)
            if payload is None:
                blockers.append("historical_patch_revision_payload_missing")
            else:
                parsed = extract_match_patch(
                    payload["content"],
                    tab=schedule.get("Tab"),
                    match_ordinal=schedule.get("N MatchInTab"),
                )
                patch = parsed.get("patch")
                evidence.update(
                    {
                        "data_page": selected["page"],
                        "revision_id": selected["revision_id"],
                        "revision_timestamp": selected["revision_timestamp"],
                        "revision_source_url": payload["source_url"],
                        "revision_payload_sha256": payload["raw_payload_sha256"],
                        "content_sha256": payload["content_sha256"],
                        "extraction": parsed,
                    }
                )
                if parsed.get("status") != "exact":
                    blockers.append(str(parsed.get("blocker", "historical_patch_extraction_not_exact")))
                if not patch:
                    blockers.append("historical_patch_field_missing")
                retrospective_patch = retrospective.get(fixture_id, {}).get("patch")
                if retrospective_patch and patch and str(retrospective_patch) != patch:
                    blockers.append("historical_patch_conflicts_with_retrospective_patch")
        authority_status = "pre_event_revision" if patch and not blockers else "unavailable"
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture_id,
            "event_start": row["event_start"],
            "as_of": row["as_of"],
            "patch": patch,
            "client_patch": f"16.{int(patch.split('.', 1)[1])}" if patch and patch.startswith("26.") else None,
            "authority_status": authority_status,
            "pregame_authorized": authority_status == "pre_event_revision",
            "blockers": sorted(set(blockers)),
            "evidence": evidence,
        }
        receipt["evidence_hash"] = _sha_object({"fixture_id": fixture_id, "evidence": evidence})
        receipts.append(receipt)
    receipts.sort(key=lambda row: (row["event_start"], row["fixture_id"]))
    receipt_path = output_dir / "patch-receipts.jsonl"
    receipt_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipts).encode("utf-8")
    _write_atomic(receipt_path, receipt_bytes)
    pre_event = sum(row["authority_status"] == "pre_event_revision" for row in receipts)
    blocker_counts: dict[str, int] = defaultdict(int)
    for row in receipts:
        for blocker in row["blockers"]:
            blocker_counts[blocker] += 1
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "captured_at": _now(),
        "fixture_count": len(receipts),
        "pre_event_revision_fixture_count": pre_event,
        "pregame_authorized_fixture_count": pre_event,
        "unavailable_fixture_count": len(receipts) - pre_event,
        "schedule_pages": schedule_pages,
        "history_manifest_count": len(history_manifests),
        "selected_revision_count": len(revision_payloads),
        "retrospective_manifest": str(retrospective_manifest),
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": _sha_bytes(receipt_bytes),
        "outcome_fields_requested": [],
        "outcome_fields_emitted": False,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "claim_ceiling": {
            "pre_event_patch_authority": pre_event == len(receipts),
            "winner_prediction": False,
            "publication": False,
        },
    }
    manifest = {**unsigned, "manifest_sha256": _sha_object(unsigned)}
    _write_json(output_dir / "receipt-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retrospective-manifest", type=Path, default=DEFAULT_RETROSPECTIVE_MANIFEST)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    try:
        result = build_receipts(
            args.run_dir,
            args.output_dir,
            retrospective_manifest=args.retrospective_manifest,
            workers=args.workers,
            delay=args.delay,
            timeout=args.timeout,
        )
    except (OSError, ValueError, PatchRevisionError) as exc:
        print(str(exc))
        return 2
    summary = {
        key: result.get(key)
        for key in (
            "fixture_count",
            "pre_event_revision_fixture_count",
            "pregame_authorized_fixture_count",
            "unavailable_fixture_count",
            "history_manifest_count",
            "selected_revision_count",
            "manifest_sha256",
            "blocker_counts",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
