"""Capture exact public Betano Brazil help-center terms without authority.

The snapshot records the exact response bytes and system-clocked transport
window for the public League of Legends market description and the general
postponement/cancellation article.  Those pages are useful provenance but do
not resolve every esports remake, forfeit, or market-specific settlement case,
so the artifact is explicitly incomplete and non-authorizing.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from lol_kills.v2.data.common import sha256_canonical_object


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "scryglass:betano-br-public-terms-snapshot:v1"
RESULT_STATE = "PUBLIC_TERMS_BYTES_CAPTURED_INCOMPLETE_NON_AUTHORIZING"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/bookmaker-terms-snapshot-v1.json"
)
SOURCE_LOCATOR = "lol_kills/v2/market/betano_terms_snapshot_v1.py"
MAX_SOURCE_BYTES = 2_000_000
LOL_ARTICLE_ID = 34703909314589
CANCELLATION_ARTICLE_ID = 6414148470301
SOURCE_SPECS = (
    (
        "league_of_legends_market_description",
        LOL_ARTICLE_ID,
        "https://support.betano.bet.br/api/v2/help_center/pt-br/articles/34703909314589.json",
    ),
    (
        "general_postponement_and_cancellation",
        CANCELLATION_ARTICLE_ID,
        "https://support.betano.bet.br/api/v2/help_center/pt-br/articles/6414148470301.json",
    ),
)
AUTHORITY = {
    "bookmaker_terms_authority": False,
    "settlement_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Exact public help-center bytes and limited extracted statements only. "
    "This incomplete snapshot is not bookmaker-terms alignment review, "
    "settlement authority, quote authority, or betting authority."
)


class BetanoTermsSnapshotError(RuntimeError):
    """The public terms snapshot is missing, malformed, or overclaimed."""


def _clock_sample(clock: Callable[[], datetime], label: str) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise BetanoTermsSnapshotError(
            f"{label} clock must return a timezone-aware datetime"
        )
    return observed.astimezone(timezone.utc)


def _time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise BetanoTermsSnapshotError(f"{label} must be RFC-3339")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BetanoTermsSnapshotError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise BetanoTermsSnapshotError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise BetanoTermsSnapshotError(
                    f"{label} contains duplicate key {key!r}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except BetanoTermsSnapshotError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise BetanoTermsSnapshotError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise BetanoTermsSnapshotError(f"{label} must be a JSON object")
    return value


def _article(raw: bytes, *, expected_id: int, label: str) -> dict[str, Any]:
    payload = _strict_object(raw, label)
    article = payload.get("article")
    if not isinstance(article, Mapping):
        raise BetanoTermsSnapshotError(f"{label} article is missing")
    if article.get("id") != expected_id:
        raise BetanoTermsSnapshotError(f"{label} article id changed")
    if article.get("locale") != "pt-br" or article.get("source_locale") != "pt-br":
        raise BetanoTermsSnapshotError(f"{label} locale changed")
    if article.get("draft") is not False or article.get("outdated") is not False:
        raise BetanoTermsSnapshotError(f"{label} is draft or outdated")
    for field in ("html_url", "title", "updated_at", "edited_at", "body"):
        if not isinstance(article.get(field), str) or not article[field].strip():
            raise BetanoTermsSnapshotError(f"{label} {field} is missing")
    html_url = urlparse(article["html_url"])
    if html_url.scheme != "https" or html_url.hostname != "support.betano.bet.br":
        raise BetanoTermsSnapshotError(f"{label} html URL is not official Betano")
    _time(article["updated_at"], f"{label}.updated_at")
    _time(article["edited_at"], f"{label}.edited_at")
    return dict(article)


def _extraction(name: str, article: Mapping[str, Any]) -> dict[str, Any]:
    body = str(article["body"])
    if name == "league_of_legends_market_description":
        markers = (
            "Vencedor do Mapa:",
            "quem destruir a base inimiga.",
        )
        meaning = "map_winner_is_team_that_destroys_enemy_base"
    elif name == "general_postponement_and_cancellation":
        markers = (
            'princípio do "Mesmo Dia"',
            "a aposta continua válida",
            "a aposta será anulada e o valor será reembolsado",
            "todas as apostas pendentes relacionadas a esse evento serão anuladas e reembolsadas",
        )
        meaning = "general_same_day_continuation_and_next_day_void_refund_rule"
    else:
        raise BetanoTermsSnapshotError(f"unsupported source name: {name}")
    if any(marker not in body for marker in markers):
        raise BetanoTermsSnapshotError(f"{name} expected public wording is absent")
    return {
        "article_id": article["id"],
        "title": article["title"],
        "html_url": article["html_url"],
        "updated_at": article["updated_at"],
        "edited_at": article["edited_at"],
        "meaning": meaning,
        "required_markers_present": True,
        "extraction_is_complete_settlement_rule": False,
    }


def _fetch_source(
    *,
    name: str,
    article_id: int,
    url: str,
    opener: Callable[..., Any],
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "support.betano.bet.br"
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise BetanoTermsSnapshotError("terms endpoint is not the frozen public host")
    started = _clock_sample(clock, f"{name}.request_started")
    request = Request(
        url,
        headers={"User-Agent": "Scryglass-private-provenance-capture/1.0"},
        method="GET",
    )
    try:
        with opener(request, timeout=20) as response:
            raw = response.read(MAX_SOURCE_BYTES + 1)
            status = int(response.getcode())
            final_url = str(response.geturl())
            response_headers = {
                key.casefold(): value
                for key, value in response.headers.items()
                if key.casefold() in {"content-type", "etag", "last-modified"}
            }
    except (OSError, ValueError) as exc:
        raise BetanoTermsSnapshotError(f"{name} public fetch failed") from exc
    received = _clock_sample(clock, f"{name}.response_received")
    if received < started:
        raise BetanoTermsSnapshotError(f"{name} transport clock moved backwards")
    if status != 200 or not raw or len(raw) > MAX_SOURCE_BYTES:
        raise BetanoTermsSnapshotError(f"{name} response is empty or non-200")
    final = urlparse(final_url)
    if final.scheme != "https" or final.hostname not in {
        "support.betano.bet.br",
        "betanobr.zendesk.com",
    }:
        raise BetanoTermsSnapshotError(f"{name} final URL is not HTTPS")
    article = _article(raw, expected_id=article_id, label=name)
    return {
        "name": name,
        "article_id": article_id,
        "request_url": url,
        "final_url": final_url,
        "http_status": status,
        "response_headers": dict(sorted(response_headers.items())),
        "request_started_at_utc": started.isoformat(),
        "response_received_at_utc": received.isoformat(),
        "transport_elapsed_seconds": (received - started).total_seconds(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
        "extraction": _extraction(name, article),
    }


def _source_lock(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file():
        raise BetanoTermsSnapshotError("terms snapshot source is missing")
    raw = path.read_bytes()
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def capture_betano_terms_snapshot_v1(
    *,
    root: Path = ROOT,
    opener: Callable[..., Any] = urlopen,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    sources = [
        _fetch_source(
            name=name,
            article_id=article_id,
            url=url,
            opener=opener,
            clock=clock,
        )
        for name, article_id, url in SOURCE_SPECS
    ]
    locked_at = _clock_sample(clock, "snapshot.locked_at")
    latest_transport = max(
        _time(item["response_received_at_utc"], "response_received_at_utc")
        for item in sources
    )
    if locked_at < latest_transport:
        raise BetanoTermsSnapshotError("snapshot lock predates source transport")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": locked_at.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_fetch_and_builder",
            "observed_wall_clock_utc": locked_at.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "bookmaker_id": "betano-brazil",
        "sources": sources,
        "coverage": {
            "confirmed": [
                "public_lol_map_winner_description",
                "general_same_day_postponement_cancellation_handling",
            ],
            "unresolved": [
                "esports_specific_remake",
                "esports_specific_forfeit",
                "map_specific_void_priority",
                "bookmaker_result_provider",
                "conflicting_rule_priority",
                "price_acceptance_limits_and_execution",
            ],
            "complete_bookmaker_terms_snapshot": False,
            "independent_alignment_review_present": False,
        },
        "source_lock": _source_lock(root),
        "authority": dict(AUTHORITY),
        "decision_outputs": {
            "settled_result": None,
            "probability": None,
            "fair_odds": None,
            "expected_value": None,
            "recommendation": None,
            "stake": None,
        },
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_betano_terms_snapshot_v1(payload, root=root)


def _validate_source_record(
    record: Mapping[str, Any], *, name: str, article_id: int, url: str
) -> dict[str, Any]:
    expected_keys = {
        "name",
        "article_id",
        "request_url",
        "final_url",
        "http_status",
        "response_headers",
        "request_started_at_utc",
        "response_received_at_utc",
        "transport_elapsed_seconds",
        "raw_sha256",
        "raw_base64",
        "extraction",
    }
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise BetanoTermsSnapshotError(f"{name} source record changed")
    if (
        record.get("name") != name
        or record.get("article_id") != article_id
        or record.get("request_url") != url
        or record.get("http_status") != 200
    ):
        raise BetanoTermsSnapshotError(f"{name} source identity changed")
    started = _time(record.get("request_started_at_utc"), f"{name}.started")
    received = _time(record.get("response_received_at_utc"), f"{name}.received")
    elapsed = record.get("transport_elapsed_seconds")
    if (
        received < started
        or type(elapsed) not in (int, float)
        or float(elapsed) < 0
        or abs(float(elapsed) - (received - started).total_seconds()) > 1e-6
    ):
        raise BetanoTermsSnapshotError(f"{name} transport timing changed")
    try:
        raw = base64.b64decode(str(record.get("raw_base64")), validate=True)
    except (ValueError, TypeError) as exc:
        raise BetanoTermsSnapshotError(f"{name} raw base64 is invalid") from exc
    if (
        not raw
        or base64.b64encode(raw).decode("ascii") != record.get("raw_base64")
        or hashlib.sha256(raw).hexdigest() != record.get("raw_sha256")
    ):
        raise BetanoTermsSnapshotError(f"{name} raw bytes changed")
    article = _article(raw, expected_id=article_id, label=name)
    if record.get("extraction") != _extraction(name, article):
        raise BetanoTermsSnapshotError(f"{name} extraction changed")
    final = urlparse(str(record.get("final_url")))
    if (
        final.scheme != "https"
        or final.hostname not in {"support.betano.bet.br", "betanobr.zendesk.com"}
        or final.username is not None
        or final.password is not None
    ):
        raise BetanoTermsSnapshotError(f"{name} final URL is unsafe")
    headers = record.get("response_headers")
    if not isinstance(headers, Mapping) or any(
        key not in {"content-type", "etag", "last-modified"} for key in headers
    ):
        raise BetanoTermsSnapshotError(f"{name} response headers changed")
    return dict(record)


def validate_betano_terms_snapshot_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BetanoTermsSnapshotError("terms snapshot must be an object")
    value = dict(payload)
    expected_keys = {
        "schema_version",
        "result_state",
        "locked_at_utc",
        "clock_attestation",
        "bookmaker_id",
        "sources",
        "coverage",
        "source_lock",
        "authority",
        "decision_outputs",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected_keys:
        raise BetanoTermsSnapshotError("terms snapshot keys changed")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
        or value.get("bookmaker_id") != "betano-brazil"
    ):
        raise BetanoTermsSnapshotError("terms snapshot identity changed")
    unsigned = dict(value)
    declared = unsigned.pop("artifact_sha256", None)
    if declared != sha256_canonical_object(unsigned):
        raise BetanoTermsSnapshotError("terms snapshot canonical hash changed")
    locked_at = _time(value.get("locked_at_utc"), "locked_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_fetch_and_builder",
        "observed_wall_clock_utc": locked_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise BetanoTermsSnapshotError("terms snapshot clock attestation changed")
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != len(SOURCE_SPECS):
        raise BetanoTermsSnapshotError("terms source inventory changed")
    checked_sources = [
        _validate_source_record(record, name=name, article_id=article_id, url=url)
        for record, (name, article_id, url) in zip(sources, SOURCE_SPECS)
    ]
    if locked_at < max(
        _time(item["response_received_at_utc"], "source.received")
        for item in checked_sources
    ):
        raise BetanoTermsSnapshotError("terms lock predates source response")
    expected_coverage = {
        "confirmed": [
            "public_lol_map_winner_description",
            "general_same_day_postponement_cancellation_handling",
        ],
        "unresolved": [
            "esports_specific_remake",
            "esports_specific_forfeit",
            "map_specific_void_priority",
            "bookmaker_result_provider",
            "conflicting_rule_priority",
            "price_acceptance_limits_and_execution",
        ],
        "complete_bookmaker_terms_snapshot": False,
        "independent_alignment_review_present": False,
    }
    if value.get("coverage") != expected_coverage:
        raise BetanoTermsSnapshotError("terms coverage was overstated")
    source_lock = value.get("source_lock")
    expected_source_lock = _source_lock(root)
    if source_lock != expected_source_lock:
        raise BetanoTermsSnapshotError("terms capture source drifted")
    if value.get("authority") != AUTHORITY:
        raise BetanoTermsSnapshotError("terms snapshot exceeds authority")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise BetanoTermsSnapshotError("terms snapshot contains a decision output")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise BetanoTermsSnapshotError("terms snapshot claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace terms snapshot: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.out if args.out.is_absolute() else args.root / args.out
    try:
        payload = capture_betano_terms_snapshot_v1(root=args.root)
        raw_sha256 = write_no_clobber(output, payload)
    except (OSError, ValueError, BetanoTermsSnapshotError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "snapshot": str(output),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
                "complete_bookmaker_terms_snapshot": False,
                "authorizing": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
