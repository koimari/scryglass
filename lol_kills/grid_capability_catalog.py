"""Private, read-only GRID schema and capability catalog.

This module never downloads match files or opens the Series Events WebSocket.
It performs GraphQL introspection, bounded metadata probes, and optional
inspection of already-local Series Events archives. Secret values are used
only in request headers and are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.etl.grid_ingest import (
    FILE_LIST_BASE,
    GRAPHQL_ENDPOINT,
    SERIES_ENDPOINT,
    _api_key,
    _headers,
)
from lol_kills.net import require_https_url


CATALOG_SCHEMA = "scryglass.grid.capability-catalog.v1"
CATALOG_VERSION = "1.0.0"
DEFAULT_LOCAL_EVENTS_DIR = Path("data/lol/warehouse/raw_grid")
DEFAULT_OUTPUT = (
    Path.home()
    / ".codex"
    / "skills"
    / "query-grid-research"
    / "assets"
    / "grid-capability-catalog.v1.json"
)

INTROSPECTION_QUERY = """
query ScryglassCapabilityIntrospection {
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      kind
      name
      description
      fields(includeDeprecated: true) {
        name
        description
        isDeprecated
        deprecationReason
        args {
          name
          description
          defaultValue
          type {
            kind name
            ofType {
              kind name
              ofType {
                kind name
                ofType { kind name }
              }
            }
          }
        }
        type {
          kind name
          ofType {
            kind name
            ofType {
              kind name
              ofType { kind name }
            }
          }
        }
      }
      inputFields {
        name
        description
        defaultValue
        type {
          kind name
          ofType {
            kind name
            ofType {
              kind name
              ofType { kind name }
            }
          }
        }
      }
      interfaces { kind name }
      enumValues(includeDeprecated: true) {
        name
        description
        isDeprecated
        deprecationReason
      }
      possibleTypes { kind name }
    }
    directives {
      name
      description
      locations
      args {
        name
        description
        defaultValue
        type {
          kind name
          ofType {
            kind name
            ofType { kind name }
          }
        }
      }
    }
  }
}
""".strip()

CENTRAL_METADATA_QUERY = """
query ScryglassMetadataProbe {
  title(id: 3) { id name nameShortened }
  dataProviders { name description }
  seriesFormats { id name }
}
""".strip()

SERIES_STATE_METADATA_QUERY = """
query ScryglassSeriesStateMetadataProbe($id: ID!) {
  seriesState(id: $id) {
    id
    valid
    started
    finished
    updatedAt
    games {
      id
      type
      started
      finished
    }
  }
}
""".strip()

SAFE_HEADER_NAMES = {
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "ratelimit-limit",
    "ratelimit-remaining",
    "ratelimit-reset",
}


class GridCatalogError(RuntimeError):
    """A credential-free catalog discovery failure."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        str(name).lower(): str(value)
        for name, value in headers.items()
        if str(name).lower() in SAFE_HEADER_NAMES
    }


def _request_json_with_headers(
    url: str,
    key: str,
    *,
    method: str = "GET",
    body: Mapping[str, Any] | None = None,
    timeout: float = 30,
) -> tuple[dict[str, Any], dict[str, str], int]:
    url = require_https_url(url, hosts={"api.grid.gg"}, allow_subdomains=True)
    encoded = _canonical_bytes(body) if body is not None else None
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=_headers(key, json_body=body is not None),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status = int(getattr(response, "status", 200))
            headers = _safe_headers(dict(response.headers.items()))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        detail = detail.replace(key, "***")
        raise GridCatalogError(
            f"GRID request failed with HTTP {exc.code} at {url}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GridCatalogError(
            f"GRID request failed at {url}: {type(exc.reason).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise GridCatalogError(f"GRID returned non-object JSON at {url}")
    return payload, headers, status


def _graphql(
    endpoint: str,
    key: str,
    query: str,
    variables: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str], int]:
    payload, headers, status = _request_json_with_headers(
        endpoint,
        key,
        method="POST",
        body={"query": query, "variables": dict(variables or {})},
    )
    errors = payload.get("errors")
    if errors:
        messages = [
            str(error.get("message") or "unknown GraphQL error")[:240]
            for error in errors
            if isinstance(error, Mapping)
        ]
        raise GridCatalogError(
            f"GRID GraphQL rejected the safe discovery query at {endpoint}: "
            + "; ".join(messages or ["unknown error"])
        )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise GridCatalogError(f"GRID GraphQL response has no data at {endpoint}")
    return data, headers, status


def _named_type(type_ref: Mapping[str, Any] | None) -> str | None:
    current = type_ref
    while isinstance(current, Mapping):
        name = current.get("name")
        if name:
            return str(name)
        current = current.get("ofType")
    return None


def _type_expression(type_ref: Mapping[str, Any] | None) -> str:
    if not isinstance(type_ref, Mapping):
        return "UNKNOWN"
    kind = str(type_ref.get("kind") or "")
    name = type_ref.get("name")
    if kind == "NON_NULL":
        return f"{_type_expression(type_ref.get('ofType'))}!"
    if kind == "LIST":
        return f"[{_type_expression(type_ref.get('ofType'))}]"
    return str(name or kind or "UNKNOWN")


def _root_operations(schema: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    types = {
        str(row.get("name")): row
        for row in schema.get("types") or []
        if isinstance(row, Mapping) and row.get("name")
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for operation in ("queryType", "mutationType", "subscriptionType"):
        root = schema.get(operation)
        root_name = str(root.get("name") or "") if isinstance(root, Mapping) else ""
        if not root_name:
            result[operation] = []
            continue
        type_row = types.get(root_name) or {}
        result[operation] = [
            {
                "name": str(field.get("name") or ""),
                "return_type": _type_expression(field.get("type")),
                "return_named_type": _named_type(field.get("type")),
                "arguments": [
                    {
                        "name": str(argument.get("name") or ""),
                        "type": _type_expression(argument.get("type")),
                        "default_value": argument.get("defaultValue"),
                    }
                    for argument in field.get("args") or []
                    if isinstance(argument, Mapping)
                ],
                "deprecated": bool(field.get("isDeprecated")),
            }
            for field in type_row.get("fields") or []
            if isinstance(field, Mapping)
        ]
    return result


def _field_availability(endpoint_id: str, type_name: str, field_name: str) -> dict[str, str]:
    if endpoint_id == "central_data":
        return {
            "class": "metadata_or_schedule",
            "basis": "central_data_endpoint",
            "leakage_rule": "Do not use mutable or post-start metadata as pregame evidence without an as-of receipt.",
        }
    is_final_signal = (
        field_name in {"finished", "forfeited"}
        and type_name in {"SeriesState", "GameState", "SegmentState"}
    ) or (field_name == "won" and "TeamState" in type_name)
    if is_final_signal:
        return {
            "class": "final_outcome_signal",
            "basis": "field_semantics",
            "leakage_rule": "Outcome signal: never use as a checkpoint predictor.",
        }
    return {
        "class": "live_and_final_snapshot_or_unknown",
        "basis": "series_state_endpoint_schema_only",
        "leakage_rule": "Use only a captured value whose provider game clock is at or before the checkpoint; schema presence does not prove historical availability.",
    }


def _annotated_fields(endpoint_id: str, schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for type_row in schema.get("types") or []:
        if not isinstance(type_row, Mapping):
            continue
        type_name = str(type_row.get("name") or "")
        if not type_name or type_name.startswith("__"):
            continue
        for field in type_row.get("fields") or []:
            if not isinstance(field, Mapping):
                continue
            rows.append(
                {
                    "type": type_name,
                    "field": str(field.get("name") or ""),
                    "graphql_type": _type_expression(field.get("type")),
                    "availability": _field_availability(
                        endpoint_id, type_name, str(field.get("name") or "")
                    ),
                }
            )
    return rows


def _schema_record(
    endpoint_id: str,
    endpoint: str,
    schema: Mapping[str, Any],
    headers: Mapping[str, str],
    status: int,
) -> dict[str, Any]:
    public_types = [
        row
        for row in schema.get("types") or []
        if isinstance(row, Mapping)
        and row.get("name")
        and not str(row["name"]).startswith("__")
    ]
    return {
        "endpoint_id": endpoint_id,
        "url": endpoint,
        "transport": "https_graphql",
        "authenticated": True,
        "read_only_discovery": True,
        "http_status": status,
        "rate_limit_headers_observed": dict(headers),
        "schema_sha256": _sha256(schema),
        "type_count_excluding_introspection": len(public_types),
        "root_operations": _root_operations(schema),
        "field_availability_annotations": _annotated_fields(endpoint_id, schema),
        "introspection": schema,
    }


def _safe_metadata_summary(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_metadata_summary(child)
            for key, child in sorted(value.items())
            if not any(
                marker in str(key).lower()
                for marker in ("url", "token", "key", "secret", "signature")
            )
        }
    if isinstance(value, list):
        return [_safe_metadata_summary(child) for child in value]
    return value


def _probe_file_listing(
    key: str, series_id: str
) -> dict[str, Any]:
    url = f"{FILE_LIST_BASE}/{series_id}"
    payload, headers, status = _request_json_with_headers(url, key)
    files = payload.get("files")
    if not isinstance(files, list):
        return {
            "status": "unavailable",
            "blocker": "file_listing_response_missing_files_array",
            "http_status": status,
            "rate_limit_headers_observed": headers,
        }
    summaries = []
    for row in files:
        if not isinstance(row, Mapping):
            continue
        safe = _safe_metadata_summary(row)
        file_id = str(safe.get("id") or "")
        if file_id in {"state-grid", "state-details-riot-game-1", "state-summary-riot-game-1"}:
            temporal_class = "final_game_only_artifact"
            leakage_rule = "Outcome/final-state artifact; labels and verification only, never checkpoint predictors."
        elif file_id.startswith("replay-"):
            temporal_class = "completed_game_replay_artifact"
            leakage_rule = "Replay contains the whole game; derive checkpoints only with strict event-time filtering."
        elif file_id.startswith("events-"):
            temporal_class = "timestamped_event_stream"
            leakage_rule = "May contain the whole game; use only events at or before the declared checkpoint."
        else:
            temporal_class = "unknown"
            leakage_rule = "Unavailable for modeling until temporal semantics are verified."
        summaries.append(
            {
                "metadata_keys": sorted(str(key) for key in safe),
                "metadata": safe,
                "temporal_class": temporal_class,
                "leakage_rule": leakage_rule,
            }
        )
    return {
        "status": "confirmed",
        "series_id": str(series_id),
        "http_status": status,
        "rate_limit_headers_observed": headers,
        "file_count": len(summaries),
        "files": summaries,
        "download_attempted": False,
        "signed_urls_retained": False,
    }


def _event_family(event: Mapping[str, Any]) -> str:
    event_type = event.get("type")
    if isinstance(event_type, Mapping):
        return ".".join(
            str(event_type.get(part) or "*")
            for part in ("actor", "action", "target")
        )
    if isinstance(event_type, str) and event_type.strip():
        return event_type.strip()
    actor = event.get("actor") if isinstance(event.get("actor"), str) else "*"
    action = event.get("action") if isinstance(event.get("action"), str) else "*"
    target = event.get("target") if isinstance(event.get("target"), str) else "*"
    if actor != "*" or action != "*" or target != "*":
        return ".".join((actor, action, target))
    return str(event_type or "unknown")


def scan_local_event_archives(root: Path) -> dict[str, Any]:
    archives = sorted(root.glob("*_grid.jsonl.zip")) if root.is_dir() else []
    families: Counter[str] = Counter()
    family_fields: dict[str, Counter[str]] = defaultdict(Counter)
    transaction_fields: Counter[str] = Counter()
    full_state_locations: Counter[str] = Counter()
    series_ids: set[str] = set()
    archive_hashes: list[dict[str, Any]] = []
    transaction_count = 0
    event_count = 0
    for path in archives:
        digest = hashlib.sha256()
        with path.open("rb") as raw_handle:
            for chunk in iter(lambda: raw_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        archive_hashes.append(
            {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        )
        with zipfile.ZipFile(path) as archive:
            members = sorted(name for name in archive.namelist() if name.endswith(".jsonl"))
            for member in members:
                with archive.open(member) as handle:
                    for raw in handle:
                        if not raw.strip():
                            continue
                        row = json.loads(raw)
                        if not isinstance(row, Mapping):
                            continue
                        transaction_count += 1
                        transaction_fields.update(str(key) for key in row)
                        if row.get("seriesId") is not None:
                            series_ids.add(str(row["seriesId"]))
                        if isinstance(row.get("seriesState"), Mapping):
                            full_state_locations["transaction.seriesState"] += 1
                        for event in row.get("events") or []:
                            if not isinstance(event, Mapping):
                                continue
                            event_count += 1
                            family = _event_family(event)
                            families[family] += 1
                            family_fields[family].update(str(key) for key in event)
                            if isinstance(event.get("seriesState"), Mapping):
                                full_state_locations["event.seriesState"] += 1
    return {
        "source": "already_local_series_events_archives",
        "archive_count": len(archives),
        "archive_receipts": archive_hashes,
        "series_ids": sorted(series_ids),
        "transaction_count": transaction_count,
        "event_count": event_count,
        "transaction_fields": sorted(transaction_fields),
        "full_state_locations": dict(sorted(full_state_locations.items())),
        "event_families": [
            {
                "family": family,
                "observed_count": count,
                "fields": sorted(family_fields[family]),
                "coverage_status": "observed_in_local_fixture_not_exhaustive",
            }
            for family, count in sorted(families.items())
        ],
    }


def _capability_matrix(
    endpoints: Sequence[Mapping[str, Any]],
    file_listing: Mapping[str, Any],
    local_events: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root_names = {
        endpoint["endpoint_id"]: {
            row["name"]
            for row in endpoint.get("root_operations", {}).get("queryType", [])
        }
        for endpoint in endpoints
    }
    return [
        {
            "capability": "central_metadata",
            "status": "confirmed" if "allSeries" in root_names.get("central_data", set()) else "unverified",
            "evidence": "authenticated_graphql_introspection",
        },
        {
            "capability": "identity_crosswalk_queries",
            "status": "confirmed"
            if {
                "gameIdByExternalId",
                "playerIdByExternalId",
                "seriesIdByExternalId",
                "teamIdByExternalId",
            }.issubset(root_names.get("central_data", set()))
            else "partial_or_unverified",
            "evidence": "authenticated_graphql_introspection",
        },
        {
            "capability": "live_or_final_series_state_snapshot",
            "status": "confirmed"
            if "seriesState" in root_names.get("series_state", set())
            else "unverified",
            "evidence": "authenticated_graphql_introspection",
        },
        {
            "capability": "historical_file_listing",
            "status": str(file_listing.get("status") or "unverified"),
            "evidence": "bounded_authenticated_file_list_probe",
        },
        {
            "capability": "historical_file_download",
            "status": "not_tested",
            "evidence": "explicitly_out_of_scope_no_download_attempted",
        },
        {
            "capability": "series_events_websocket",
            "status": "locally_observed_and_configured"
            if int(local_events.get("archive_count") or 0) > 0
            else "unverified",
            "evidence": "existing_local_archives_and_integration_code_no_new_connection",
        },
        {
            "capability": "bookmaker_or_market_odds",
            "status": "not_exposed_in_introspected_graphql_schemas",
            "evidence": "no_root_operation_or_type_name_identified",
        },
    ]


def build_catalog(
    *,
    key: str,
    local_events_dir: Path,
    probe_series_id: str | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    endpoints: list[dict[str, Any]] = []
    for endpoint_id, endpoint in (
        ("central_data", GRAPHQL_ENDPOINT),
        ("series_state", SERIES_ENDPOINT),
    ):
        data, headers, status = _graphql(endpoint, key, INTROSPECTION_QUERY)
        schema = data.get("__schema")
        if not isinstance(schema, Mapping):
            raise GridCatalogError(f"{endpoint_id} introspection returned no __schema")
        endpoints.append(
            _schema_record(endpoint_id, endpoint, schema, headers, status)
        )

    central_probe, central_headers, central_status = _graphql(
        GRAPHQL_ENDPOINT, key, CENTRAL_METADATA_QUERY
    )
    metadata_probes: dict[str, Any] = {
        "central_data": {
            "status": "confirmed",
            "http_status": central_status,
            "rate_limit_headers_observed": central_headers,
            "result": _safe_metadata_summary(central_probe),
        }
    }
    file_listing: dict[str, Any] = {
        "status": "not_tested",
        "blocker": "no_probe_series_id",
    }
    if probe_series_id:
        state_probe, state_headers, state_status = _graphql(
            SERIES_ENDPOINT,
            key,
            SERIES_STATE_METADATA_QUERY,
            {"id": str(probe_series_id)},
        )
        metadata_probes["series_state"] = {
            "status": "confirmed",
            "http_status": state_status,
            "rate_limit_headers_observed": state_headers,
            "result": _safe_metadata_summary(state_probe),
        }
        file_listing = _probe_file_listing(key, str(probe_series_id))

    local_events = scan_local_event_archives(local_events_dir)
    catalog: dict[str, Any] = {
        "schema_version": CATALOG_SCHEMA,
        "catalog_version": CATALOG_VERSION,
        "generated_at": generated_at or _utc_now(),
        "scope": {
            "privacy": "private_personal_research_only",
            "publication": False,
            "redistribution": False,
            "model_authority": False,
            "market_edge_claim_authority": False,
        },
        "provenance": {
            "method": "authenticated_read_only_graphql_introspection_and_bounded_metadata_probes",
            "credentials_serialized": False,
            "match_files_downloaded": False,
            "websocket_connections_opened": False,
            "introspection_query_sha256": _sha256(INTROSPECTION_QUERY),
            "local_event_source_sha256": _sha256(local_events.get("archive_receipts") or []),
        },
        "endpoints": endpoints,
        "metadata_probes": metadata_probes,
        "file_listing_probe": file_listing,
        "local_series_events_observations": local_events,
        "capabilities": _capability_matrix(endpoints, file_listing, local_events),
        "pagination": {
            "central_data": {
                "style": "cursor_connection",
                "arguments_observed": ["after", "before", "first", "last"],
                "page_info_fields": ["hasNextPage", "hasPreviousPage", "startCursor", "endCursor"],
                "safe_rule": "Use a bounded first value, follow endCursor only while needed, and record every cutoff.",
            },
            "series_state": {
                "style": "single_series_id",
                "safe_rule": "Query one verified series id; do not scan guessed identifiers.",
            },
            "file_download": {
                "style": "series_scoped_listing",
                "safe_rule": "List only an already-verified series id; do not download in discovery mode.",
            },
        },
        "rate_limits": {
            "observed_response_headers": {
                endpoint["endpoint_id"]: endpoint["rate_limit_headers_observed"]
                for endpoint in endpoints
            },
            "documented_or_local_integration": {
                "series_state_graphql": "unknown_from_schema",
                "series_events_websocket": "unknown_from_schema",
                "file_download": "HTTP 429 is handled by existing integration; contractual limit not established here",
            },
            "safe_default": "Serialize requests, use bounded pages, honor Retry-After, and stop on 429.",
        },
        "identity_rules": [
            "Treat GRID series, game, team, and player IDs as provider identifiers, never Riot IDs.",
            "Resolve external IDs only through explicit externalLinks or *IdByExternalId queries.",
            "Require one-to-one provider-to-Riot game and team mappings; quarantine ambiguity or conflict.",
            "Never infer PUUID, platform game ID, side, player identity, or roster from names alone.",
            "Bind every derived row to exact schema, query, source cutoff, and response/content hashes.",
        ],
        "freshness_and_leakage": [
            "Central metadata updatedAt is source freshness, not checkpoint game time.",
            "SeriesState updatedAt is not sufficient for checkpoint eligibility; retain provider game clock and sequence.",
            "For a checkpoint, select only state known at or before the checkpoint and enforce the declared maximum-age rule.",
            "Never use finished, won, final duration, final totals, or post-checkpoint revisions as predictors.",
            "Schema presence does not prove that a field was populated at the historical checkpoint.",
        ],
        "known_limitations": [
            "GraphQL introspection describes shape, not account-wide row coverage or historical retention.",
            "Field availability annotations are conservative; unknown fields require timestamped observations before modeling.",
            "Local event-family observations cover only the listed archive receipts and are not an exhaustive GRID event taxonomy.",
            "Series Events has no GraphQL schema and was not connected during this discovery.",
            "Historical file payload download and completeness were intentionally not tested.",
            "No bookmaker or market-odds capability appears in the introspected schemas; separate product access remains unverified.",
            "No authenticated prospective latency claim is supported by this catalog.",
        ],
    }
    catalog["catalog_sha256"] = _sha256(
        {key: value for key, value in catalog.items() if key != "catalog_sha256"}
    )
    return catalog


def write_catalog(path: Path, catalog: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        catalog, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a private, read-only GRID capability catalog."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-events-dir", type=Path, default=DEFAULT_LOCAL_EVENTS_DIR)
    parser.add_argument(
        "--probe-series-id",
        help="Already-local verified GRID series ID for state/file-list metadata only.",
    )
    parser.add_argument("--grid-env-file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    key = _api_key(args.grid_env_file)
    catalog = build_catalog(
        key=key,
        local_events_dir=args.local_events_dir,
        probe_series_id=args.probe_series_id,
    )
    write_catalog(args.out, catalog)
    print(
        json.dumps(
            {
                "status": "ok",
                "path": str(args.out.expanduser().resolve()),
                "catalog_sha256": catalog["catalog_sha256"],
                "endpoint_type_counts": {
                    row["endpoint_id"]: row["type_count_excluding_introspection"]
                    for row in catalog["endpoints"]
                },
                "match_files_downloaded": False,
                "credentials_serialized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
