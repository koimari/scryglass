"""Build a resumable, hash-addressed Obsidian vault from the League Wiki.

The vault is a research source mirror, not the mechanics engine.  Page
wikitext is retained so that a future parser can be audited against the
source revision; the generated front matter records the exact page and
revision used.  Numeric patch data must be reconciled separately against a
patch-pinned client-data source before it is allowed into a calculation.

Examples:

    python3 -m lol_kills.knowledge.league_wiki_vault inventory \
      --vault data/lol/knowledge/obsidian/league-wiki --namespace 0

    python3 -m lol_kills.knowledge.league_wiki_vault snapshot \
      --vault data/lol/knowledge/obsidian/league-wiki --namespace 0

The default is deliberately namespace 0 only.  ``--all-content-namespaces``
is available when a full wiki inventory is wanted, but it includes TFT, LoR,
WR, Universe, files, templates, and modules as well as League gameplay pages.
The operation can be interrupted and resumed; unchanged revisions are not
downloaded again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from lol_kills.net import require_https_url


API_URL = "https://wiki.leagueoflegends.com/en-us/api.php"
SOURCE_URL = "https://wiki.leagueoflegends.com/en-us/"
USER_AGENT = "Scryglass research source mirror/0.1 (+local research)"
SCHEMA_VERSION = "scryglass:league-wiki-vault:v1"
DEFAULT_VAULT = Path("data/lol/knowledge/obsidian/league-wiki")

# Namespaces with user-authored or source-bearing pages.  Namespace 0 is the
# normal gameplay/wiki article namespace.  The remaining namespaces are kept
# explicit so a caller cannot accidentally ingest talk pages or special pages.
CONTENT_NAMESPACES = (
    0,
    4,
    6,
    10,
    12,
    14,
    110,
    828,
    2900,
    3000,
    3002,
    3004,
    3006,
    3008,
    9592,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _api_request(params: dict[str, Any], *, retries: int = 4) -> dict[str, Any]:
    query = urlencode({key: str(value) for key, value in params.items()})
    url = require_https_url(
        f"{API_URL}?{query}", hosts={"wiki.leagueoflegends.com"}
    )
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            # A stalled Wiki request must not hold a resumable capture for
            # several minutes.  The caller can safely retry the page batch.
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            retry_after = 1.0
            if isinstance(exc, HTTPError) and exc.headers.get("Retry-After"):
                try:
                    retry_after = max(1.0, float(exc.headers["Retry-After"]))
                except ValueError:
                    pass
            time.sleep(retry_after * (attempt + 1))
    raise RuntimeError(f"League Wiki API request failed: {last_error}") from last_error


def _parse_namespaces(raw: str | None, all_content: bool) -> tuple[int, ...]:
    if all_content:
        return CONTENT_NAMESPACES
    if raw is None:
        return (0,)
    values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    if not values:
        raise ValueError("--namespace must contain at least one integer")
    if any(value < 0 for value in values):
        raise ValueError("negative namespaces are not content namespaces")
    return values


def _title_path(vault: Path, namespace: int, title: str) -> Path:
    """Return a safe Obsidian path while keeping subpage hierarchy readable."""

    pieces = [piece for piece in title.replace("\\", "/").split("/") if piece]
    if not pieces or any(piece in {".", ".."} for piece in pieces):
        raise ValueError(f"unsafe wiki title: {title!r}")
    encoded = [quote(piece, safe="-_.()'") for piece in pieces]
    # ``Path.with_suffix`` treats a title's final dot as a filename suffix and
    # causes collisions for titles such as ``A.D.M.I.N. (Teamfight Tactics)``.
    # Preserve the complete encoded title and append the Obsidian extension.
    return vault / "pages" / f"ns-{namespace}" / Path(*encoded[:-1]) / f"{encoded[-1]}.md"


def _path_collision_keys(vault: Path, namespaces: Iterable[int]) -> set[str]:
    """Return case-folded title paths that collide on macOS filesystems."""

    catalog_path = vault / "catalog.jsonl"
    if not catalog_path.exists():
        return set()
    counts: Counter[str] = Counter()
    wanted = set(namespaces)
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        namespace = int(row.get("namespace", -1))
        if namespace in wanted and namespace != 6:
            counts[str(_title_path(vault, namespace, str(row["title"]))).casefold()] += 1
    return {path for path, count in counts.items() if count > 1}


def _page_path(
    vault: Path,
    namespace: int,
    title: str,
    page_id: int,
    collision_keys: set[str],
) -> Path:
    base = _title_path(vault, namespace, title)
    if str(base).casefold() not in collision_keys:
        return base
    pieces = [piece for piece in title.replace("\\", "/").split("/") if piece]
    encoded = [quote(piece, safe="-_.()'") for piece in pieces]
    return (
        vault
        / "pages"
        / f"ns-{namespace}"
        / Path(*encoded[:-1])
        / f"{encoded[-1]}--page-{page_id}.md"
    )


def _revision_path(
    vault: Path,
    namespace: int,
    title: str,
    page_id: int,
    revision_id: int,
    collision_keys: set[str],
) -> Path:
    pieces = [piece for piece in title.replace("\\", "/").split("/") if piece]
    encoded = [quote(piece, safe="-_.()'") for piece in pieces]
    base = _title_path(vault, namespace, title)
    leaf = encoded[-1]
    if str(base).casefold() in collision_keys:
        leaf = f"{leaf}--page-{page_id}"
    return vault / "revisions" / f"ns-{namespace}" / Path(*encoded[:-1]) / leaf / f"rev-{revision_id}.md"


def _source_page_url(title: str) -> str:
    return SOURCE_URL + quote(title.replace(" ", "_"), safe="/_():,'")


def _page_metadata(page: dict[str, Any], retrieved_at: str) -> dict[str, Any]:
    return {
        "page_id": int(page["pageid"]),
        "namespace": int(page["ns"]),
        "title": str(page["title"]),
        "source_url": _source_page_url(str(page["title"])),
        "retrieved_at": retrieved_at,
    }


def _iter_inventory(namespaces: Iterable[int], *, delay: float) -> Iterable[dict[str, Any]]:
    for namespace in namespaces:
        continuation: dict[str, Any] = {}
        while True:
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "list": "allpages",
                "apnamespace": namespace,
                "aplimit": "500",
                "apfilterredir": "all",
            }
            params.update(continuation)
            payload = _api_request(params)
            retrieved_at = _utc_now()
            for page in payload.get("query", {}).get("allpages", []):
                yield _page_metadata(page, retrieved_at)
            if "continue" not in payload:
                break
            continuation = {
                "apcontinue": payload["continue"]["apcontinue"],
                "continue": payload["continue"]["continue"],
            }
            time.sleep(delay)


def _write_readme(vault: Path, namespaces: tuple[int, ...]) -> None:
    readme = f"""# League Wiki source vault

This folder is a local, source-preserving research mirror of the League of
Legends Wiki. It is not itself the mechanics engine.

- API source: `{API_URL}`
- Page source: `{SOURCE_URL}`
- Namespace selection: `{', '.join(str(value) for value in namespaces)}`
- Page files retain the latest retrieved wikitext and revisioned copies.
- Exact client numbers must be reconciled against patch-pinned Riot or
  CommunityDragon data before they enter a calculation.
- Wiki content is attributed to the League Wiki and may carry additional
  Riot Games asset terms; this local mirror is not a public redistribution
  surface.

Useful files:

- `catalog.jsonl`: every page title/page ID in the selected namespaces.
- `latest.jsonl`: latest successfully retrieved page revision metadata.
- `snapshot-manifest.json`: content hashes and retrieval summary.
- `pages/`: Obsidian-readable pages with source front matter.
- `revisions/`: immutable revision-keyed copies of retrieved wikitext.
"""
    _write_bytes_atomic(vault / "README.md", readme.encode("utf-8"))


def inventory(vault: Path, namespaces: tuple[int, ...], *, delay: float) -> dict[str, Any]:
    vault.mkdir(parents=True, exist_ok=True)
    _write_readme(vault, namespaces)
    catalog_path = vault / "catalog.jsonl"
    if catalog_path.exists():
        catalog_path.unlink()
    counts: Counter[int] = Counter()
    total = 0
    for page in _iter_inventory(namespaces, delay=delay):
        _append_jsonl(catalog_path, page)
        counts[page["namespace"]] += 1
        total += 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "inventory",
        "retrieved_at": _utc_now(),
        "source_api": API_URL,
        "namespaces": list(namespaces),
        "page_count": total,
        "pages_by_namespace": {str(key): value for key, value in sorted(counts.items())},
        "catalog_sha256": _sha256_bytes(catalog_path.read_bytes()),
    }
    _write_json_atomic(vault / "inventory-manifest.json", manifest)
    return manifest


def _iter_page_batches(
    namespaces: Iterable[int], *, batch_size: int, delay: float, start_title: str | None = None
) -> Iterable[list[dict[str, Any]]]:
    for namespace_index, namespace in enumerate(namespaces):
        continuation: dict[str, Any] = {}
        while True:
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "generator": "allpages",
                "gapnamespace": namespace,
                "gaplimit": batch_size,
                "gapfilterredir": "all",
                "prop": "info|revisions",
                "rvprop": "ids|timestamp|sha1|content",
                "rvslots": "main",
            }
            if start_title and namespace_index == 0 and not continuation:
                params["gapfrom"] = start_title
            params.update(continuation)
            payload = _api_request(params)
            pages = list(payload.get("query", {}).get("pages", []))
            pages.sort(key=lambda page: (int(page.get("ns", namespace)), str(page.get("title", ""))))
            if pages:
                yield pages
            if "continue" not in payload:
                break
            continuation = {
                "gapcontinue": payload["continue"]["gapcontinue"],
                "continue": payload["continue"]["continue"],
            }
            time.sleep(delay)


def _extract_revision(page: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    revisions = page.get("revisions") or []
    if not revisions:
        return None
    revision = revisions[0]
    content = (
        revision.get("slots", {})
        .get("main", {})
        .get("content")
    )
    if not isinstance(content, str):
        return None
    metadata = {
        "revision_id": int(revision["revid"]),
        "parent_revision_id": int(revision["parentid"])
        if revision.get("parentid") is not None
        else None,
        "revision_timestamp": str(revision["timestamp"]),
        "api_sha1": str(revision.get("sha1")) if revision.get("sha1") else None,
        "content_sha256": _sha256_bytes(content.encode("utf-8")),
        "content_model": str(page.get("contentmodel", "wikitext")),
        "content_format": "text/x-wiki",
    }
    return metadata, content


def _page_document(page: dict[str, Any], revision: dict[str, Any], content: str) -> bytes:
    metadata = _page_metadata(page, _utc_now())
    lines = [
        "---",
        f"schema_version: {json.dumps(SCHEMA_VERSION)}",
        f"source_kind: {json.dumps('league_wiki_wikitext')}",
        f"title: {json.dumps(metadata['title'], ensure_ascii=False)}",
        f"page_id: {metadata['page_id']}",
        f"namespace: {metadata['namespace']}",
        f"source_url: {json.dumps(metadata['source_url'])}",
        f"retrieved_at: {json.dumps(metadata['retrieved_at'])}",
        f"revision_id: {revision['revision_id']}",
        f"revision_timestamp: {json.dumps(revision['revision_timestamp'])}",
        f"parent_revision_id: {json.dumps(revision['parent_revision_id'])}",
        f"api_sha1: {json.dumps(revision['api_sha1'])}",
        f"content_sha256: {json.dumps(revision['content_sha256'])}",
        "license_note: \"CC BY-SA 3.0; additional terms may apply\"",
        "---",
        "",
        "<!-- Source-preserving wikitext begins below. Do not treat prose as executable mechanics without reconciliation. -->",
        content.rstrip("\n"),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def snapshot(
    vault: Path,
    namespaces: tuple[int, ...],
    *,
    delay: float,
    batch_size: int,
    max_pages: int | None,
    start_title: str | None = None,
) -> dict[str, Any]:
    vault.mkdir(parents=True, exist_ok=True)
    _write_readme(vault, namespaces)
    latest_path = vault / "latest.jsonl"
    errors_path = vault / "errors.jsonl"
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    if latest_path.exists():
        with latest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[(int(row["namespace"]), str(row["title"]))] = row

    processed = 0
    written = 0
    skipped = 0
    errors = 0
    counts: Counter[int] = Counter()
    collision_keys = _path_collision_keys(vault, namespaces)
    for pages in _iter_page_batches(
        namespaces, batch_size=batch_size, delay=delay, start_title=start_title
    ):
        for page in pages:
            if max_pages is not None and processed >= max_pages:
                break
            processed += 1
            namespace = int(page["ns"])
            title = str(page["title"])
            key = (namespace, title)
            extracted = _extract_revision(page)
            if extracted is None:
                errors += 1
                _append_jsonl(
                    errors_path,
                    {"namespace": namespace, "title": title, "error": "missing revision content"},
                )
                continue
            revision, content = extracted
            previous = latest.get(key)
            page_id = int(page["pageid"])
            current_path = _page_path(vault, namespace, title, page_id, collision_keys)
            current_relative = str(current_path.relative_to(vault))
            previous_is_intact = bool(
                previous
                and previous.get("revision_id") == revision["revision_id"]
                and previous.get("document_path") == current_relative
                and current_path.exists()
                and previous.get("document_sha256") == _sha256_bytes(current_path.read_bytes())
            )
            if previous_is_intact:
                skipped += 1
                counts[namespace] += 1
                continue
            document = _page_document(page, revision, content)
            revision_path = _revision_path(
                vault,
                namespace,
                title,
                page_id,
                revision["revision_id"],
                collision_keys,
            )
            _write_bytes_atomic(current_path, document)
            _write_bytes_atomic(revision_path, document)
            row = {
                **_page_metadata(page, _utc_now()),
                **revision,
                "document_sha256": _sha256_bytes(document),
                "document_path": str(current_path.relative_to(vault)),
                "revision_path": str(revision_path.relative_to(vault)),
            }
            latest[key] = row
            _append_jsonl(latest_path, row)
            written += 1
            counts[namespace] += 1
        if max_pages is not None and processed >= max_pages:
            break

    latest_rows = sorted(latest.values(), key=lambda row: (row["namespace"], row["title"]))
    latest_canonical = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in latest_rows
    ).encode("utf-8")
    _write_bytes_atomic(latest_path, latest_canonical)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "snapshot",
        "retrieved_at": _utc_now(),
        "source_api": API_URL,
        "namespaces": list(namespaces),
        "processed_pages": processed,
        "written_pages": written,
        "skipped_unchanged_pages": skipped,
        "error_pages": errors,
        "latest_page_count": len(latest_rows),
        "latest_by_namespace": {
            str(namespace): sum(1 for row in latest_rows if row["namespace"] == namespace)
            for namespace in namespaces
        },
        "latest_jsonl_sha256": _sha256_bytes(latest_canonical),
        "resumable": True,
        "complete": max_pages is None,
        "start_title": start_title,
    }
    _write_json_atomic(vault / "snapshot-manifest.json", manifest)
    return manifest


def repair(vault: Path, namespaces: tuple[int, ...], *, batch_size: int = 20) -> dict[str, Any]:
    """Repair only missing, colliding, or hash-mismatched text documents."""

    if batch_size < 1 or batch_size > 50:
        raise ValueError("batch_size must be in [1, 50]")
    latest_path = vault / "latest.jsonl"
    if not latest_path.exists():
        raise ValueError(f"missing latest checkpoint: {latest_path}")
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for line in latest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(int(row["namespace"]), str(row["title"]))] = row
    collision_keys = _path_collision_keys(vault, namespaces)
    targets: list[dict[str, Any]] = []
    for row in latest.values():
        namespace = int(row["namespace"])
        if namespace not in namespaces or namespace == 6:
            continue
        page_id = int(row["page_id"])
        current_path = _page_path(vault, namespace, str(row["title"]), page_id, collision_keys)
        current_relative = str(current_path.relative_to(vault))
        intact = bool(
            row.get("document_path") == current_relative
            and current_path.exists()
            and row.get("document_sha256") == _sha256_bytes(current_path.read_bytes())
        )
        if not intact:
            targets.append(row)
    repaired = 0
    errors = 0
    error_rows: list[dict[str, Any]] = []
    for start in range(0, len(targets), batch_size):
        batch = targets[start : start + batch_size]
        payload = _api_request(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "pageids": "|".join(str(row["page_id"]) for row in batch),
                "prop": "info|revisions",
                "rvprop": "ids|timestamp|sha1|content",
                "rvslots": "main",
            }
        )
        pages = payload.get("query", {}).get("pages", [])
        by_page_id = {int(page["pageid"]): page for page in pages if page.get("pageid") is not None}
        for row in batch:
            page_id = int(row["page_id"])
            page = by_page_id.get(page_id)
            if page is None:
                errors += 1
                error_rows.append({"page_id": page_id, "title": row["title"], "error": "page_missing_from_repair_response"})
                continue
            extracted = _extract_revision(page)
            if extracted is None:
                errors += 1
                error_rows.append({"page_id": page_id, "title": row["title"], "error": "missing_revision_content"})
                continue
            revision, content = extracted
            namespace = int(page["ns"])
            title = str(page["title"])
            current_path = _page_path(vault, namespace, title, page_id, collision_keys)
            revision_path = _revision_path(vault, namespace, title, page_id, revision["revision_id"], collision_keys)
            document = _page_document(page, revision, content)
            _write_bytes_atomic(current_path, document)
            _write_bytes_atomic(revision_path, document)
            latest[(namespace, title)] = {
                **_page_metadata(page, _utc_now()),
                **revision,
                "document_sha256": _sha256_bytes(document),
                "document_path": str(current_path.relative_to(vault)),
                "revision_path": str(revision_path.relative_to(vault)),
            }
            repaired += 1
    latest_rows = sorted(latest.values(), key=lambda row: (int(row["namespace"]), str(row["title"])))
    latest_canonical = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in latest_rows
    ).encode("utf-8")
    _write_bytes_atomic(latest_path, latest_canonical)
    if error_rows:
        for row in error_rows:
            _append_jsonl(vault / "errors.jsonl", row)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "repair",
        "retrieved_at": _utc_now(),
        "source_api": API_URL,
        "namespaces": list(namespaces),
        "target_count": len(targets),
        "repaired_pages": repaired,
        "error_pages": errors,
        "latest_page_count": len(latest_rows),
        "latest_jsonl_sha256": _sha256_bytes(latest_canonical),
        "resumable": True,
        "complete": False,
    }
    _write_json_atomic(vault / "snapshot-manifest.json", manifest)
    return manifest


def checkpoint(vault: Path) -> dict[str, Any]:
    """Reconcile an interrupted append log into an explicit partial manifest."""

    latest_path = vault / "latest.jsonl"
    if not latest_path.exists():
        raise ValueError(f"missing latest checkpoint: {latest_path}")
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    with latest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[(int(row["namespace"]), str(row["title"]))] = row
    rows = sorted(latest.values(), key=lambda row: (row["namespace"], row["title"]))
    canonical = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    _write_bytes_atomic(latest_path, canonical)
    counts = Counter(int(row["namespace"]) for row in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": "checkpoint",
        "retrieved_at": _utc_now(),
        "source_api": API_URL,
        "namespaces": sorted(counts),
        "latest_page_count": len(rows),
        "latest_by_namespace": {str(key): value for key, value in sorted(counts.items())},
        "latest_jsonl_sha256": _sha256_bytes(canonical),
        "resumable": True,
        "complete": False,
        "status": "partial_checkpoint_after_interrupted_snapshot",
    }
    _write_json_atomic(vault / "snapshot-manifest.json", manifest)
    return manifest


def validate(vault: Path, *, require_complete: bool = False) -> dict[str, Any]:
    """Validate inventory, checkpoint coverage, documents, and exact hashes.

    Namespace 6 (File) is intentionally metadata-only for the Obsidian vault,
    so its catalog count is required while its page-body count is not.
    """

    inventory_path = vault / "inventory-manifest.json"
    catalog_path = vault / "catalog.jsonl"
    latest_path = vault / "latest.jsonl"
    if not inventory_path.exists() or not catalog_path.exists() or not latest_path.exists():
        raise ValueError("vault validation requires inventory-manifest.json, catalog.jsonl, and latest.jsonl")
    inventory_manifest = json.loads(inventory_path.read_text(encoding="utf-8"))
    catalog_rows = [
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog_sha = _sha256_bytes(catalog_path.read_bytes())
    expected_catalog_sha = inventory_manifest.get("catalog_sha256")
    if expected_catalog_sha != catalog_sha:
        raise ValueError("catalog_sha256 does not match inventory manifest")
    catalog_keys = [(int(row["namespace"]), str(row["title"])) for row in catalog_rows]
    duplicate_catalog_keys = len(catalog_keys) - len(set(catalog_keys))
    latest_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for line in latest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["namespace"]), str(row["title"]))
        latest_by_key[key] = row
    catalog_counts = Counter(int(row["namespace"]) for row in catalog_rows)
    latest_counts = Counter(key[0] for key in latest_by_key)
    extra_latest_keys = sorted(set(latest_by_key) - set(catalog_keys))
    file_namespace = 6
    text_namespaces = sorted(
        namespace for namespace in catalog_counts if namespace != file_namespace
    )
    missing_text = sorted(
        f"{namespace}:{title}"
        for namespace, title in catalog_keys
        if namespace != file_namespace and (namespace, title) not in latest_by_key
    )
    invalid_documents: list[str] = []
    for key, row in latest_by_key.items():
        if key[0] == file_namespace:
            continue
        document_path = row.get("document_path")
        if not isinstance(document_path, str):
            invalid_documents.append(f"{key[0]}:{key[1]}:missing_document_path")
            continue
        path = vault / document_path
        if not path.exists() or not path.is_file():
            invalid_documents.append(f"{key[0]}:{key[1]}:missing_document")
            continue
        if row.get("document_sha256") != _sha256_bytes(path.read_bytes()):
            invalid_documents.append(f"{key[0]}:{key[1]}:document_hash_mismatch")
        if row.get("content_sha256") and not isinstance(row["content_sha256"], str):
            invalid_documents.append(f"{key[0]}:{key[1]}:invalid_content_hash")
    complete = not duplicate_catalog_keys and not missing_text and not invalid_documents
    result = {
        "schema_version": SCHEMA_VERSION,
        "operation": "validate",
        "vault": str(vault),
        "inventory_page_count": len(catalog_rows),
        "catalog_counts": {str(key): value for key, value in sorted(catalog_counts.items())},
        "latest_counts": {str(key): value for key, value in sorted(latest_counts.items())},
        "file_namespace_metadata_only": True,
        "text_namespaces": text_namespaces,
        "duplicate_catalog_keys": duplicate_catalog_keys,
        "missing_text_page_count": len(missing_text),
        "invalid_document_count": len(invalid_documents),
        "legacy_extra_latest_count": len(extra_latest_keys),
        "legacy_extra_latest_examples": [f"{namespace}:{title}" for namespace, title in extra_latest_keys[:20]],
        "missing_text_examples": missing_text[:20],
        "invalid_document_examples": invalid_documents[:20],
        "complete": complete,
        "completion_status": "complete_with_legacy_extras" if complete and extra_latest_keys else "complete" if complete else "incomplete",
        "catalog_sha256": catalog_sha,
    }
    if require_complete and not complete:
        raise ValueError(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def finalize(vault: Path) -> dict[str, Any]:
    """Mark the snapshot manifest complete only after validation passes."""

    result = validate(vault, require_complete=True)
    manifest_path = vault / "snapshot-manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            manifest = raw
    manifest.update(
        {
            "complete": True,
            "completion_status": result["completion_status"],
            "validated_at": _utc_now(),
            "validation_catalog_sha256": result["catalog_sha256"],
            "validation_latest_page_count": result["latest_counts"],
            "validation_legacy_extra_latest_count": result["legacy_extra_latest_count"],
        }
    )
    _write_json_atomic(manifest_path, manifest)
    result["snapshot_manifest_complete"] = True
    result["snapshot_manifest_path"] = str(manifest_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inventory", "snapshot", "repair", "checkpoint", "validate", "finalize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
        if command in {"checkpoint", "validate", "finalize"}:
            if command == "validate":
                subparser.add_argument("--require-complete", action="store_true")
            continue
        subparser.add_argument("--namespace", help="comma-separated namespace IDs")
        subparser.add_argument("--all-content-namespaces", action="store_true")
        subparser.add_argument("--delay", type=float, default=0.25)
        if command == "snapshot":
            subparser.add_argument("--batch-size", type=int, default=50)
            subparser.add_argument("--max-pages", type=int)
            subparser.add_argument("--start-title")
        elif command == "repair":
            subparser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        if args.command == "checkpoint":
            result = checkpoint(args.vault)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "validate":
            result = validate(args.vault, require_complete=args.require_complete)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "finalize":
            result = finalize(args.vault)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        namespaces = _parse_namespaces(args.namespace, args.all_content_namespaces)
        if args.command == "inventory":
            result = inventory(args.vault, namespaces, delay=max(0.0, args.delay))
        elif args.command == "snapshot":
            if args.batch_size < 1 or args.batch_size > 50:
                raise ValueError("--batch-size must be in [1, 50]")
            result = snapshot(
                args.vault,
                namespaces,
                delay=max(0.0, args.delay),
                batch_size=args.batch_size,
                max_pages=args.max_pages,
                start_title=args.start_title,
            )
        else:
            result = repair(args.vault, namespaces, batch_size=args.batch_size)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
