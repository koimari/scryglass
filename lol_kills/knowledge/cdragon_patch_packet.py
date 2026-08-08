"""Capture a patch-pinned CommunityDragon mechanics packet.

This is a raw-source and normalization step.  It does not claim that every
client calculation is already executable: the packet records the original
formula graph and marks any missing champion bin as incomplete.  The later
mechanics kernel must implement and micro-test individual calculation parts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCHEMA_VERSION = "scryglass:cdragon-patch-packet:v1"
SOURCE_ROOT = "https://raw.communitydragon.org"
USER_AGENT = "Scryglass mechanics research source capture/0.1 (+local research)"
DEFAULT_OUTPUT_ROOT = Path("data/lol/knowledge/patch-packets/cdragon")
PLUGIN_ROOT = "plugins/rcp-be-lol-game-data/global/default/v1"
CHAMPION_SUMMARY_PATH = f"{PLUGIN_ROOT}/champion-summary.json"
ITEMS_PATH = f"{PLUGIN_ROOT}/items.json"
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_json(path: Path, value: Any) -> None:
    _write_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _client_patch_for(patch: str) -> str:
    """Map a public 2026 patch label to its exact client namespace.

    Riot's public 2026 patch notes use the ``26.x`` season label while the
    client archive uses the game's major version, ``16.x``.  The minor patch
    is preserved exactly; this is a namespace mapping, not a nearest-patch
    fallback. Unknown labels remain unchanged and therefore fail closed if
    the source does not exist.
    """

    match = re.fullmatch(r"26\.(\d{1,2})", patch.strip("/"))
    return f"16.{int(match.group(1))}" if match else patch.strip("/")


class CDragonClient:
    def __init__(
        self,
        patch: str,
        *,
        source_patch: str | None = None,
        delay: float = 0.15,
        timeout: float = 60.0,
    ):
        self.patch = patch.strip("/")
        self.source_patch = (source_patch or _client_patch_for(self.patch)).strip("/")
        self.delay = max(0.0, delay)
        self.timeout = timeout

    def get(self, relative_path: str, *, retries: int = 4) -> tuple[bytes, str]:
        url = f"{SOURCE_ROOT}/{self.source_patch}/{relative_path}"
        request = Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                if self.delay:
                    time.sleep(self.delay)
                return body, url
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 == retries:
                    break
                time.sleep(float(attempt + 1))
        raise RuntimeError(f"CommunityDragon request failed for {url}: {last_error}") from last_error

    def probe(self, relative_path: str) -> dict[str, Any]:
        """Probe one exact-patch path without substituting another patch."""

        url = f"{SOURCE_ROOT}/{self.source_patch}/{relative_path}"
        request = Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                status = int(getattr(response, "status", 200))
                content_type = str(response.headers.get("Content-Type", ""))
            if self.delay:
                time.sleep(self.delay)
            return {
                "url": url,
                "status_code": status,
                "available": status == 200,
                "sha256": _sha256(body),
                "bytes": len(body),
                "content_type": content_type,
            }
        except HTTPError as exc:
            body = exc.read()
            return {
                "url": url,
                "status_code": int(exc.code),
                "available": False,
                "sha256": _sha256(body),
                "bytes": len(body),
                "error": str(exc),
            }
        except (URLError, TimeoutError) as exc:
            return {
                "url": url,
                "status_code": None,
                "available": False,
                "error": str(exc),
            }


def _safe_alias(alias: str) -> str:
    candidate = alias.strip().lower()
    if not SAFE_NAME_RE.fullmatch(candidate):
        raise ValueError(f"unsupported champion alias for client path: {alias!r}")
    return candidate


def _character_record(bin_payload: dict[str, Any]) -> dict[str, Any] | None:
    for key, value in bin_payload.items():
        if (
            isinstance(value, dict)
            and key.endswith("/CharacterRecords/Root")
            and value.get("__type") == "CharacterRecord"
        ):
            return value
    return None


def _numeric_base_value(value: Any) -> float | None:
    if not isinstance(value, dict) or "baseValue" not in value:
        return None
    number = value["baseValue"]
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        return None
    return float(number)


def _extract_stats(record: dict[str, Any] | None) -> dict[str, float | None]:
    if record is None:
        return {}
    fields = {
        "base_health": "baseHPModifiable",
        "health_per_level": "hpPerLevelModifiable",
        "base_health_regen": "baseStaticHPRegenModifiable",
        "health_regen_per_level": "hpRegenPerLevelModifiable",
        "base_attack_damage": "baseDamageModifiable",
        "attack_damage_per_level": "damagePerLevelModifiable",
        "base_armor": "baseArmorModifiable",
        "armor_per_level": "armorPerLevelModifiable",
        "base_magic_resist": "baseMR",
        "base_move_speed": "baseMoveSpeedModifiable",
        "attack_range": "attackRangeModifiable",
        "attack_speed": "attackSpeedModifiable",
        "attack_speed_per_level": "attackSpeedPerLevelModifiable",
    }
    out: dict[str, float] = {}
    for name, key in fields.items():
        number = _numeric_base_value(record.get(key))
        if number is not None:
            out[name] = number
    return out


def _spell_data_values(spell: dict[str, Any]) -> list[dict[str, Any]]:
    values = spell.get("mSpell", {}).get("DataValues", [])
    if not isinstance(values, list):
        return []
    return [
        {"name": item.get("name"), "values": item.get("values")}
        for item in values
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def _extract_spell(bin_payload: dict[str, Any], path: str) -> dict[str, Any] | None:
    value = bin_payload.get(path)
    if not isinstance(value, dict) or value.get("__type") != "SpellObject":
        return None
    spell = value.get("mSpell")
    if not isinstance(spell, dict):
        return None
    return {
        "path": path,
        "object_name": value.get("ObjectName"),
        "script_name": value.get("mScriptName"),
        "spell_tags": spell.get("mSpellTags", []),
        "data_values": _spell_data_values(value),
        "spell_calculations": spell.get("mSpellCalculations", {}),
        "cooldown_time": spell.get("cooldownTime"),
        "cast_range": spell.get("castRange"),
        "cast_radius": spell.get("castRadius"),
        "targeting_type": spell.get("mTargetingTypeData"),
        "client_targeters": spell.get("mClientData", {}).get("mTargeterDefinitions", []),
        "bot_data": value.get("BotData"),
    }


def _extract_mechanics(bin_payload: dict[str, Any]) -> dict[str, Any]:
    record = _character_record(bin_payload)
    paths: list[str] = []
    if record:
        for ability in record.get("mAbilities", []):
            if not isinstance(ability, str):
                continue
            ability_object = bin_payload.get(ability)
            if not isinstance(ability_object, dict):
                continue
            children = ability_object.get("mChildSpells", [])
            for child in children:
                if isinstance(child, str) and child in bin_payload:
                    paths.append(child)
    spells = []
    for path in sorted(set(paths)):
        extracted = _extract_spell(bin_payload, path)
        if extracted is not None:
            spells.append(extracted)
    return {
        "character_name": record.get("mCharacterName") if record else None,
        "champion_id": record.get("characterToolData", {}).get("championId") if record else None,
        "stats": _extract_stats(record),
        "spell_names": record.get("spellNames", []) if record else [],
        "passive_spell": record.get("mCharacterPassiveSpell") if record else None,
        "spells": spells,
        "raw_record_present": record is not None,
        "formula_semantics_status": "raw_formula_graph_preserved",
        "execution_status": "not_yet_implemented",
    }


def _requested_champions(
    summary: list[dict[str, Any]], requested: set[str] | None, max_champions: int | None
) -> list[dict[str, Any]]:
    visible = [
        row
        for row in summary
        if isinstance(row, dict) and isinstance(row.get("id"), int) and row["id"] > 0
    ]
    if requested:
        requested_lower = {value.casefold() for value in requested}
        visible = [
            row
            for row in visible
            if str(row.get("name", "")).casefold() in requested_lower
            or str(row.get("alias", "")).casefold() in requested_lower
            or str(row.get("id")) in requested_lower
        ]
    visible.sort(key=lambda row: (int(row["id"]), str(row.get("name", ""))))
    if max_champions is not None:
        visible = visible[:max_champions]
    return visible


def capture(
    patch: str,
    output_root: Path,
    *,
    source_patch: str | None = None,
    requested_champions: set[str] | None = None,
    max_champions: int | None = None,
    include_bins: bool = True,
    delay: float = 0.15,
) -> dict[str, Any]:
    output = output_root / patch
    output.mkdir(parents=True, exist_ok=True)
    client = CDragonClient(patch, source_patch=source_patch, delay=delay)
    files: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def fetch_json(relative_path: str, local_path: Path) -> Any | None:
        try:
            body, url = client.get(relative_path)
            _write_atomic(output / local_path, body)
            files.append(
                {
                    "path": str(local_path),
                    "url": url,
                    "sha256": _sha256(body),
                    "bytes": len(body),
                }
            )
            return json.loads(body)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": relative_path, "error": str(exc)})
            return None

    summary_value = fetch_json(CHAMPION_SUMMARY_PATH, Path("raw/champion-summary.json"))
    if not isinstance(summary_value, list):
        raise RuntimeError("champion summary was not retrieved")
    item_value = fetch_json(ITEMS_PATH, Path("raw/items.json"))
    champions = _requested_champions(summary_value, requested_champions, max_champions)
    normalized: list[dict[str, Any]] = []
    for summary in champions:
        champion_id = int(summary["id"])
        name = str(summary.get("name", champion_id))
        alias = str(summary.get("alias", name))
        champion_path = Path(f"raw/champions/{champion_id}.json")
        champion_value = fetch_json(
            f"{PLUGIN_ROOT}/champions/{champion_id}.json", champion_path
        )
        if not isinstance(champion_value, dict):
            normalized.append(
                {
                    "id": champion_id,
                    "name": name,
                    "alias": alias,
                    "status": "missing_champion_json",
                }
            )
            continue
        mechanics: dict[str, Any] | None = None
        bin_path: str | None = None
        bin_sha256: str | None = None
        if include_bins:
            safe_alias = _safe_alias(str(champion_value.get("alias", alias)))
            relative_bin = f"game/data/characters/{safe_alias}/{safe_alias}.bin.json"
            local_bin = Path(f"raw/{relative_bin}")
            bin_value = fetch_json(relative_bin, local_bin)
            if isinstance(bin_value, dict):
                mechanics = _extract_mechanics(bin_value)
                bin_path = str(local_bin)
                bin_sha256 = next(
                    (
                        row["sha256"]
                        for row in files
                        if row["path"] == str(local_bin)
                    ),
                    None,
                )
        normalized.append(
            {
                "id": champion_id,
                "name": str(champion_value.get("name", name)),
                "alias": str(champion_value.get("alias", alias)),
                "roles": champion_value.get("roles", []),
                "tactical_info": champion_value.get("tacticalInfo"),
                "playstyle_info": champion_value.get("playstyleInfo"),
                "champion_json_path": str(champion_path),
                "bin_json_path": bin_path,
                "bin_sha256": bin_sha256,
                "mechanics": mechanics,
                "status": "raw_and_normalized" if mechanics else "client_summary_only",
            }
        )

    normalized_payload = {
        "schema_version": SCHEMA_VERSION,
        "patch": patch,
        "client_patch": client.source_patch,
        "source": "CommunityDragon",
        "source_root": f"{SOURCE_ROOT}/{client.source_patch}/",
        "retrieved_at": _utc_now(),
        "champions": normalized,
        "items_count": len(item_value) if isinstance(item_value, list) else None,
        "normalization": {
            "stats": "CharacterRecords/Root modifiable base values",
            "spells": "AbilityObject child SpellObjects with DataValues and raw mSpellCalculations",
            "execution_status": "not_yet_implemented",
        },
    }
    _write_json(output / "mechanics-index.json", normalized_payload)
    manifest_unsigned = {
        "schema_version": SCHEMA_VERSION,
        "patch": patch,
        "client_patch": client.source_patch,
        "source": "CommunityDragon",
        "source_root": f"{SOURCE_ROOT}/{client.source_patch}/",
        "retrieved_at": _utc_now(),
        "requested_champion_count": len(champions),
        "captured_file_count": len(files),
        "files": files,
        "errors": errors,
        "exact_patch_source": True,
        "mechanics_execution_status": "not_yet_implemented",
    }
    manifest = {
        **manifest_unsigned,
        "manifest_sha256": _sha256(_json_bytes(manifest_unsigned)),
    }
    _write_json(output / "manifest.json", manifest)
    return {
        "patch": patch,
        "client_patch": client.source_patch,
        "output": str(output),
        "champions": len(champions),
        "files": len(files),
        "errors": len(errors),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def capture_patch_matrix(
    patches: Iterable[str],
    output_root: Path,
    *,
    client_patch_map: Mapping[str, str] | None = None,
    probe_only: bool = False,
    delay: float = 0.15,
    workers: int = 1,
) -> dict[str, Any]:
    """Probe/capture a patch list without nearest-patch fallback.

    Every requested patch gets a durable ``probe.json``. A patch is eligible
    for capture only when both the champion summary and item payload resolve
    at the exact client namespace recorded for that public patch. Missing
    patches remain explicit blocked records so a later resume can fill them
    without rewriting successful packets.
    """

    normalized = tuple(
        sorted({str(patch).strip("/") for patch in patches if str(patch).strip("/")})
    )
    if not normalized:
        raise ValueError("at least one patch is required")
    if workers < 1:
        raise ValueError("workers must be positive")

    def capture_one(patch: str) -> dict[str, Any]:
        source_patch = str(
            (client_patch_map or {}).get(patch) or _client_patch_for(patch)
        ).strip("/")
        patch_dir = output_root / patch
        probe_path = patch_dir / "probe.json"
        existing: dict[str, Any] | None = None
        if probe_path.exists():
            try:
                candidate = json.loads(probe_path.read_text(encoding="utf-8"))
                if (
                    isinstance(candidate, dict)
                    and candidate.get("patch") == patch
                    and candidate.get("client_patch") == source_patch
                ):
                    existing = candidate
            except (OSError, json.JSONDecodeError):
                existing = None
        if existing is not None and existing.get("status") in {
            "blocked",
            "captured",
            "captured_authority_blocked",
        }:
            return existing

        client = CDragonClient(patch, source_patch=source_patch, delay=delay)
        summary_probe = client.probe(CHAMPION_SUMMARY_PATH)
        item_probe = client.probe(ITEMS_PATH)
        retrieved_at = _utc_now()
        exact_source = bool(
            summary_probe.get("available") and item_probe.get("available")
        )
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "patch": patch,
            "client_patch": source_patch,
            "retrieved_at": retrieved_at,
            "source_root": f"{SOURCE_ROOT}/{source_patch}/",
            "probes": {
                "champion_summary": summary_probe,
                "items": item_probe,
            },
            "exact_patch_source": exact_source,
            "status": (
                "available_probe_only"
                if exact_source and probe_only
                else "blocked"
                if not exact_source
                else "available"
            ),
        }
        if exact_source and not probe_only:
            try:
                result = capture(
                    patch,
                    output_root,
                    source_patch=source_patch,
                    include_bins=True,
                    delay=delay,
                )
                row["capture"] = result
                row["status"] = "captured"
                try:
                    from lol_kills.knowledge.patch_authority import (
                        build_cdragon_patch_packet,
                    )

                    packet = build_cdragon_patch_packet(
                        output_root / patch / "mechanics-index.json",
                        expected_patch=patch,
                        manifest_path=output_root / patch / "manifest.json",
                    )
                    packet_path = output_root / patch / "authority-packet.json"
                    row["authority_packet_path"] = str(packet_path)
                    row["authority_packet_sha256"] = packet.write(packet_path)
                    row["executable_cell_count"] = len(packet.executable_cells)
                except (OSError, ValueError, RuntimeError) as exc:
                    row["status"] = "captured_authority_blocked"
                    row["authority_error"] = str(exc)
            except (OSError, RuntimeError, ValueError) as exc:
                row["status"] = "capture_failed"
                row["capture_error"] = str(exc)
        row["row_sha256"] = _sha256(_json_bytes(row))
        _write_json(probe_path, row)
        return row

    if workers == 1:
        rows = [capture_one(patch) for patch in normalized]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(capture_one, normalized))
    rows.sort(key=lambda row: str(row.get("patch", "")))

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "matrix_kind": "communitydragon_2026_patch_matrix",
        "retrieved_at": _utc_now(),
        "patches": rows,
        "claim_ceiling": {
            "exact_mechanics": any(
                row.get("status") == "captured" and row.get("executable_cell_count", 0)
                for row in rows
            ),
            "full_game_emulation": False,
            "prediction": False,
            "publication": False,
        },
    }
    manifest = {**unsigned, "manifest_sha256": _sha256(_json_bytes(unsigned))}
    _write_json(output_root / "matrix-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", action="append")
    parser.add_argument(
        "--probe-matrix-2026",
        action="store_true",
        help="probe/capture 26.01 through 26.15 as exact paths",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--client-patch",
        help="exact CommunityDragon client namespace for a single public patch label",
    )
    parser.add_argument("--champion", action="append", dest="champions")
    parser.add_argument("--max-champions", type=int)
    parser.add_argument("--no-bins", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args(argv)
    if args.probe_matrix_2026:
        patches = args.patch or [f"26.{index:02d}" for index in range(1, 16)]
        try:
            result = capture_patch_matrix(
                patches,
                args.output_root,
                probe_only=args.probe_only,
                delay=args.delay,
                workers=args.workers,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.patch or len(args.patch) != 1:
        parser.error("--patch is required exactly once unless --probe-matrix-2026 is used")
    if args.max_champions is not None and args.max_champions < 1:
        parser.error("--max-champions must be positive")
    try:
        result = capture(
            args.patch[0],
            args.output_root,
            source_patch=args.client_patch,
            requested_champions=set(args.champions or []),
            max_champions=args.max_champions,
            include_bins=not args.no_bins,
            delay=args.delay,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
