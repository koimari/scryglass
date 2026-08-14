from __future__ import annotations

import json
from pathlib import Path

from lol_kills.v2.patch_identity import canonical_patch
from lol_kills.v2.champions.atoms.consume import AtomBridge


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json"
BRIDGE_PATH = ROOT / "data/lol/v2/champions/lcc-atom-bridge-v1.json"


def test_26_16_receipt_binds_source_and_bridge() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    bridge = json.loads(BRIDGE_PATH.read_text(encoding="utf-8"))
    validated_bridge = AtomBridge.load(BRIDGE_PATH)
    identity = canonical_patch(receipt["public_patch"])

    assert receipt["schema_version"] == "scryglass:lcc-atom-refresh-receipt:v1"
    assert identity.public_patch == receipt["public_patch"] == "26.16"
    assert identity.client_patch == receipt["client_patch"] == "16.16"
    assert receipt["source_root"].endswith("/16.16/")
    assert receipt["base_champion_count"] == len(bridge["champions"]) == 173
    assert validated_bridge.artifact_sha256 == receipt["atom_bridge_artifact_sha256"]
    provenance = bridge["provenance"]
    assert provenance["data_patch"] == receipt["public_patch"]
    assert provenance["client_patch"] == receipt["client_patch"]
    assert provenance["source_version"] == receipt["source_version"]
    assert provenance["lcc_commit"] == receipt["lcc_commit"]


def test_26_16_receipt_keeps_unreviewed_claims_closed() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    ceiling = receipt["claim_ceiling"]

    assert ceiling["exact_source"] is True
    assert ceiling["full_game_emulation"] is False
    assert ceiling["prediction"] is False
    assert ceiling["publication"] is False
    assert receipt["raw_packet_retained"] is False
    assert receipt["raw_packet_path_ignored"] is True
