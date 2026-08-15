from __future__ import annotations

import pytest

from lol_kills.knowledge.cdragon_patch_packet import CDragonClient
from lol_kills.v2.patch_identity import (
    PatchIdentityError,
    client_patch,
    corrected_oe_patch_token,
    public_patch,
)


def test_public_and_client_patch_namespaces_are_explicit() -> None:
    assert public_patch("26.15") == "26.15"
    assert public_patch("16.15") == "26.15"
    assert public_patch("16.16.1") == "26.16"
    assert client_patch("26.01") == "16.1"
    assert client_patch("26.16") == "16.16"


def test_2026_source_tokens_keep_exact_public_patch_mapping() -> None:
    assert client_patch("16.15") == "16.15"
    assert public_patch("16.15") == "26.15"
    assert client_patch("16.16") == "16.16"
    assert public_patch("16.16") == "26.16"


def test_unsupported_patch_is_rejected_without_guessing() -> None:
    with pytest.raises(PatchIdentityError):
        public_patch("16.15rc1")
    with pytest.raises(PatchIdentityError):
        public_patch("14.24")


def test_cdragon_client_keeps_26_15_public_label() -> None:
    client = CDragonClient("16.15")
    assert client.patch == "26.15"
    assert client.source_patch == "16.15"


def test_oe_26_16_correction_requires_realm_evidence() -> None:
    unchanged, no_rule = corrected_oe_patch_token("16.15", "2026-08-13T10:00:00Z")
    assert unchanged == "16.15"
    assert no_rule is None

    corrected, rule = corrected_oe_patch_token(
        "16.15",
        "2026-08-13T10:00:00Z",
        realm_kind="live",
    )
    assert corrected == "16.16"
    assert rule is not None
    unchanged, no_rule = corrected_oe_patch_token("16.15", "2026-08-11T17:59:59Z")
    assert unchanged == "16.15"
    assert no_rule is None

    unchanged, no_rule = corrected_oe_patch_token(
        "16.15",
        "2026-08-13T10:00:00Z",
        realm_kind="tournament",
    )
    assert unchanged == "16.15"
    assert no_rule is None


def test_oe_26_16_correction_does_not_replace_an_exact_newer_token() -> None:
    corrected, rule = corrected_oe_patch_token("16.16", "2026-08-13T10:00:00Z")
    assert corrected == "16.16"
    assert rule is None
