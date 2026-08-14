from __future__ import annotations

import pytest

from lol_kills.knowledge.cdragon_patch_packet import CDragonClient
from lol_kills.v2.patch_identity import (
    PatchIdentityError,
    client_patch,
    public_patch,
)


def test_public_and_client_patch_namespaces_are_explicit() -> None:
    assert public_patch("26.15") == "26.15"
    assert public_patch("16.15") == "26.15"
    assert public_patch("16.16.1") == "26.16"
    assert client_patch("26.01") == "16.1"
    assert client_patch("26.16") == "16.16"


def test_unsupported_patch_is_rejected_without_guessing() -> None:
    with pytest.raises(PatchIdentityError):
        public_patch("16.15rc1")
    with pytest.raises(PatchIdentityError):
        public_patch("14.24")


def test_cdragon_client_keeps_26_15_public_label() -> None:
    client = CDragonClient("16.15")
    assert client.patch == "26.15"
    assert client.source_patch == "16.15"

