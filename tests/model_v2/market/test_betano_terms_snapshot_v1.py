from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.data.common import sha256_canonical_object
from lol_kills.v2.market import betano_terms_snapshot_v1 as terms
from lol_kills.v2.market.betano_terms_snapshot_registry_v1 import (
    REGISTERED_SNAPSHOT_ARTIFACT_SHA256,
    REGISTERED_SNAPSHOT_RAW_SHA256,
    validate_registered_betano_terms_snapshot_v1,
)


ROOT = Path(__file__).resolve().parents[3]
START = datetime(2026, 8, 2, 2, 0, tzinfo=timezone.utc)


def article(article_id: int) -> bytes:
    if article_id == terms.LOL_ARTICLE_ID:
        title = "Como se joga o E-sport League of Legends (LoL)?"
        body = "<strong>Vencedor do Mapa:</strong> quem destruir a base inimiga."
    else:
        title = "O que acontece se o jogo em que apostei for adiado ou cancelado?"
        body = (
            '<strong>princípio do "Mesmo Dia"</strong> '
            "a aposta continua válida; a aposta será anulada e o valor será "
            "reembolsado; todas as apostas pendentes relacionadas a esse evento "
            "serão anuladas e reembolsadas"
        )
    return json.dumps(
        {
            "article": {
                "id": article_id,
                "html_url": f"https://support.betano.bet.br/hc/pt-br/articles/{article_id}",
                "title": title,
                "source_locale": "pt-br",
                "locale": "pt-br",
                "draft": False,
                "outdated": False,
                "updated_at": "2026-08-01T23:20:16Z",
                "edited_at": "2026-03-17T18:40:47Z",
                "body": body,
            }
        },
        ensure_ascii=False,
    ).encode("utf-8")


class Response:
    def __init__(self, raw: bytes, url: str):
        self.raw = raw
        self.url = url
        self.headers = {"Content-Type": "application/json; charset=utf-8"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int = -1) -> bytes:
        return self.raw if limit < 0 else self.raw[:limit]

    def getcode(self) -> int:
        return 200

    def geturl(self) -> str:
        return self.url


def build() -> dict:
    moments = iter(START + timedelta(milliseconds=index) for index in range(5))

    def opener(request, *, timeout: int):
        assert timeout == 20
        article_id = int(request.full_url.rsplit("/", 1)[-1].split(".", 1)[0])
        return Response(article(article_id), request.full_url)

    return terms.capture_betano_terms_snapshot_v1(
        root=ROOT,
        opener=opener,
        clock=lambda: next(moments),
    )


def rehash(payload: dict) -> None:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    payload["artifact_sha256"] = sha256_canonical_object(unsigned)


def test_public_snapshot_is_exact_incomplete_and_non_authorizing() -> None:
    payload = build()

    assert payload["coverage"]["complete_bookmaker_terms_snapshot"] is False
    assert payload["coverage"]["independent_alignment_review_present"] is False
    assert all(value is False for value in payload["authority"].values())
    assert all(value is None for value in payload["decision_outputs"].values())
    assert [item["article_id"] for item in payload["sources"]] == [
        terms.LOL_ARTICLE_ID,
        terms.CANCELLATION_ARTICLE_ID,
    ]
    assert all(item["raw_sha256"] for item in payload["sources"])
    assert payload["clock_attestation"]["user_supplied_timestamp_allowed"] is False


def test_rehashed_snapshot_cannot_claim_complete_terms() -> None:
    payload = deepcopy(build())
    payload["coverage"]["complete_bookmaker_terms_snapshot"] = True
    rehash(payload)

    with pytest.raises(terms.BetanoTermsSnapshotError, match="overstated"):
        terms.validate_betano_terms_snapshot_v1(payload, root=ROOT)


def test_raw_article_tampering_is_detected() -> None:
    payload = deepcopy(build())
    payload["sources"][0]["raw_sha256"] = "0" * 64
    rehash(payload)

    with pytest.raises(terms.BetanoTermsSnapshotError, match="raw bytes changed"):
        terms.validate_betano_terms_snapshot_v1(payload, root=ROOT)


def test_writer_is_no_clobber(tmp_path: Path) -> None:
    payload = build()
    path = tmp_path / "terms.json"
    assert len(terms.write_no_clobber(path, payload)) == 64
    with pytest.raises(FileExistsError, match="refusing to replace"):
        terms.write_no_clobber(path, payload)


def test_registered_public_snapshot_replays_but_stays_incomplete() -> None:
    checked = validate_registered_betano_terms_snapshot_v1(root=ROOT)
    raw = (ROOT / terms.DEFAULT_OUTPUT).read_bytes()

    assert checked["artifact_sha256"] == REGISTERED_SNAPSHOT_ARTIFACT_SHA256
    assert hashlib.sha256(raw).hexdigest() == REGISTERED_SNAPSHOT_RAW_SHA256
    assert checked["coverage"]["complete_bookmaker_terms_snapshot"] is False
    assert checked["coverage"]["independent_alignment_review_present"] is False
