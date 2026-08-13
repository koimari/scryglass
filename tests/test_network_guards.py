from __future__ import annotations

import pytest

from lol_kills.etl import oe_ingest
from lol_kills.net import NetworkTargetError, require_https_url


def test_require_https_url_accepts_exact_and_allowed_subdomain() -> None:
    assert require_https_url(
        "https://lol.fandom.com/api.php?x=1", hosts={"lol.fandom.com"}
    ).startswith("https://lol.fandom.com/")
    assert require_https_url(
        "https://example.public.blob.vercel-storage.com/packs/a.json",
        hosts={"blob.vercel-storage.com"},
        allow_subdomains=True,
    ).startswith("https://example.public.blob.vercel-storage.com/")


@pytest.mark.parametrize(
    "url",
    (
        "http://lol.fandom.com/api.php",
        "file:///etc/passwd",
        "https://lol.fandom.com.evil.test/api.php",
        "https://user:pass@lol.fandom.com/api.php",
        "https://lol.fandom.com:8443/api.php",
    ),
)
def test_require_https_url_rejects_unsafe_targets(url: str) -> None:
    with pytest.raises(NetworkTargetError):
        require_https_url(url, hosts={"lol.fandom.com"})


def test_oe_metadata_probe_rejects_non_google_target() -> None:
    with pytest.raises(NetworkTargetError):
        oe_ingest._remote_file_signature("file:///etc/passwd")
