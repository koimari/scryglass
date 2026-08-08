from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message

import pytest

from lol_kills.export.blob_retention import BlobIdentity
from lol_kills.export.vercel_blob_transport import (
    VercelBlobAmbiguousMutationError,
    VercelBlobDeadlineError,
    VercelBlobTransport,
    VercelBlobTransportError,
)


TOKEN = "vercel_blob_rw_store-a_secret-token-never-print"
STORE = "store-a"


class Response:
    def __init__(
        self,
        body=b"{}",
        *,
        url=None,
        headers=None,
        on_read=None,
        read_error=None,
    ):
        self._body = body
        self._url = url
        self._on_read = on_read
        self._read_error = read_error
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value
        self.closed = False

    def read(self, *_args):
        if self._on_read is not None:
            self._on_read()
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def geturl(self):
        return self._url

    def close(self):
        self.closed = True


class Opener:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        result = self.scripted.pop(0)
        if isinstance(result, urllib.error.HTTPError) and result.url is None:
            result.url = request.full_url
        if isinstance(result, Exception):
            raise result
        result._url = result._url or request.full_url
        return result


def page(blobs=(), *, cursor=None, more=False):
    body = {"blobs": list(blobs), "hasMore": more}
    if cursor is not None:
        body["cursor"] = cursor
    return Response(json.dumps(body).encode())


def blob(path="packs/a file.json", size=3, etag="etag-a"):
    return {"pathname": path, "size": size, "etag": etag}


def transport(scripted, *, clock=lambda: 100):
    return VercelBlobTransport(TOKEN, STORE, opener=Opener(scripted), clock=clock)


def http_error(status, code=None, *, url=None, payload=None, fp=None):
    if payload is None:
        payload = (
            json.dumps({"error": {"code": code}}).encode()
            if code is not None
            else b""
        )
    return urllib.error.HTTPError(
        url,
        status,
        "x",
        Message(),
        io.BytesIO(payload) if fp is None else fp,
    )


def request_headers(request):
    return {key.lower(): value for key, value in request.header_items()}


def test_constructor_requires_explicit_credentials_and_redacts_token():
    with pytest.raises(ValueError):
        VercelBlobTransport("", STORE)
    with pytest.raises(ValueError):
        VercelBlobTransport("secret-token-never-print", STORE)
    with pytest.raises(ValueError):
        VercelBlobTransport(TOKEN, "")
    with pytest.raises(ValueError, match="does not match"):
        VercelBlobTransport("vercel_blob_rw_other_secret", STORE)
    value = VercelBlobTransport(TOKEN, "store_store-a")
    assert value.store_id == STORE
    assert TOKEN not in repr(value)
    assert "<redacted>" in repr(value)


@pytest.mark.parametrize(
    "store_id",
    [
        " evil",
        "evil ",
        "evil.example/steal",
        "evil.example?x=1",
        "evil.example#fragment",
        "-leading",
        "trailing-",
        "UPPER",
        "évil",
        "a" * 64,
    ],
)
def test_constructor_rejects_noncanonical_or_hostile_store_ids(store_id):
    token = f"vercel_blob_rw_{store_id}_secret"
    with pytest.raises(ValueError):
        VercelBlobTransport(token, store_id)


def test_list_page_uses_expanded_pagination_and_explicit_store_header():
    t = transport([page([blob()], cursor="next+cursor", more=True)])
    result = t.list_page(STORE, cursor="old cursor", limit=99, deadline_epoch=101)
    request, timeout = t._opener.calls[0]
    assert request.full_url == "https://vercel.com/api/blob?limit=99&mode=expanded&cursor=old+cursor"
    headers = request_headers(request)
    assert headers["authorization"] == f"Bearer {TOKEN}"
    assert headers["x-api-version"] == "12"
    assert headers["x-vercel-blob-store-id"] == STORE
    assert timeout == 1.0
    assert result == {"storeId": STORE, "blobs": [blob()], "hasMore": True, "cursor": "next+cursor"}


@pytest.mark.parametrize("payload", [b"not-json", b"[]", b'{"blobs":[],"hasMore":"no"}', b'{"blobs":[{}],"hasMore":false}', b'{"blobs":[],"hasMore":true}'])
def test_list_schema_is_fail_closed(payload):
    with pytest.raises(VercelBlobTransportError):
        transport([Response(payload)]).list_page(STORE, cursor=None, limit=1, deadline_epoch=101)


def test_list_rejects_noncanonical_blob_metadata():
    with pytest.raises(VercelBlobTransportError, match="metadata"):
        transport([page([blob("../escape")])]).list_page(
            STORE, cursor=None, limit=1, deadline_epoch=101
        )


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_list_never_converts_auth_rate_or_server_errors_to_misses(status):
    with pytest.raises(VercelBlobTransportError, match=f"HTTP {status}"):
        transport([http_error(status)]).list_page(STORE, cursor=None, limit=1, deadline_epoch=101)


def test_deadline_expiry_prevents_io():
    t = transport([], clock=lambda: 100)
    with pytest.raises(VercelBlobDeadlineError):
        t.list_page(STORE, cursor=None, limit=1, deadline_epoch=100)
    assert t._opener.calls == []


def test_read_deadline_must_hold_after_full_body_and_closes_response():
    now = {"value": 100}
    response = page([])
    response._on_read = lambda: now.update(value=101)
    t = transport([response], clock=lambda: now["value"])
    with pytest.raises(VercelBlobDeadlineError, match="completed"):
        t.list_page(STORE, cursor=None, limit=1, deadline_epoch=101)
    assert len(t._opener.calls) == 1
    assert response.closed


def test_rejects_cross_store_call_and_redirect():
    with pytest.raises(VercelBlobTransportError, match="store_id"):
        transport([]).list_page("other", cursor=None, limit=1, deadline_epoch=101)
    t = transport([Response(b'{}', url="https://evil.example/")])
    with pytest.raises(VercelBlobTransportError, match="redirect"):
        t.list_page(STORE, cursor=None, limit=1, deadline_epoch=101)


@pytest.mark.parametrize(
    "url",
    [
        "http://vercel.com/api/blob",
        "https://vercel.com.evil.example/api/blob",
        "https://vercel.com@evil.example/api/blob",
        "https://evil.example/api/blob",
        "https://vercel.com:443/api/blob",
        "https://vercel.com/api/blob/extra",
        "https://store-a.public.blob.vercel-storage.com/lease?credential=1",
        "https://other.public.blob.vercel-storage.com/lease",
    ],
)
def test_credentials_are_never_sent_to_noncanonical_urls(url):
    t = transport([])
    with pytest.raises(VercelBlobTransportError, match="credential URL"):
        t._request("GET", url, 101)
    assert t._opener.calls == []


def test_put_if_absent_uses_official_query_and_only_conflict_is_a_miss():
    response = Response(json.dumps({"pathname": "a b", "etag": "new"}).encode())
    t = transport([response])
    assert t.put_if_absent(STORE, "a b", b"abc", deadline_epoch=101) == BlobIdentity("a b", 3, "new")
    request, _ = t._opener.calls[0]
    assert request.full_url == "https://vercel.com/api/blob/?pathname=a+b"
    headers = request_headers(request)
    assert headers["x-vercel-blob-access"] == "public"
    assert headers["x-add-random-suffix"] == "0"
    assert "x-allow-overwrite" not in headers
    assert request.data == b"abc"
    assert (
        transport([http_error(412, "precondition_failed")]).put_if_absent(
            STORE, "a", b"x", deadline_epoch=101
        )
        is None
    )
    for error in (
        http_error(409, "precondition_failed"),
        http_error(412, "store_not_found"),
        http_error(412, payload=b"not-json"),
        http_error(412, payload=b'{"code":"precondition_failed"}'),
    ):
        with pytest.raises(VercelBlobTransportError, match="HTTP"):
            transport([error]).put_if_absent(
                STORE, "a", b"x", deadline_epoch=101
            )
    for status in (401, 429, 500):
        with pytest.raises(VercelBlobTransportError):
            transport([http_error(status)]).put_if_absent(STORE, "a", b"x", deadline_epoch=101)


def test_mutation_deadline_crossing_and_timeout_are_ambiguous_and_not_retried():
    now = {"value": 100}
    response = Response(
        b'{"pathname":"a","etag":"new"}',
        on_read=lambda: now.update(value=101),
    )
    t = transport([response], clock=lambda: now["value"])
    with pytest.raises(VercelBlobAmbiguousMutationError, match="ambiguous"):
        t.put_if_absent(STORE, "a", b"x", deadline_epoch=101)
    assert len(t._opener.calls) == 1
    assert response.closed

    timed_out = transport([TimeoutError("socket timeout")])
    with pytest.raises(VercelBlobAmbiguousMutationError, match="ambiguous"):
        timed_out.put_if_absent(STORE, "a", b"x", deadline_epoch=101)
    assert len(timed_out._opener.calls) == 1

    failed_body = Response(read_error=TimeoutError("body timeout"))
    body_timeout = transport([failed_body])
    with pytest.raises(VercelBlobAmbiguousMutationError, match="ambiguous"):
        body_timeout.put_if_absent(STORE, "a", b"x", deadline_epoch=101)
    assert len(body_timeout._opener.calls) == 1
    assert failed_body.closed


def test_conditional_error_body_must_also_finish_before_mutation_deadline():
    now = {"value": 100}
    body = Response(
        b'{"error":{"code":"precondition_failed"}}',
        on_read=lambda: now.update(value=101),
    )
    error = http_error(412, fp=body)
    t = transport([error], clock=lambda: now["value"])
    with pytest.raises(VercelBlobAmbiguousMutationError, match="ambiguous"):
        t.put_if_absent(STORE, "a", b"x", deadline_epoch=101)
    assert len(t._opener.calls) == 1
    assert body.closed


def test_put_if_match_uses_etag_overwrite_and_validates_returned_path():
    t = transport([Response(json.dumps({"pathname": "a", "etag": "new"}).encode())])
    assert t.put_if_match(STORE, "a", b"xy", etag="old", deadline_epoch=101) == BlobIdentity("a", 2, "new")
    headers = request_headers(t._opener.calls[0][0])
    assert headers["x-if-match"] == "old"
    assert headers["x-allow-overwrite"] == "1"
    for status, code in ((404, "not_found"), (412, "precondition_failed")):
        assert (
            transport([http_error(status, code)]).put_if_match(
                STORE, "a", b"x", etag="old", deadline_epoch=101
            )
            is None
        )
    for error in (
        http_error(404, "store_not_found"),
        http_error(412, "not_found"),
        http_error(412, payload=b""),
        http_error(409, "precondition_failed"),
    ):
        with pytest.raises(VercelBlobTransportError, match="HTTP"):
            transport([error]).put_if_match(
                STORE, "a", b"x", etag="old", deadline_epoch=101
            )
    with pytest.raises(VercelBlobTransportError, match="pathname mismatch"):
        transport([Response(b'{"pathname":"other","etag":"new"}')]).put_if_match(STORE, "a", b"x", etag="old", deadline_epoch=101)


def test_get_binds_authenticated_inventory_body_length_and_etag():
    listed = page([blob("lease.json", 3, "etag-a")])
    body = Response(b"abc", headers={"etag": "etag-a"})
    t = transport([listed, body])
    assert t.get_blob(STORE, "lease.json", deadline_epoch=101) == (b"abc", BlobIdentity("lease.json", 3, "etag-a"))
    assert t._opener.calls[0][0].full_url == "https://vercel.com/api/blob?prefix=lease.json&limit=1000&mode=expanded"
    assert t._opener.calls[1][0].full_url == "https://store-a.public.blob.vercel-storage.com/lease.json"
    public_headers = request_headers(t._opener.calls[1][0])
    assert public_headers["authorization"] == f"Bearer {TOKEN}"
    assert "x-api-version" not in public_headers
    assert "x-vercel-blob-store-id" not in public_headers
    for bad in [Response(b"abcd", headers={"etag": "etag-a"}), Response(b"abc", headers={"etag": "wrong"})]:
        with pytest.raises(VercelBlobTransportError, match="identity mismatch"):
            transport([page([blob("lease.json", 3, "etag-a")]), bad]).get_blob(STORE, "lease.json", deadline_epoch=101)


def test_get_404_is_not_found_but_other_failures_are_not_misses():
    assert transport([page([])]).get_blob(STORE, "missing", deadline_epoch=101) is None
    public_url = "https://store-a.public.blob.vercel-storage.com/missing"
    assert (
        transport(
            [
                page([blob("missing", 1, "old")]),
                http_error(404, url=public_url),
            ]
        ).get_blob(STORE, "missing", deadline_epoch=101)
        is None
    )
    with pytest.raises(VercelBlobTransportError):
        transport([http_error(401)]).get_blob(STORE, "missing", deadline_epoch=101)


def test_delete_prebinds_exact_identity_and_uses_conditional_api():
    t = transport([page([blob("a", 3, "old")]), Response(b"")])
    assert t.delete_if_match(STORE, "a", etag="old", deadline_epoch=101) == BlobIdentity("a", 3, "old")
    request, _ = t._opener.calls[1]
    assert request.full_url == "https://vercel.com/api/blob/delete"
    assert request.data == b'{"urls":["a"]}'
    headers = request_headers(request)
    assert headers["content-type"] == "application/json"
    assert headers["x-if-match"] == "old"
    assert transport([page([blob("a", 3, "new")])]).delete_if_match(STORE, "a", etag="old", deadline_epoch=101) is None
    for status, code in ((404, "not_found"), (412, "precondition_failed")):
        assert (
            transport(
                [page([blob("a", 3, "old")]), http_error(status, code)]
            ).delete_if_match(
                STORE, "a", etag="old", deadline_epoch=101
            )
            is None
        )
    for error in (
        http_error(404, "store_not_found"),
        http_error(412, "not_found"),
        http_error(412, payload=b"not-json"),
        http_error(409, "precondition_failed"),
    ):
        with pytest.raises(VercelBlobTransportError, match="HTTP"):
            transport([page([blob("a", 3, "old")]), error]).delete_if_match(
                STORE, "a", etag="old", deadline_epoch=101
            )
    with pytest.raises(VercelBlobTransportError):
        transport([page([blob("a", 3, "old")]), http_error(500)]).delete_if_match(STORE, "a", etag="old", deadline_epoch=101)


def test_errors_do_not_leak_token():
    t = transport([http_error(500)])
    with pytest.raises(VercelBlobTransportError) as exc:
        t.list_page(STORE, cursor=None, limit=1, deadline_epoch=101)
    assert TOKEN not in str(exc.value)
