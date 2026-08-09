"""Fail-closed Vercel Blob adapter for the G0-209 retention executor.

This deliberately implements only the small authenticated transport surface the
retention guard needs.  It accepts only an explicit Vercel Blob read-write token,
binds that token to one explicit store, does not retry mutations, and never
follows redirects.

The absolute deadline is checked before I/O, after response headers, and after
the full response body is consumed.  The stdlib socket timeout is not proof that
Vercel cancelled a request already sent; a mutation that times out or crosses
the deadline is therefore reported as having an ambiguous outcome and is never
retried.  A successful write establishes metadata only; it does not prove
remote content integrity.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from lol_kills.export.blob_retention import BlobIdentity, TransportResultError


_API_ROOT = "https://vercel.com/api/blob"
_API_VERSION = "12"
_MAX_TIMEOUT_SECONDS = 30.0
_MAX_ERROR_BODY_BYTES = 16_384
_READ_WRITE_TOKEN_PREFIX = "vercel_blob_rw_"


class VercelBlobTransportError(TransportResultError):
    """A redacted, fail-closed Vercel Blob transport failure."""


class VercelBlobDeadlineError(VercelBlobTransportError):
    """The guard deadline expired before a read-only request completed."""


class VercelBlobAmbiguousMutationError(VercelBlobTransportError):
    """A sent mutation may have committed, but its timely result is unproven."""


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise VercelBlobTransportError("unexpected HTTP redirect")


class VercelBlobTransport:
    """Read-write-token-authenticated, explicit-store Vercel Blob transport.

    The caller must supply the same ``store_id`` to every protocol call.  The
    adapter accepts either Vercel's ``store_<id>`` notation or the bare ID, then
    uses the bare form required by its control-plane header and public hostname.
    The store encoded in the read-write token must match exactly.
    """

    def __init__(
        self,
        token: str,
        store_id: str,
        *,
        opener: Any | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        normalized_store = self._normalize_store_id(store_id)
        token_store = self._read_write_token_store(token)
        if token_store != normalized_store:
            raise ValueError("token store does not match explicit store_id")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._token = token
        self.store_id = normalized_store
        self._clock = clock
        self._opener = opener or urllib.request.build_opener(_RejectRedirect())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(store_id={self.store_id!r}, token=<redacted>)"

    def list_page(
        self,
        store_id: str,
        *,
        cursor: str | None,
        limit: int,
        deadline_epoch: int,
    ) -> dict[str, object]:
        self._store(store_id)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise VercelBlobTransportError("invalid list limit")
        if cursor is not None and (type(cursor) is not str or not cursor):
            raise VercelBlobTransportError("invalid pagination cursor")
        query: dict[str, str] = {"limit": str(limit), "mode": "expanded"}
        if cursor is not None:
            query["cursor"] = cursor
        payload = self._json_request(
            "GET", self._list_api_url(query), deadline_epoch
        )
        return self._page(payload)

    def get_blob(
        self, store_id: str, pathname: str, *, deadline_epoch: int
    ) -> tuple[bytes, BlobIdentity] | None:
        self._store(store_id)
        self._pathname(pathname)
        identity = self._lookup(pathname, deadline_epoch)
        if identity is None:
            return None
        # Vercel's SDK constructs exactly this public hostname from the
        # store-id/pathname.  We disable redirect following and bind both ETag
        # and byte length to the authenticated list result before returning.
        public_url = (
            f"https://{self.store_id}.public.blob.vercel-storage.com/"
            f"{urllib.parse.quote(pathname, safe='/')}"
        )
        response = self._request(
            "GET", public_url, deadline_epoch, public_not_found=True
        )
        if response is None:
            return None
        try:
            try:
                body = response.read()
                header_etag = response.headers.get("etag")
            except Exception:
                raise VercelBlobTransportError(
                    "blob read response failed"
                ) from None
        finally:
            response.close()
        self._prove_completion(deadline_epoch, "blob read", mutation=False)
        if type(body) is not bytes or type(header_etag) is not str:
            raise VercelBlobTransportError("blob read response is malformed")
        if len(body) != identity.size or header_etag != identity.etag:
            raise VercelBlobTransportError("blob read identity mismatch")
        return body, identity

    def put_if_absent(
        self, store_id: str, pathname: str, content: bytes, *, deadline_epoch: int
    ) -> BlobIdentity | None:
        self._store(store_id)
        self._pathname(pathname)
        self._content(content)
        response = self._request(
            "PUT",
            self._put_api_url({"pathname": pathname}),
            deadline_epoch,
            content=content,
            extra_headers={
                "x-vercel-blob-access": "public",
                "x-add-random-suffix": "0",
                "x-content-length": str(len(content)),
            },
            conditional_errors={(412, "precondition_failed")},
        )
        if response is None:
            return None
        return self._put_identity(
            response,
            pathname,
            len(content),
            "put-if-absent",
            deadline_epoch,
        )

    def put_if_match(
        self,
        store_id: str,
        pathname: str,
        content: bytes,
        *,
        etag: str,
        deadline_epoch: int,
    ) -> BlobIdentity | None:
        self._store(store_id)
        self._pathname(pathname)
        self._content(content)
        self._etag(etag)
        response = self._request(
            "PUT",
            self._put_api_url({"pathname": pathname}),
            deadline_epoch,
            content=content,
            extra_headers={
                "x-vercel-blob-access": "public",
                "x-add-random-suffix": "0",
                "x-allow-overwrite": "1",
                "x-if-match": etag,
                "x-content-length": str(len(content)),
            },
            conditional_errors={
                (404, "not_found"),
                (412, "precondition_failed"),
            },
        )
        if response is None:
            return None
        return self._put_identity(
            response,
            pathname,
            len(content),
            "put-if-match",
            deadline_epoch,
        )

    def delete_if_match(
        self,
        store_id: str,
        pathname: str,
        *,
        etag: str,
        deadline_epoch: int,
    ) -> BlobIdentity | None:
        self._store(store_id)
        self._pathname(pathname)
        self._etag(etag)
        prior = self._lookup(pathname, deadline_epoch)
        if prior is None or prior.etag != etag:
            return None
        body = json.dumps({"urls": [pathname]}, separators=(",", ":")).encode("utf-8")
        response = self._request(
            "POST",
            f"{_API_ROOT}/delete",
            deadline_epoch,
            content=body,
            extra_headers={
                "content-type": "application/json",
                "x-if-match": etag,
                "x-content-length": str(len(body)),
            },
            conditional_errors={
                (404, "not_found"),
                (412, "precondition_failed"),
            },
        )
        if response is None:
            return None
        # Delete's official success contract has no response identity.  The
        # preflight list entry plus the exact conditional success is the only
        # capability this method returns.
        self._response_bytes(
            response,
            deadline_epoch,
            "delete-if-match",
            mutation=True,
        )
        return prior

    def _store(self, store_id: str) -> None:
        if type(store_id) is not str or store_id != self.store_id:
            raise VercelBlobTransportError("store_id does not match transport")

    @staticmethod
    def _normalize_store_id(store_id: str) -> str:
        if type(store_id) is not str or not store_id or store_id != store_id.strip():
            raise ValueError("store_id must be an exact nonempty string")
        normalized = store_id
        if normalized.startswith("store_"):
            normalized = normalized[len("store_") :]
        if (
            not normalized
            or len(normalized) > 63
            or normalized != normalized.lower()
            or not normalized[0].isalnum()
            or not normalized[-1].isalnum()
            or not normalized.isascii()
            or any(
                not (character.isalnum() or character == "-")
                for character in normalized
            )
        ):
            raise ValueError("store_id is malformed")
        return normalized

    @classmethod
    def _read_write_token_store(cls, token: str) -> str:
        if (
            type(token) is not str
            or not token
            or token != token.strip()
            or not token.isascii()
            or any(
                character.isspace()
                or ord(character) < 33
                or ord(character) == 127
                for character in token
            )
            or not token.startswith(_READ_WRITE_TOKEN_PREFIX)
        ):
            raise ValueError("token must be an exact Vercel Blob read-write token")
        encoded = token[len(_READ_WRITE_TOKEN_PREFIX) :]
        token_store, separator, secret = encoded.partition("_")
        if not separator or not secret:
            raise ValueError("token must be an exact Vercel Blob read-write token")
        try:
            # Vercel tokens can preserve mixed-case store IDs. Public Blob
            # hostnames use the lowercase form, so normalize only the token
            # identity before the existing hostname-safe validation.
            return cls._normalize_store_id(token_store.casefold())
        except ValueError as error:
            raise ValueError("token contains a malformed store identity") from error

    @staticmethod
    def _pathname(pathname: str) -> None:
        try:
            BlobIdentity(pathname, 0, "validation")
        except Exception as error:
            raise VercelBlobTransportError("pathname is invalid") from error

    @staticmethod
    def _content(content: bytes) -> None:
        if type(content) is not bytes:
            raise VercelBlobTransportError("content must be exact bytes")

    @staticmethod
    def _etag(etag: str) -> None:
        if type(etag) is not str or not etag:
            raise VercelBlobTransportError("etag is invalid")

    def _response_bytes(
        self,
        response: Any,
        deadline_epoch: int,
        operation: str,
        *,
        mutation: bool,
    ) -> bytes:
        try:
            try:
                raw = response.read()
            except Exception:
                if mutation:
                    raise VercelBlobAmbiguousMutationError(
                        f"{operation} outcome is ambiguous after response failure"
                    ) from None
                raise VercelBlobTransportError(
                    f"{operation} response read failed"
                ) from None
        finally:
            response.close()
        self._prove_completion(deadline_epoch, operation, mutation=mutation)
        if type(raw) is not bytes:
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{operation} outcome is ambiguous after malformed response"
                )
            raise VercelBlobTransportError(f"{operation} response is malformed")
        return raw

    def _json_object(
        self,
        response: Any,
        operation: str,
        deadline_epoch: int,
        *,
        mutation: bool,
    ) -> dict[str, object]:
        raw = self._response_bytes(
            response,
            deadline_epoch,
            operation,
            mutation=mutation,
        )
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VercelBlobTransportError(f"{operation} response is not JSON") from error
        if type(parsed) is not dict:
            raise VercelBlobTransportError(f"{operation} response is not an object")
        return dict(parsed)

    def _json_request(self, method: str, url: str, deadline_epoch: int) -> dict[str, object]:
        response = self._request(method, url, deadline_epoch)
        if response is None:  # List never declares a conditional miss.
            raise VercelBlobTransportError("unexpected conditional list result")
        return self._json_object(
            response,
            "list",
            deadline_epoch,
            mutation=False,
        )

    def _page(self, payload: dict[str, object]) -> dict[str, object]:
        raw_blobs, has_more = payload.get("blobs"), payload.get("hasMore")
        cursor = payload.get("cursor")
        if type(raw_blobs) is not list or type(has_more) is not bool:
            raise VercelBlobTransportError("list response schema is malformed")
        if len(raw_blobs) > 1000:
            raise VercelBlobTransportError("list response exceeds page limit")
        if cursor is not None and (type(cursor) is not str or not cursor):
            raise VercelBlobTransportError("list cursor is malformed")
        if has_more and cursor is None:
            raise VercelBlobTransportError("list continuation cursor is missing")
        blobs: list[dict[str, object]] = []
        for raw in raw_blobs:
            if type(raw) is not dict:
                raise VercelBlobTransportError("list blob entry is malformed")
            item = dict(raw)
            try:
                identity = BlobIdentity(item["pathname"], item["size"], item["etag"])
            except Exception as error:
                raise VercelBlobTransportError("list blob metadata is malformed") from error
            blobs.append(
                {"pathname": identity.pathname, "size": identity.size, "etag": identity.etag}
            )
        return {"storeId": self.store_id, "blobs": blobs, "hasMore": has_more, "cursor": cursor}

    def _lookup(self, pathname: str, deadline_epoch: int) -> BlobIdentity | None:
        payload = self._json_request(
            "GET",
            self._list_api_url(
                {"prefix": pathname, "limit": "1000", "mode": "expanded"}
            ),
            deadline_epoch,
        )
        page = self._page(payload)
        if page["hasMore"]:
            raise VercelBlobTransportError("exact blob lookup is ambiguous")
        matches = [
            item
            for item in page["blobs"]  # type: ignore[index]
            if type(item) is dict and item.get("pathname") == pathname
        ]
        if len(matches) > 1:
            raise VercelBlobTransportError("exact blob lookup returned duplicates")
        if not matches:
            return None
        item = matches[0]
        return BlobIdentity(item["pathname"], item["size"], item["etag"])

    def _put_identity(
        self,
        response: Any,
        pathname: str,
        size: int,
        operation: str,
        deadline_epoch: int,
    ) -> BlobIdentity:
        payload = self._json_object(
            response,
            operation,
            deadline_epoch,
            mutation=True,
        )
        try:
            returned_path, returned_etag = payload["pathname"], payload["etag"]
            identity = BlobIdentity(returned_path, size, returned_etag)
        except Exception as error:
            raise VercelBlobTransportError(f"{operation} response is malformed") from error
        if identity.pathname != pathname:
            raise VercelBlobTransportError(f"{operation} pathname mismatch")
        return identity

    @staticmethod
    def _list_api_url(query: dict[str, str]) -> str:
        return f"{_API_ROOT}?{urllib.parse.urlencode(query)}"

    @staticmethod
    def _put_api_url(query: dict[str, str]) -> str:
        return f"{_API_ROOT}/?{urllib.parse.urlencode(query)}"

    def _clock_value(self) -> float:
        try:
            value = self._clock()
        except Exception as error:
            raise VercelBlobDeadlineError("clock failed") from error
        if type(value) not in (int, float):
            raise VercelBlobDeadlineError("clock returned an invalid value")
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise VercelBlobDeadlineError("clock returned an invalid value")
        return float(value)

    def _timeout(self, deadline_epoch: int) -> float:
        if type(deadline_epoch) is not int:
            raise VercelBlobDeadlineError("deadline must be an exact integer epoch")
        remaining = deadline_epoch - self._clock_value()
        if remaining <= 0:
            raise VercelBlobDeadlineError("deadline expired before request")
        return min(float(remaining), _MAX_TIMEOUT_SECONDS)

    def _prove_completion(
        self,
        deadline_epoch: int,
        operation: str,
        *,
        mutation: bool,
    ) -> None:
        if type(deadline_epoch) is not int:
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{operation} outcome is ambiguous without an exact deadline"
                )
            raise VercelBlobDeadlineError("deadline must be an exact integer epoch")
        try:
            remaining = deadline_epoch - self._clock_value()
        except VercelBlobDeadlineError:
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{operation} outcome is ambiguous because completion time is unproven"
                ) from None
            raise
        if remaining <= 0:
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{operation} outcome is ambiguous after deadline"
                )
            raise VercelBlobDeadlineError(
                f"deadline expired before {operation} completed"
            )

    def _credential_url_kind(self, url: str) -> str:
        if type(url) is not str or not url:
            raise VercelBlobTransportError("credential URL is malformed")
        try:
            parsed = urllib.parse.urlsplit(url)
            port = parsed.port
        except (TypeError, ValueError):
            raise VercelBlobTransportError("credential URL is malformed") from None
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.fragment
        ):
            raise VercelBlobTransportError("credential URL is not permitted")
        if (
            parsed.netloc == "vercel.com"
            and parsed.hostname == "vercel.com"
            and parsed.path in {"/api/blob", "/api/blob/", "/api/blob/delete"}
        ):
            return "api"
        public_host = f"{self.store_id}.public.blob.vercel-storage.com"
        if (
            parsed.netloc == public_host
            and parsed.hostname == public_host
            and parsed.path.startswith("/")
            and not parsed.query
        ):
            return "public"
        raise VercelBlobTransportError("credential URL is not permitted")

    def _error_code(
        self,
        error: urllib.error.HTTPError,
        deadline_epoch: int,
        operation: str,
        *,
        mutation: bool,
    ) -> str | None:
        try:
            try:
                raw = error.read(_MAX_ERROR_BODY_BYTES + 1)
            except Exception:
                if mutation:
                    raise VercelBlobAmbiguousMutationError(
                        f"{operation} outcome is ambiguous after response failure"
                    ) from None
                raise VercelBlobTransportError(
                    f"{operation} error response read failed"
                ) from None
        finally:
            error.close()
        self._prove_completion(deadline_epoch, operation, mutation=mutation)
        if type(raw) is not bytes or len(raw) > _MAX_ERROR_BODY_BYTES:
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if type(payload) is not dict or type(payload.get("error")) is not dict:
            return None
        code = payload["error"].get("code")
        if type(code) is not str or not code:
            return None
        return code

    def _request(
        self,
        method: str,
        url: str,
        deadline_epoch: int,
        *,
        content: bytes | None = None,
        extra_headers: dict[str, str] | None = None,
        conditional_errors: set[tuple[int, str]] | None = None,
        public_not_found: bool = False,
    ) -> Any | None:
        timeout = self._timeout(deadline_epoch)
        url_kind = self._credential_url_kind(url)
        mutation = method in {"DELETE", "PATCH", "POST", "PUT"}
        headers = {"authorization": f"Bearer {self._token}"}
        if url_kind == "api":
            headers.update(
                {
                    "x-api-version": _API_VERSION,
                    "x-vercel-blob-store-id": self.store_id,
                }
            )
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(url, data=content, headers=headers, method=method)
        try:
            response = self._opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            try:
                final_url = error.geturl()
            except Exception:
                error.close()
                if mutation:
                    raise VercelBlobAmbiguousMutationError(
                        f"{method} outcome is ambiguous because response URL is unavailable"
                    ) from None
                raise VercelBlobTransportError(
                    "response URL is unavailable"
                ) from None
            try:
                final_kind = self._credential_url_kind(final_url)
            except VercelBlobTransportError:
                error.close()
                if mutation:
                    raise VercelBlobAmbiguousMutationError(
                        f"{method} outcome is ambiguous after an invalid redirect"
                    ) from None
                raise VercelBlobTransportError(
                    "unexpected HTTP redirect"
                ) from None
            if final_url != url or final_kind != url_kind:
                error.close()
                if mutation:
                    raise VercelBlobAmbiguousMutationError(
                        f"{method} outcome is ambiguous after redirect"
                    )
                raise VercelBlobTransportError("unexpected HTTP redirect")
            try:
                self._prove_completion(deadline_epoch, method, mutation=mutation)
            except VercelBlobTransportError:
                error.close()
                raise
            code = self._error_code(
                error,
                deadline_epoch,
                method,
                mutation=mutation,
            )
            if public_not_found and url_kind == "public" and error.code == 404:
                return None
            if conditional_errors and (error.code, code) in conditional_errors:
                return None
            raise VercelBlobTransportError(f"{method} request failed with HTTP {error.code}") from None
        except VercelBlobTransportError as error:
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{method} outcome is ambiguous after transport failure"
                ) from error
            raise
        except Exception as error:
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{method} outcome is ambiguous after transport failure"
                ) from error
            raise VercelBlobTransportError(f"{method} request failed") from error
        try:
            final_url = response.geturl()
        except Exception:
            response.close()
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{method} outcome is ambiguous because response URL is unavailable"
                ) from None
            raise VercelBlobTransportError("response URL is unavailable") from None
        try:
            final_kind = self._credential_url_kind(final_url)
        except VercelBlobTransportError:
            response.close()
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{method} outcome is ambiguous after an invalid redirect"
                ) from None
            raise VercelBlobTransportError("unexpected HTTP redirect") from None
        if final_url != url or final_kind != url_kind:
            response.close()
            if mutation:
                raise VercelBlobAmbiguousMutationError(
                    f"{method} outcome is ambiguous after redirect"
                )
            raise VercelBlobTransportError("unexpected HTTP redirect")
        try:
            self._prove_completion(deadline_epoch, method, mutation=mutation)
        except VercelBlobTransportError:
            response.close()
            raise
        return response
