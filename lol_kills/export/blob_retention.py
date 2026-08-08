"""G0-209 Vercel Blob retention guard, policy v1.

The hard stop is a pre-upload stop.  A plan whose current or projected peak is
at least 850,000,000 decimal bytes cannot contain a write.  At that boundary an
exact, nonempty deletion-only plan remains available only when it strictly
reduces persistent retained bytes.  There is no runtime exception, force flag,
approval object, verifier, or environment override.  A future exception would
require a new content-addressed policy version reviewed and released as code.

Trust boundary: the deployed source/configuration and authenticated transport
implementation are trusted.  Plans, list pages, transport responses, clocks,
and concurrent callers are untrusted.  The guard prevents unauthorized plan
execution and coordinates cooperating writers; it cannot defend against stolen
write credentials or rewritten deployed code.

Vercel Blob write responses establish only pathname, size, and opaque ETag.
They do not provide a content SHA-256.  The sealed-plan hash identifies intended
local bytes; it is not remote content-integrity evidence.  Readback is outside
storage-quota gating.
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, final


POLICY_VERSION = "g0-209-retention-v1"
TARGET_BYTES = 500_000_000
WARNING_BYTES = 700_000_000
HARD_STOP_BYTES = 850_000_000
MAX_PAGE_SIZE = 1_000
MAX_INVENTORY_AGE_SECONDS = 300
LEASE_TTL_SECONDS = 120
LEASE_MARGIN_SECONDS = 10
CONTROL_PREFIX = "_scryglass_retention/"
LEASE_SCHEMA = "scryglass-retention-lease-v1"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


POLICY_SHA256 = _digest(
    {
        "version": POLICY_VERSION,
        "target": TARGET_BYTES,
        "warning": WARNING_BYTES,
        "hard_stop": HARD_STOP_BYTES,
        "hard_stop_inclusive": True,
        "deletion_only_monotonic_recovery": True,
        "runtime_override": False,
        "order": ["writes", "deletions", "lease_release"],
        "lease_bytes_counted": True,
        "inventory_max_age": MAX_INVENTORY_AGE_SECONDS,
        "control_prefix": CONTROL_PREFIX,
    }
)


class RetentionError(RuntimeError):
    pass


class BoundaryTypeError(RetentionError, TypeError):
    pass


class PlanError(RetentionError):
    pass


class InventoryError(RetentionError):
    pass


class LeaseError(RetentionError):
    pass


class HardStopError(RetentionError):
    def __init__(self, current: int, peak: int, reason: str) -> None:
        self.current_retained_bytes = current
        self.peak_retained_bytes = peak
        self.reason = reason
        super().__init__(
            f"{reason}: current={current}, peak={peak}, "
            f"inclusive hard stop={HARD_STOP_BYTES}"
        )


class TransportResultError(RetentionError):
    pass


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise BoundaryTypeError(f"{name} must be an exact nonempty str")
    return value


def _size(value: object, name: str = "size") -> int:
    if type(value) is not int or value < 0:
        raise BoundaryTypeError(f"{name} must be an exact nonnegative int")
    return value


def _path(pathname: str, *, control: bool) -> str:
    _text(pathname, "pathname")
    if (
        unicodedata.normalize("NFC", pathname) != pathname
        or pathname.startswith("/")
        or pathname.endswith("/")
        or "\\" in pathname
        or "//" in pathname
        or any(part in {"", ".", ".."} for part in pathname.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 for char in pathname)
    ):
        raise PlanError(f"noncanonical pathname: {pathname!r}")
    if not control and (
        pathname == CONTROL_PREFIX.removesuffix("/") or pathname.startswith(CONTROL_PREFIX)
    ):
        raise PlanError(f"reserved control pathname: {pathname!r}")
    return pathname


@final
@dataclass(frozen=True)
class BlobIdentity:
    pathname: str
    size: int
    etag: str

    def __post_init__(self) -> None:
        # Cheap construction check; the executor seal is authoritative.
        _path(self.pathname, control=True)
        _size(self.size)
        _text(self.etag, "etag")


class WriteMode(str, Enum):
    NEW_IMMUTABLE = "NEW_IMMUTABLE"
    OVERWRITE = "OVERWRITE"


@final
@dataclass(frozen=True)
class PlannedWrite:
    pathname: str
    content: bytes
    mode: WriteMode

    def __post_init__(self) -> None:
        _path(self.pathname, control=False)
        if type(self.content) is not bytes or type(self.mode) is not WriteMode:
            raise BoundaryTypeError("write requires exact bytes and WriteMode")


@final
@dataclass(frozen=True)
class RetentionPlan:
    store_id: str
    writer_id: str
    run_id: str
    writes: tuple[PlannedWrite, ...]
    deletions: tuple[BlobIdentity, ...] = ()

    def __post_init__(self) -> None:
        # Nested values are intentionally inspected only by _seal_plan.
        _text(self.store_id, "store_id")
        _text(self.writer_id, "writer_id")
        _text(self.run_id, "run_id")
        if type(self.writes) is not tuple or type(self.deletions) is not tuple:
            raise BoundaryTypeError("writes and deletions must be exact tuples")


@dataclass(frozen=True)
class _Identity:
    pathname: str
    size: int
    etag: str

    def object(self) -> dict[str, object]:
        return {"pathname": self.pathname, "size": self.size, "etag": self.etag}


@dataclass(frozen=True)
class _Write:
    pathname: str
    content: bytes
    mode: WriteMode
    content_sha256: str

    @property
    def size(self) -> int:
        return len(self.content)

    def object(self) -> dict[str, object]:
        return {
            "pathname": self.pathname,
            "size": self.size,
            "mode": self.mode.value,
            "intended_content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class _Plan:
    store_id: str
    writer_id: str
    run_id: str
    writes: tuple[_Write, ...]
    deletions: tuple[_Identity, ...]
    sha256: str


def _public_identity(value: object, name: str, *, user_path: bool) -> _Identity:
    if type(value) is not BlobIdentity:
        raise BoundaryTypeError(f"{name} must be an exact BlobIdentity")
    pathname, size, etag = value.pathname, value.size, value.etag
    _path(pathname, control=not user_path)
    _size(size, f"{name}.size")
    _text(etag, f"{name}.etag")
    return _Identity(pathname, size, etag)


def _seal_plan(value: object) -> _Plan:
    """Authoritative one-time deep snapshot of all untrusted plan values."""

    if type(value) is not RetentionPlan:
        raise BoundaryTypeError("execute requires an exact RetentionPlan")
    store_id, writer_id, run_id = value.store_id, value.writer_id, value.run_id
    raw_writes, raw_deletions = value.writes, value.deletions
    _text(store_id, "store_id")
    _text(writer_id, "writer_id")
    _text(run_id, "run_id")
    if type(raw_writes) is not tuple or type(raw_deletions) is not tuple:
        raise BoundaryTypeError("writes and deletions must be exact tuples")

    writes: list[_Write] = []
    for raw in raw_writes:
        if type(raw) is not PlannedWrite:
            raise BoundaryTypeError("every write must be an exact PlannedWrite")
        pathname, content, mode = raw.pathname, raw.content, raw.mode
        _path(pathname, control=False)
        if type(content) is not bytes or type(mode) is not WriteMode:
            raise BoundaryTypeError("write requires exact bytes and WriteMode")
        snapshot = memoryview(content).tobytes()
        writes.append(
            _Write(pathname, snapshot, mode, hashlib.sha256(snapshot).hexdigest())
        )
    deletions = [
        _public_identity(raw, "deletion", user_path=True) for raw in raw_deletions
    ]
    writes.sort(key=lambda item: item.pathname)
    deletions.sort(key=lambda item: item.pathname)
    write_paths = [item.pathname for item in writes]
    delete_paths = [item.pathname for item in deletions]
    if len(set(write_paths)) != len(write_paths) or len(set(delete_paths)) != len(
        delete_paths
    ):
        raise PlanError("duplicate planned pathname")
    if set(write_paths) & set(delete_paths):
        raise PlanError("conflicting write and deletion pathname")
    payload = {
        "policy_sha256": POLICY_SHA256,
        "store_id": store_id,
        "writer_id": writer_id,
        "run_id": run_id,
        "writes": [item.object() for item in writes],
        "deletions": [item.object() for item in deletions],
    }
    return _Plan(
        store_id, writer_id, run_id, tuple(writes), tuple(deletions), _digest(payload)
    )


class ResultState(str, Enum):
    NORMAL = "normal"
    OVER_TARGET = "over_target"
    WARNING = "warning"
    RECOVERY = "deletion_only_recovery"


@final
@dataclass(frozen=True)
class OperationStatus:
    kind: str
    pathname: str
    success: bool
    detail: str | None = None


@final
@dataclass(frozen=True)
class RetentionResult:
    state: ResultState
    policy_sha256: str
    plan_sha256: str
    inventory_sha256: str
    current_retained_bytes: int
    peak_retained_bytes: int
    projected_final_bytes: int
    actual_final_bytes: int | None
    operations: tuple[OperationStatus, ...]
    success: bool
    lease_released: bool
    remote_content_verified: bool = False


class BlobTransport(Protocol):
    """Trusted authenticated adapter whose returned values remain untrusted."""

    def list_page(
        self,
        store_id: str,
        *,
        cursor: str | None,
        limit: int,
        deadline_epoch: int,
    ) -> dict[str, object]: ...

    def get_blob(
        self, store_id: str, pathname: str, *, deadline_epoch: int
    ) -> tuple[bytes, BlobIdentity] | None: ...

    def put_if_absent(
        self, store_id: str, pathname: str, content: bytes, *, deadline_epoch: int
    ) -> BlobIdentity | None: ...

    def put_if_match(
        self,
        store_id: str,
        pathname: str,
        content: bytes,
        *,
        etag: str,
        deadline_epoch: int,
    ) -> BlobIdentity | None: ...

    def delete_if_match(
        self,
        store_id: str,
        pathname: str,
        *,
        etag: str,
        deadline_epoch: int,
    ) -> BlobIdentity | None: ...


def _put_result(value: object, pathname: str, size: int, operation: str) -> _Identity:
    try:
        identity = _public_identity(value, f"{operation} result", user_path=False)
    except RetentionError as error:
        raise TransportResultError(f"{operation} returned no exact identity") from error
    if identity.pathname != pathname or identity.size != size:
        raise TransportResultError(f"{operation} returned wrong pathname or size")
    return identity


def _delete_result(value: object, expected: _Identity, operation: str) -> None:
    try:
        identity = _public_identity(value, f"{operation} result", user_path=False)
    except RetentionError as error:
        raise TransportResultError(f"{operation} returned no exact identity") from error
    if identity != expected:
        raise TransportResultError(f"{operation} returned wrong pathname/size/ETag")


@dataclass(frozen=True)
class _Lease:
    store_id: str
    pathname: str
    size: int
    etag: str
    writer_id: str
    run_id: str
    expires_at: int

    @property
    def identity(self) -> _Identity:
        return _Identity(self.pathname, self.size, self.etag)


class _LeaseSession:
    """ETag-capability lease; renewed before every page, authorization, and mutation."""

    def __init__(
        self,
        transport: BlobTransport,
        clock: Callable[[], int],
        plan: _Plan,
        *,
        ttl: int,
        margin: int,
    ) -> None:
        if type(ttl) is not int or type(margin) is not int or not (0 <= margin < ttl <= 600):
            raise LeaseError("invalid lease TTL/margin")
        self.transport, self.clock, self.ttl, self.margin = transport, clock, ttl, margin
        self.lease = self._acquire(plan)

    def _now(self) -> int:
        try:
            now = self.clock()
        except Exception as error:
            raise LeaseError("clock failed") from error
        if type(now) is not int or now < 0:
            raise LeaseError("clock must return an exact nonnegative int")
        return now

    @staticmethod
    def _pathname(store_id: str) -> str:
        return (
            f"{CONTROL_PREFIX}leases/"
            f"{hashlib.sha256(store_id.encode()).hexdigest()}.json"
        )

    @staticmethod
    def _epoch(value: int) -> str:
        if value >= 10**20:
            raise LeaseError("clock exceeds fixed-width lease time")
        return f"{value:020d}"

    def _body(self, plan: _Plan, expires: int) -> bytes:
        return _json_bytes(
            {
                "schema": LEASE_SCHEMA,
                "store_id": plan.store_id,
                "writer_id": plan.writer_id,
                "run_id": plan.run_id,
                "expires_at": self._epoch(expires),
            }
        )

    def _existing(
        self, result: object, pathname: str, store_id: str
    ) -> tuple[dict[str, object], _Identity]:
        if type(result) is not tuple or len(result) != 2 or type(result[0]) is not bytes:
            raise LeaseError("existing lease read is malformed")
        raw, public_identity = result
        identity = _put_result(public_identity, pathname, len(raw), "lease read")
        try:
            record = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LeaseError("existing lease body is malformed") from error
        fields = {"schema", "store_id", "writer_id", "run_id", "expires_at"}
        if (
            type(record) is not dict
            or set(record) != fields
            or record.get("schema") != LEASE_SCHEMA
            or record.get("store_id") != store_id
            or type(record.get("expires_at")) is not str
            or len(record["expires_at"]) != 20
            or not record["expires_at"].isdigit()
        ):
            raise LeaseError("existing lease body is malformed")
        return dict(record), identity

    def _acquire(self, plan: _Plan) -> _Lease:
        now = self._now()
        expires = now + self.ttl
        pathname = self._pathname(plan.store_id)
        body = self._body(plan, expires)
        deadline = expires - self.margin
        result = self.transport.put_if_absent(
            plan.store_id, pathname, body, deadline_epoch=deadline
        )
        if result is None:
            read = self.transport.get_blob(
                plan.store_id, pathname, deadline_epoch=deadline
            )
            record, prior = self._existing(read, pathname, plan.store_id)
            if int(record["expires_at"]) > now:  # type: ignore[arg-type]
                raise LeaseError("exclusive lease is already held")
            result = self.transport.put_if_match(
                plan.store_id,
                pathname,
                body,
                etag=prior.etag,
                deadline_epoch=deadline,
            )
            if result is None:
                raise LeaseError("lease takeover lost its ETag race")
        identity = _put_result(result, pathname, len(body), "lease acquisition")
        lease = _Lease(
            plan.store_id,
            pathname,
            identity.size,
            identity.etag,
            plan.writer_id,
            plan.run_id,
            expires,
        )
        try:
            self._bound(lease)
        except LeaseError as original:
            try:
                cleanup = self.transport.delete_if_match(
                    lease.store_id,
                    lease.pathname,
                    etag=lease.etag,
                    deadline_epoch=lease.expires_at,
                )
                _delete_result(cleanup, lease.identity, "late lease-create cleanup")
            except Exception as cleanup_error:
                original.cleanup_error = (  # type: ignore[attr-defined]
                    "exact ETag lease cleanup was not proven: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        return lease

    def _renew_body(self, expires: int) -> bytes:
        return _json_bytes(
            {
                "schema": LEASE_SCHEMA,
                "store_id": self.lease.store_id,
                "writer_id": self.lease.writer_id,
                "run_id": self.lease.run_id,
                "expires_at": self._epoch(expires),
            }
        )

    def renew(self) -> None:
        now = self._now()
        if now + self.margin >= self.lease.expires_at:
            raise LeaseError("lease expired or crossed renewal margin")
        expires = now + self.ttl
        body = self._renew_body(expires)
        if len(body) != self.lease.size:
            raise LeaseError("lease renewal changed accounted bytes")
        result = self.transport.put_if_match(
            self.lease.store_id,
            self.lease.pathname,
            body,
            etag=self.lease.etag,
            deadline_epoch=self.lease.expires_at - self.margin,
        )
        if result is None:
            raise LeaseError("lease renewal lost ETag capability")
        identity = _put_result(
            result, self.lease.pathname, self.lease.size, "lease renewal"
        )
        self.lease = _Lease(
            self.lease.store_id,
            self.lease.pathname,
            identity.size,
            identity.etag,
            self.lease.writer_id,
            self.lease.run_id,
            expires,
        )
        self._bound(self.lease)

    def _bound(self, lease: _Lease) -> None:
        if self._now() + self.margin >= lease.expires_at:
            raise LeaseError("lease crossed operation-time margin")

    @property
    def deadline(self) -> int:
        return self.lease.expires_at - self.margin

    def prove_after_call(self) -> None:
        self._bound(self.lease)

    def release(self) -> None:
        # One consolidated path: renew when possible, then conditionally delete
        # the latest ETag.  A successor's different ETag can never be deleted.
        try:
            self.renew()
        except LeaseError:
            pass
        result = self.transport.delete_if_match(
            self.lease.store_id,
            self.lease.pathname,
            etag=self.lease.etag,
            deadline_epoch=self.lease.expires_at,
        )
        try:
            _delete_result(result, self.lease.identity, "lease release")
        except RetentionError as error:
            raise LeaseError("lease release lost exact ETag capability") from error


@dataclass(frozen=True)
class _Inventory:
    store_id: str
    acquired_at: int
    blobs: tuple[_Identity, ...]
    sha256: str

    @property
    def retained_bytes(self) -> int:
        return sum(item.size for item in self.blobs)

    def paths(self) -> dict[str, _Identity]:
        return {item.pathname: item for item in self.blobs}


def _page(value: object, store_id: str) -> tuple[tuple[_Identity, ...], bool, str | None]:
    if type(value) is not dict:
        raise InventoryError("page must be an exact dict")
    page = dict(value)
    if type(page.get("storeId")) is not str or page["storeId"] != store_id:
        raise InventoryError("page store mismatch")
    raw_blobs, more, cursor = page.get("blobs"), page.get("hasMore"), page.get("cursor")
    if type(raw_blobs) is not list or len(raw_blobs) > MAX_PAGE_SIZE or type(more) is not bool:
        raise InventoryError("malformed page")
    snapshot = tuple(raw_blobs)
    if more and not snapshot:
        raise InventoryError("ambiguous empty intermediate page")
    blobs: list[_Identity] = []
    for raw in snapshot:
        if type(raw) is not dict:
            raise InventoryError("blob entry must be an exact dict")
        item = dict(raw)
        try:
            pathname, size, etag = item["pathname"], item["size"], item["etag"]
            _path(pathname, control=True)
            _size(size)
            _text(etag, "etag")
        except (KeyError, RetentionError) as error:
            raise InventoryError("malformed blob entry") from error
        blobs.append(_Identity(pathname, size, etag))
    if cursor is not None and type(cursor) is not str:
        raise InventoryError("cursor must be exact str")
    return tuple(blobs), more, cursor


def _inventory(
    transport: BlobTransport,
    store_id: str,
    acquired_at: int,
    lease: _LeaseSession,
) -> _Inventory:
    cursor: str | None = None
    seen: set[str] = set()
    blobs: list[_Identity] = []
    while True:
        lease.renew()
        response = transport.list_page(
            store_id, cursor=cursor, limit=MAX_PAGE_SIZE, deadline_epoch=lease.deadline
        )
        lease.prove_after_call()
        page_blobs, more, next_cursor = _page(response, store_id)
        blobs.extend(page_blobs)
        if not more:
            break
        if not next_cursor or next_cursor in seen:
            raise InventoryError("missing or repeated cursor")
        seen.add(next_cursor)
        cursor = next_cursor
    ordered = sorted(blobs, key=lambda item: item.pathname)
    if len({item.pathname for item in ordered}) != len(ordered):
        raise InventoryError("duplicate inventory pathname")
    listed_lease = [item for item in ordered if item.pathname == lease.lease.pathname]
    if len(listed_lease) != 1 or listed_lease[0].size != lease.lease.size:
        raise InventoryError("inventory does not contain the exact visible lease bytes")
    ordered = [
        item for item in ordered if item.pathname != lease.lease.pathname
    ] + [lease.lease.identity]
    ordered.sort(key=lambda item: item.pathname)
    payload = {"store_id": store_id, "blobs": [item.object() for item in ordered]}
    return _Inventory(store_id, acquired_at, tuple(ordered), _digest(payload))


@dataclass(frozen=True)
class _Projection:
    state: ResultState
    current: int
    peak: int
    final: int


def _project(inventory: _Inventory, plan: _Plan, lease: _Identity) -> _Projection:
    existing = inventory.paths()
    if existing.get(lease.pathname) != lease:
        raise InventoryError("inventory does not bind latest lease identity")
    for write in plan.writes:
        prior = existing.get(write.pathname)
        if write.mode is WriteMode.NEW_IMMUTABLE and prior is not None:
            raise PlanError(f"new immutable collision: {write.pathname}")
        if write.mode is WriteMode.OVERWRITE and prior is None:
            raise PlanError(f"overwrite is missing: {write.pathname}")
    for deletion in plan.deletions:
        if existing.get(deletion.pathname) != deletion:
            raise PlanError(f"deletion identity mismatch: {deletion.pathname}")

    current = inventory.retained_bytes
    retained, peak = current, current
    for write in plan.writes:
        prior = existing.get(write.pathname)
        retained += write.size - (prior.size if prior else 0)
        peak = max(peak, retained)
    for deletion in plan.deletions:
        retained -= deletion.size
    persistent_before = current - lease.size
    final = retained - lease.size
    if final < 0:
        raise PlanError("projected retained bytes became negative")

    if plan.writes and (current >= HARD_STOP_BYTES or peak >= HARD_STOP_BYTES):
        raise HardStopError(current, peak, "writes are forbidden at the hard stop")
    if current >= HARD_STOP_BYTES:
        if not plan.deletions or final >= persistent_before:
            raise HardStopError(
                current, peak, "hard-stop recovery must be nonempty and strictly decreasing"
            )
        return _Projection(ResultState.RECOVERY, current, peak, final)
    if peak >= HARD_STOP_BYTES:
        # Defensive: the only way to reach this branch would be a future new
        # operation kind not covered by the write rule above.
        raise HardStopError(current, peak, "projected peak reaches the hard stop")
    state = (
        ResultState.WARNING
        if peak >= WARNING_BYTES
        else ResultState.OVER_TARGET
        if peak >= TARGET_BYTES
        else ResultState.NORMAL
    )
    return _Projection(state, current, peak, final)


class RetentionExecutor:
    """Seal, inventory, project, and execute one retention plan."""

    def __init__(
        self,
        transport: BlobTransport,
        *,
        clock: Callable[[], int] | None = None,
        lease_ttl_seconds: int = LEASE_TTL_SECONDS,
        lease_margin_seconds: int = LEASE_MARGIN_SECONDS,
    ) -> None:
        self.transport = transport
        self.clock = clock or (lambda: int(time.time()))
        self.ttl, self.margin = lease_ttl_seconds, lease_margin_seconds

    def execute(self, plan: RetentionPlan) -> RetentionResult:
        sealed = _seal_plan(plan)
        lease = _LeaseSession(
            self.transport,
            self.clock,
            sealed,
            ttl=self.ttl,
            margin=self.margin,
        )
        try:
            acquired_at = lease._now()
            inventory = _inventory(
                self.transport, sealed.store_id, acquired_at, lease
            )
            lease.renew()
            # Renewal changes only ETag, not accounted bytes.
            ordered = [
                item
                for item in inventory.blobs
                if item.pathname != lease.lease.pathname
            ] + [lease.lease.identity]
            ordered.sort(key=lambda item: item.pathname)
            inventory = _Inventory(
                inventory.store_id,
                inventory.acquired_at,
                tuple(ordered),
                _digest(
                    {
                        "store_id": inventory.store_id,
                        "blobs": [item.object() for item in ordered],
                    }
                ),
            )
            now = lease._now()
            age = now - inventory.acquired_at
            if age < 0 or age > MAX_INVENTORY_AGE_SECONDS:
                raise InventoryError("inventory is future-dated or stale")
            projection = _project(inventory, sealed, lease.lease.identity)
        except Exception:
            self._release_preflight(lease)
            raise

        statuses: list[OperationStatus] = []
        actual: int | None = projection.current
        existing = inventory.paths()
        try:
            for write in sealed.writes:
                lease.renew()
                prior = existing.get(write.pathname)
                if write.mode is WriteMode.NEW_IMMUTABLE:
                    response = self.transport.put_if_absent(
                        sealed.store_id,
                        write.pathname,
                        write.content,
                        deadline_epoch=lease.deadline,
                    )
                else:
                    assert prior is not None
                    response = self.transport.put_if_match(
                        sealed.store_id,
                        write.pathname,
                        write.content,
                        etag=prior.etag,
                        deadline_epoch=lease.deadline,
                    )
                identity = _put_result(
                    response, write.pathname, write.size, "planned write"
                )
                statuses.append(OperationStatus("write", identity.pathname, True))
                actual += write.size - (prior.size if prior else 0)
                lease.prove_after_call()
            for deletion in sealed.deletions:
                lease.renew()
                response = self.transport.delete_if_match(
                    sealed.store_id,
                    deletion.pathname,
                    etag=deletion.etag,
                    deadline_epoch=lease.deadline,
                )
                _delete_result(response, deletion, "planned deletion")
                statuses.append(OperationStatus("delete", deletion.pathname, True))
                actual -= deletion.size
                lease.prove_after_call()
        except Exception as error:
            statuses.append(
                OperationStatus(
                    "lease_guard" if isinstance(error, LeaseError) else "mutation",
                    "<lease>" if isinstance(error, LeaseError) else self._next(sealed, statuses),
                    False,
                    str(error),
                )
            )
            if not isinstance(error, LeaseError):
                actual = None
            return self._finish(
                projection, sealed, inventory, statuses, False, actual, lease
            )
        return self._finish(
            projection, sealed, inventory, statuses, True, actual, lease
        )

    @staticmethod
    def _next(plan: _Plan, statuses: list[OperationStatus]) -> str:
        paths = [item.pathname for item in plan.writes] + [
            item.pathname for item in plan.deletions
        ]
        completed = len([item for item in statuses if item.success])
        return paths[completed] if completed < len(paths) else "<mutation>"

    @staticmethod
    def _release_preflight(lease: _LeaseSession) -> None:
        try:
            lease.release()
        except Exception as error:
            raise LeaseError("preflight failed and lease release was not proven") from error

    @staticmethod
    def _finish(
        projection: _Projection,
        plan: _Plan,
        inventory: _Inventory,
        statuses: list[OperationStatus],
        success: bool,
        actual: int | None,
        lease: _LeaseSession,
    ) -> RetentionResult:
        lease_size = lease.lease.size
        try:
            lease.release()
        except Exception as error:
            statuses.append(OperationStatus("lease_release", lease.lease.pathname, False, str(error)))
            return RetentionResult(
                projection.state,
                POLICY_SHA256,
                plan.sha256,
                inventory.sha256,
                projection.current,
                projection.peak,
                projection.final,
                None,
                tuple(statuses),
                False,
                False,
            )
        if actual is not None:
            actual -= lease_size
        return RetentionResult(
            projection.state,
            POLICY_SHA256,
            plan.sha256,
            inventory.sha256,
            projection.current,
            projection.peak,
            projection.final,
            actual,
            tuple(statuses),
            success,
            True,
        )
