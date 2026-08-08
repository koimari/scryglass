from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import pytest

import lol_kills.export.blob_retention as retention
from lol_kills.export.blob_retention import (
    CONTROL_PREFIX,
    HARD_STOP_BYTES,
    MAX_INVENTORY_AGE_SECONDS,
    TARGET_BYTES,
    WARNING_BYTES,
    BlobIdentity,
    BoundaryTypeError,
    HardStopError,
    InventoryError,
    LeaseError,
    PlanError,
    PlannedWrite,
    ResultState,
    RetentionExecutor,
    RetentionPlan,
    WriteMode,
)


STORE, WRITER, RUN = "store-a", "writer-a", "run-a"


class Clock:
    def __init__(self, value=100):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@dataclass
class Stored:
    size: int
    etag: str
    content: bytes | None = None


class FakeTransport:
    def __init__(self, blobs=(), page=None):
        self.objects = {
            item.pathname: Stored(item.size, item.etag) for item in blobs
        }
        self.page = page
        self.calls = []
        self.counter = 0
        self.renewals = 0
        self.lose_renewal = None
        self.fail_put = set()
        self.fail_delete = set()
        self.put_result = {}
        self.delete_result = {}
        self.after_list = None
        self.after_mutation = None
        self.corrupt = set()

    def _etag(self):
        self.counter += 1
        return f"e-{self.counter}"

    def identities(self):
        return [
            BlobIdentity(path, item.size, item.etag)
            for path, item in sorted(self.objects.items())
        ]

    def list_page(self, store_id, *, cursor, limit, deadline_epoch):
        self.calls.append(("list", cursor, deadline_epoch))
        default = {
            "storeId": store_id,
            "blobs": [
                {"pathname": x.pathname, "size": x.size, "etag": x.etag}
                for x in self.identities()
            ],
            "hasMore": False,
        }
        response = self.page(self, cursor, default) if self.page else default
        if self.after_list:
            self.after_list()
        return response

    def get_blob(self, store_id, pathname, *, deadline_epoch):
        self.calls.append(("get", pathname))
        item = self.objects.get(pathname)
        if not item or item.content is None:
            return None
        return item.content, BlobIdentity(pathname, item.size, item.etag)

    def put_if_absent(self, store_id, pathname, content, *, deadline_epoch):
        self.calls.append(("create", pathname))
        if pathname in self.objects or pathname in self.fail_put:
            return None
        identity = BlobIdentity(pathname, len(content), self._etag())
        saved = b"?" * len(content) if pathname in self.corrupt else content
        self.objects[pathname] = Stored(len(content), identity.etag, saved)
        if self.after_mutation:
            self.after_mutation(pathname)
        return self.put_result.get(pathname, identity)

    def put_if_match(
        self, store_id, pathname, content, *, etag, deadline_epoch
    ):
        self.calls.append(("replace", pathname, etag))
        item = self.objects.get(pathname)
        if pathname.startswith(f"{CONTROL_PREFIX}leases/"):
            self.renewals += 1
            if self.renewals == self.lose_renewal:
                return None
        if not item or item.etag != etag or pathname in self.fail_put:
            return None
        identity = BlobIdentity(pathname, len(content), self._etag())
        saved = b"?" * len(content) if pathname in self.corrupt else content
        self.objects[pathname] = Stored(len(content), identity.etag, saved)
        if self.after_mutation:
            self.after_mutation(pathname)
        return self.put_result.get(pathname, identity)

    def delete_if_match(self, store_id, pathname, *, etag, deadline_epoch):
        self.calls.append(("delete", pathname, etag))
        item = self.objects.get(pathname)
        if not item or item.etag != etag or pathname in self.fail_delete:
            return None
        identity = BlobIdentity(pathname, item.size, item.etag)
        del self.objects[pathname]
        if self.after_mutation:
            self.after_mutation(pathname)
        return self.delete_result.get(pathname, identity)

def blob(path, size, etag="data"):
    return BlobIdentity(path, size, etag)

def plan(writes=(), deletions=(), writer=WRITER, run=RUN):
    return RetentionPlan(STORE, writer, run, tuple(writes), tuple(deletions))

def executor(transport, clock=None, ttl=120, margin=10):
    return RetentionExecutor(
        transport,
        clock=clock or Clock(),
        lease_ttl_seconds=ttl,
        lease_margin_seconds=margin,
    )

def lease_size():
    transport = FakeTransport()
    sealed = retention._seal_plan(plan())
    lease = retention._LeaseSession(
        transport, Clock(), sealed, ttl=120, margin=10
    )
    size = lease.lease.size
    lease.release()
    return size

def test_exact_arithmetic_visible_lease_and_write_before_delete_peak():
    transport = FakeTransport(
        (blob("a-old", 10, "old"), blob("z-delete", 7, "delete"))
    )
    result = executor(transport).execute(
        plan(
            (
                PlannedWrite("a-old", b"12345678", WriteMode.OVERWRITE),
                PlannedWrite("b-new", b"12345", WriteMode.NEW_IMMUTABLE),
            ),
            (blob("z-delete", 7, "delete"),),
        )
    )
    control = lease_size()
    assert result.success and result.lease_released
    assert (result.current_retained_bytes, result.peak_retained_bytes) == (
        17 + control,
        20 + control,
    )
    assert result.projected_final_bytes == result.actual_final_bytes == 13
    assert not any(path.startswith(CONTROL_PREFIX) for path in transport.objects)

@pytest.mark.parametrize(
    ("amount", "state"),
    [
        (TARGET_BYTES - 1, ResultState.NORMAL),
        (TARGET_BYTES, ResultState.OVER_TARGET),
        (WARNING_BYTES, ResultState.WARNING),
        (HARD_STOP_BYTES - 1, ResultState.WARNING),
    ],
)
def test_decimal_thresholds_include_lease(amount, state):
    result = executor(
        FakeTransport((blob("data", amount - lease_size()),))
    ).execute(plan())
    assert result.peak_retained_bytes == amount and result.state is state

def test_exact_hard_stop_allows_positive_deletion_only_recovery():
    control = lease_size()
    victim = blob("victim", 10, "victim")
    keeper = blob("keeper", HARD_STOP_BYTES - control - 10, "keeper")
    result = executor(FakeTransport((keeper, victim))).execute(
        plan(deletions=(victim,))
    )
    assert result.state is ResultState.RECOVERY
    assert result.current_retained_bytes == HARD_STOP_BYTES
    assert result.projected_final_bytes == keeper.size
    assert result.actual_final_bytes == keeper.size

def test_already_over_limit_can_recover_monotonically_without_reaching_limit():
    control = lease_size()
    victim = blob("victim", 1_000_000, "victim")
    keeper = blob("keeper", 899_000_000 - control, "keeper")
    result = executor(FakeTransport((keeper, victim))).execute(
        plan(deletions=(victim,))
    )
    assert result.current_retained_bytes == 900_000_000
    assert result.projected_final_bytes == keeper.size
    assert result.projected_final_bytes > HARD_STOP_BYTES
    assert result.state is ResultState.RECOVERY

def test_hard_stop_rejects_empty_zero_byte_or_any_write(monkeypatch):
    control = lease_size()
    base = blob("base", HARD_STOP_BYTES - control, "base")
    monkeypatch.setenv("G0_209_FORCE", "1")
    with pytest.raises(HardStopError, match="strictly decreasing"):
        executor(FakeTransport((base,))).execute(plan())

    zero = blob("zero", 0, "zero")
    with pytest.raises(HardStopError, match="strictly decreasing"):
        executor(FakeTransport((base, zero))).execute(
            plan(deletions=(zero,))
        )

    # A shrinking overwrite is still an upload and remains forbidden.
    with pytest.raises(HardStopError, match="writes are forbidden"):
        executor(FakeTransport((base,))).execute(
            plan((PlannedWrite("base", b"", WriteMode.OVERWRITE),))
        )
    assert os.environ["G0_209_FORCE"] == "1"

def test_write_peak_at_hard_stop_rejects_even_when_later_delete_would_lower():
    control = lease_size()
    victim = blob("z-victim", HARD_STOP_BYTES - control - 100, "victim")
    with pytest.raises(HardStopError) as error:
        executor(FakeTransport((victim,))).execute(
            plan(
                (PlannedWrite("a-new", b"x" * 100, WriteMode.NEW_IMMUTABLE),),
                (victim,),
            )
        )
    assert error.value.peak_retained_bytes == HARD_STOP_BYTES

def test_runtime_override_arguments_and_approval_surface_do_not_exist():
    guard, batch = executor(FakeTransport()), plan()
    with pytest.raises(TypeError):
        guard.execute(batch, force=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        guard.execute(batch, approval=object())  # type: ignore[call-arg]
    assert not hasattr(retention, "ApprovalPayload")
    assert not hasattr(retention, "TrustedHmacApprovalRegistry")

def test_one_seal_controls_object_setattr_and_concurrent_mutation():
    write = PlannedWrite("sealed", b"original", WriteMode.NEW_IMMUTABLE)
    batch = plan((write,))
    started, changed = threading.Event(), threading.Event()

    def mutate():
        assert started.wait(2)
        object.__setattr__(write, "pathname", "switched")
        object.__setattr__(write, "content", b"bad")
        object.__setattr__(batch, "writes", ())
        changed.set()

    thread = threading.Thread(target=mutate)
    thread.start()

    def page(store, cursor, default):
        started.set()
        assert changed.wait(2)
        return default

    transport = FakeTransport(page=page)
    result = executor(transport).execute(batch)
    thread.join(2)
    assert result.success and transport.objects["sealed"].content == b"original"
    assert "switched" not in transport.objects

def test_switching_plan_and_nested_subclasses_reject_before_execution():
    reads = []

    class SwitchingPlan(RetentionPlan):
        def __getattribute__(self, name):
            if name != "__class__":
                reads.append(name)
                raise AssertionError
            return super().__getattribute__(name)

    with pytest.raises(BoundaryTypeError, match="exact RetentionPlan"):
        executor(FakeTransport()).execute(object.__new__(SwitchingPlan))
    assert reads == []

    class WriteSubclass(PlannedWrite):
        pass

    hostile = object.__new__(WriteSubclass)
    for name, value in (
        ("pathname", "x"),
        ("content", b"x"),
        ("mode", WriteMode.NEW_IMMUTABLE),
    ):
        object.__setattr__(hostile, name, value)
    batch = plan()
    object.__setattr__(batch, "writes", (hostile,))
    with pytest.raises(BoundaryTypeError, match="exact PlannedWrite"):
        executor(FakeTransport()).execute(batch)

@pytest.mark.parametrize(
    "pathname",
    [
        "/a",
        "a/",
        "a//b",
        "a/../b",
        "a\\b",
        "a\x7fb",
        "cafe\u0301",
        f"{CONTROL_PREFIX}leases/x",
    ],
)
def test_path_nfc_traversal_control_and_reserved_guards(pathname):
    with pytest.raises(PlanError):
        PlannedWrite(pathname, b"x", WriteMode.NEW_IMMUTABLE)

def test_duplicate_collision_overwrite_missing_and_deletion_mismatch():
    with pytest.raises(PlanError, match="duplicate"):
        executor(FakeTransport()).execute(
            plan(
                (
                    PlannedWrite("x", b"1", WriteMode.NEW_IMMUTABLE),
                    PlannedWrite("x", b"2", WriteMode.NEW_IMMUTABLE),
                )
            )
        )
    with pytest.raises(PlanError, match="collision"):
        executor(FakeTransport((blob("x", 1),))).execute(
            plan((PlannedWrite("x", b"x", WriteMode.NEW_IMMUTABLE),))
        )
    with pytest.raises(PlanError, match="missing"):
        executor(FakeTransport()).execute(
            plan((PlannedWrite("x", b"x", WriteMode.OVERWRITE),))
        )
    hard_base = blob("base", HARD_STOP_BYTES - lease_size() - 5, "base")
    with pytest.raises(PlanError, match="mismatch"):
        executor(FakeTransport((hard_base, blob("x", 5, "real")))).execute(
            plan(deletions=(blob("x", 5, "substitute"),))
        )

def test_complete_pagination_and_failure_classes():
    def two_pages(store, cursor, default):
        lease = next(x for x in default["blobs"] if x["pathname"].startswith(CONTROL_PREFIX))
        if cursor is None:
            return {
                "storeId": STORE,
                "blobs": [lease, {"pathname": "a", "size": 1, "etag": "a"}],
                "hasMore": True,
                "cursor": "next",
            }
        return {
            "storeId": STORE,
            "blobs": [{"pathname": "b", "size": 2, "etag": "b"}],
            "hasMore": False,
        }

    transport = FakeTransport((blob("a", 1, "a"), blob("b", 2, "b")), two_pages)
    assert executor(transport).execute(plan()).projected_final_bytes == 3
    assert transport.renewals >= 4

    failures = [
        lambda s, c, d: {"storeId": "wrong", "blobs": d["blobs"], "hasMore": False},
        lambda s, c, d: {"storeId": STORE, "blobs": "bad", "hasMore": False},
        lambda s, c, d: {"storeId": STORE, "blobs": [], "hasMore": True, "cursor": "x"},
        lambda s, c, d: {
            "storeId": STORE,
            "blobs": d["blobs"] + [d["blobs"][0]],
            "hasMore": False,
        },
        lambda s, c, d: {
            "storeId": STORE,
            "blobs": d["blobs"] + [{"pathname": "bad", "size": -1, "etag": "e"}],
            "hasMore": False,
        },
    ]
    for page in failures:
        store = FakeTransport(page=page)
        with pytest.raises(InventoryError):
            executor(store).execute(plan())
        assert not any(path.startswith(CONTROL_PREFIX) for path in store.objects)

def test_repeated_cursor_and_future_or_stale_inventory_reject():
    def repeated(store, cursor, default):
        return {
            "storeId": STORE,
            "blobs": default["blobs"] if cursor is None else [
                {"pathname": "x", "size": 1, "etag": "x"}
            ],
            "hasMore": True,
            "cursor": "same",
        }

    with pytest.raises(InventoryError, match="repeated"):
        executor(FakeTransport(page=repeated)).execute(plan())

    for delta in (-1, MAX_INVENTORY_AGE_SECONDS + 1):
        clock, store = Clock(), FakeTransport()
        store.after_list = lambda delta=delta: clock.advance(delta)
        with pytest.raises(InventoryError, match="future-dated or stale"):
            executor(store, clock, ttl=600, margin=1).execute(plan())


def test_lease_second_writer_takeover_lost_renewal_and_time_bound():
    clock, store = Clock(), FakeTransport()
    one = retention._LeaseSession(
        store, clock, retention._seal_plan(plan(writer="w1", run="r1")), ttl=20, margin=2
    )
    with pytest.raises(LeaseError, match="already held"):
        retention._LeaseSession(
            store, clock, retention._seal_plan(plan(writer="w2", run="r2")), ttl=20, margin=2
        )
    old = one.lease
    clock.advance(20)
    two = retention._LeaseSession(
        store, clock, retention._seal_plan(plan(writer="w2", run="r2")), ttl=20, margin=2
    )
    one.lease = old
    with pytest.raises(LeaseError):
        one.release()
    assert store.objects[two.lease.pathname].etag == two.lease.etag
    two.release()

    store = FakeTransport()
    store.lose_renewal = 1
    with pytest.raises(LeaseError, match="renewal lost"):
        executor(store).execute(plan())

    clock, store = Clock(), FakeTransport()
    store.after_list = lambda: clock.advance(111)
    with pytest.raises(LeaseError):
        executor(store, clock).execute(plan())

    clock, store, advanced = Clock(), FakeTransport(), [False]

    def advance_during_create(pathname):
        if pathname.startswith(CONTROL_PREFIX) and not advanced[0]:
            advanced[0] = True
            clock.advance(111)

    store.after_mutation = advance_during_create
    with pytest.raises(LeaseError, match="operation-time margin"):
        executor(store, clock).execute(plan())
    assert not any(path.startswith(CONTROL_PREFIX) for path in store.objects)

    clock, store = Clock(), FakeTransport()

    def replace_etag_during_create(pathname):
        if pathname.startswith(CONTROL_PREFIX):
            store.objects[pathname].etag = "successor-etag"
            clock.advance(111)

    store.after_mutation = replace_etag_during_create
    with pytest.raises(LeaseError, match="operation-time margin") as caught:
        executor(store, clock).execute(plan())
    assert store.objects[next(iter(store.objects))].etag == "successor-etag"
    assert "cleanup was not proven" in caught.value.cleanup_error


def test_wrong_response_identity_deletion_failure_and_partial_batch_are_honest():
    for wrong in (blob("wrong", 1, "e"), blob("x", 2, "e")):
        store = FakeTransport()
        store.put_result["x"] = wrong
        result = executor(store).execute(
            plan((PlannedWrite("x", b"x", WriteMode.NEW_IMMUTABLE),))
        )
        assert not result.success and result.actual_final_bytes is None

    victim = blob("victim", 5, "victim")
    store = FakeTransport((victim,))
    store.fail_delete.add("victim")
    result = executor(store).execute(plan(deletions=(victim,)))
    assert not result.success and result.actual_final_bytes is None
    assert "victim" in store.objects

    store = FakeTransport((victim,))
    store.delete_result["victim"] = blob("other", 5, "victim")
    result = executor(store).execute(plan(deletions=(victim,)))
    assert not result.success and result.actual_final_bytes is None

    store = FakeTransport()
    store.fail_put.add("b")
    result = executor(store).execute(
        plan(
            (
                PlannedWrite("a", b"a", WriteMode.NEW_IMMUTABLE),
                PlannedWrite("b", b"b", WriteMode.NEW_IMMUTABLE),
            )
        )
    )
    assert not result.success and result.operations[0].success
    assert "a" in store.objects and "b" not in store.objects


def test_result_explicitly_does_not_claim_remote_content_integrity():
    store = FakeTransport()
    store.corrupt.add("x")
    result = executor(store).execute(
        plan((PlannedWrite("x", b"intended", WriteMode.NEW_IMMUTABLE),))
    )
    assert result.success and result.remote_content_verified is False
    assert store.objects["x"].content != b"intended"
    assert not any(call[0] == "get" for call in store.calls)
