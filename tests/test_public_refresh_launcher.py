"""Behaviour tests for the self-healing public-refresh launcher preamble.

The launcher is zsh, so these tests drive the real script in a scratch clone
pair instead of reimplementing its logic. Every scenario stops at the same
place: ``ops/verify-public-refresh-env.sh`` is the first command the launcher
runs after it has reaped locks, synced the checkout, reinstalled itself and
derived ``SCRYGLASS_WORKER_COMMIT``, so a stub that reports the derived commit
and exits with a distinctive status cuts the run exactly at the end of the
self-healing preamble and never reaches Keychain, Brave or Supabase.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

GIT = "/usr/bin/git"
REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_SOURCE = REPO_ROOT / "ops/launchd/run-public-refresh.sh"

PREAMBLE_STOP = 91
VERIFY_STUB = f"""#!/bin/sh
echo "PREAMBLE_REACHED commit=${{SCRYGLASS_WORKER_COMMIT}}"
exit {PREAMBLE_STOP}
"""

NULL_COMMIT = "0" * 40


@dataclass
class Worker:
    root: Path
    repo: Path
    origin: Path
    authoring: Path
    launcher: Path
    lock: Path


def _git(*args: str, cwd: Path) -> str:
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "Scryglass Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Scryglass Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        }
    )
    result = subprocess.run(
        [GIT, *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _head(repo: Path) -> str:
    return _git("rev-parse", "--verify", "HEAD", cwd=repo)


def _dead_pid() -> int:
    for candidate in range(99000, 90000, -1):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except PermissionError:
            continue
    raise AssertionError("no dead pid available")


@pytest.fixture
def worker(tmp_path: Path) -> Worker:
    origin = tmp_path / "origin.git"
    authoring = tmp_path / "authoring"
    # A space in the worker root mirrors the installed path and catches quoting
    # regressions in the new preamble.
    root = tmp_path / "Scryglass Worker"
    repo = root / "repo"

    _git("init", "--bare", "--initial-branch=main", str(origin), cwd=tmp_path)
    _git("clone", str(origin), str(authoring), cwd=tmp_path)

    launchd_dir = authoring / "ops/launchd"
    launchd_dir.mkdir(parents=True)
    shutil.copy2(LAUNCHER_SOURCE, launchd_dir / "run-public-refresh.sh")
    verifier = authoring / "ops/verify-public-refresh-env.sh"
    verifier.write_text(VERIFY_STUB, encoding="utf-8")
    verifier.chmod(0o755)
    (authoring / "marker.txt").write_text("first\n", encoding="utf-8")
    _git("add", "ops", "marker.txt", cwd=authoring)
    _git("commit", "-m", "seed worker checkout", cwd=authoring)
    _git("push", "origin", "main", cwd=authoring)

    root.mkdir(parents=True, exist_ok=True)
    _git("clone", str(origin), str(repo), cwd=tmp_path)

    launcher = root / "run-public-refresh.sh"
    shutil.copy2(LAUNCHER_SOURCE, launcher)
    launcher.chmod(0o700)

    lock = root / "runtime/data/lol/runtime/public-refresh-worker.lock"
    lock.parent.mkdir(parents=True)
    return Worker(
        root=root,
        repo=repo,
        origin=origin,
        authoring=authoring,
        launcher=launcher,
        lock=lock,
    )


def _run(worker: Worker, *args: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for key in ("SCRYGLASS_WORKER_COMMIT", "SCRYGLASS_LAUNCHER_REEXEC"):
        env.pop(key, None)
    env["HOME"] = str(worker.root.parent)
    env["SCRYGLASS_WORKER_ROOT"] = str(worker.root)
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_SYSTEM"] = "/dev/null"
    env.update(env_extra)
    return subprocess.run(
        [str(worker.launcher), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _advance_origin(worker: Worker, path: str, body: str) -> str:
    target = worker.authoring / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    if path.endswith(".sh"):
        target.chmod(0o755)
    _git("add", path, cwd=worker.authoring)
    _git("commit", "-m", f"advance {path}", cwd=worker.authoring)
    _git("push", "origin", "main", cwd=worker.authoring)
    return _head(worker.authoring)


def test_preamble_derives_the_worker_commit_from_head(worker: Worker) -> None:
    head = _head(worker.repo)

    result = _run(worker)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert f"PREAMBLE_REACHED commit={head}" in result.stdout


def test_clean_checkout_syncs_to_origin_main_and_rebinds_the_commit(worker: Worker) -> None:
    before = _head(worker.repo)
    after = _advance_origin(worker, "marker.txt", "second\n")
    assert before != after

    result = _run(worker)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert _head(worker.repo) == after
    assert f"syncing worker checkout {before} to {after}" in result.stdout
    # The derived commit is the synced HEAD, which is what makes the
    # refresh_ledger.worker_commit check hold without an operator re-pinning it.
    assert f"PREAMBLE_REACHED commit={after}" in result.stdout


def test_dirty_checkout_is_never_reset_and_still_fails_the_run(worker: Worker) -> None:
    before = _head(worker.repo)
    after = _advance_origin(worker, "marker.txt", "second\n")
    local_edit = "work in progress\n"
    (worker.repo / "marker.txt").write_text(local_edit, encoding="utf-8")

    result = _run(worker)

    assert result.returncode == 78
    assert _head(worker.repo) == before
    assert (worker.repo / "marker.txt").read_text(encoding="utf-8") == local_edit
    assert f"not syncing {before} to {after}" in result.stderr
    assert "The worker checkout contains uncommitted files." in result.stderr
    assert "PREAMBLE_REACHED" not in result.stdout


def test_untracked_file_also_blocks_the_sync(worker: Worker) -> None:
    before = _head(worker.repo)
    _advance_origin(worker, "marker.txt", "second\n")
    (worker.repo / "stray.txt").write_text("untracked\n", encoding="utf-8")

    result = _run(worker)

    assert result.returncode == 78
    assert _head(worker.repo) == before
    assert (worker.repo / "stray.txt").exists()


def test_failed_fetch_continues_on_the_current_commit(worker: Worker) -> None:
    head = _head(worker.repo)
    _git(
        "remote",
        "set-url",
        "origin",
        str(worker.root.parent / "no-such-remote.git"),
        cwd=worker.repo,
    )

    result = _run(worker)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert "could not fetch origin/main; continuing on the current checkout." in result.stderr
    assert _head(worker.repo) == head
    assert f"PREAMBLE_REACHED commit={head}" in result.stdout


def test_stale_lock_is_reaped(worker: Worker) -> None:
    dead = _dead_pid()
    worker.lock.write_text(f"{dead}\n", encoding="utf-8")

    result = _run(worker)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert f"reaped stale lock from pid {dead}" in result.stdout
    assert not worker.lock.exists()


def test_unreadable_lock_is_reaped(worker: Worker) -> None:
    # shlock refuses an empty or malformed lock file forever, so the launcher
    # has to decide about it rather than hand it back to shlock.
    worker.lock.write_text("", encoding="utf-8")

    result = _run(worker)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert "reaped unreadable lock" in result.stdout
    assert not worker.lock.exists()


def test_live_lock_is_respected_and_never_removed(worker: Worker) -> None:
    owner = os.getpid()
    contents = f"{owner}\n"
    worker.lock.write_text(contents, encoding="utf-8")

    result = _run(worker)

    assert result.returncode == 75
    assert f"owns {worker.lock} (pid {owner})" in result.stderr
    assert worker.lock.read_text(encoding="utf-8") == contents
    assert "PREAMBLE_REACHED" not in result.stdout


def test_unreadable_lock_is_refused_rather_than_reaped(worker: Worker) -> None:
    # Staleness cannot be established without reading the owner, and reaping on
    # a guess could remove a live owner's lock. Refuse with a named path.
    worker.lock.write_text("4242\n", encoding="utf-8")
    worker.lock.chmod(0o000)

    try:
        result = _run(worker)
    finally:
        worker.lock.chmod(0o600)

    assert result.returncode == 75
    assert f"Cannot read {worker.lock} to identify its owner." in result.stderr
    assert worker.lock.exists()


def test_zero_pid_lock_is_not_treated_as_a_live_owner(worker: Worker) -> None:
    # kill(0, 0) addresses the caller's whole process group and would report a
    # dead owner as live, which would wedge the worker permanently.
    worker.lock.write_text("0\n", encoding="utf-8")

    result = _run(worker)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert "reaped unreadable lock" in result.stdout
    assert not worker.lock.exists()


def test_launcher_reinstalls_itself_and_re_execs_exactly_once(worker: Worker) -> None:
    stale = LAUNCHER_SOURCE.read_text(encoding="utf-8") + "\n# stale worker copy\n"
    worker.launcher.write_text(stale, encoding="utf-8")
    worker.launcher.chmod(0o700)

    result = _run(worker)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert result.stdout.count("installed launcher from") == 1
    assert result.stdout.count("PREAMBLE_REACHED") == 1
    installed = worker.launcher.read_text(encoding="utf-8")
    assert installed == (worker.repo / "ops/launchd/run-public-refresh.sh").read_text(
        encoding="utf-8"
    )
    assert oct(worker.launcher.stat().st_mode)[-3:] == "700"


def test_launcher_update_arriving_with_the_sync_is_adopted(worker: Worker) -> None:
    updated = LAUNCHER_SOURCE.read_text(encoding="utf-8") + "\n# shipped by main\n"
    _advance_origin(worker, "ops/launchd/run-public-refresh.sh", updated)

    result = _run(worker)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert result.stdout.count("installed launcher from") == 1
    assert worker.launcher.read_text(encoding="utf-8") == updated


def test_persistent_launcher_difference_aborts_instead_of_looping(worker: Worker) -> None:
    stale = LAUNCHER_SOURCE.read_text(encoding="utf-8") + "\n# stale worker copy\n"
    worker.launcher.write_text(stale, encoding="utf-8")
    worker.launcher.chmod(0o700)

    result = _run(worker, SCRYGLASS_LAUNCHER_REEXEC="1")

    assert result.returncode == 78
    assert "still differs" in result.stderr
    assert result.stdout.count("installed launcher from") == 0
    assert worker.launcher.read_text(encoding="utf-8") == stale


def test_pinned_commit_is_still_honoured(worker: Worker) -> None:
    head = _head(worker.repo)

    result = _run(worker, SCRYGLASS_WORKER_COMMIT=head)

    assert result.returncode == PREAMBLE_STOP, result.stderr
    assert f"PREAMBLE_REACHED commit={head}" in result.stdout


def test_pin_that_disagrees_with_head_still_stops_the_run(worker: Worker) -> None:
    result = _run(worker, SCRYGLASS_WORKER_COMMIT=NULL_COMMIT)

    assert result.returncode == 78
    assert "The worker HEAD differs from SCRYGLASS_WORKER_COMMIT." in result.stderr
    assert "PREAMBLE_REACHED" not in result.stdout


def test_malformed_pin_still_stops_the_run(worker: Worker) -> None:
    result = _run(worker, SCRYGLASS_WORKER_COMMIT="not-a-commit")

    assert result.returncode == 78
    assert "SCRYGLASS_WORKER_COMMIT must name the tested worker commit." in result.stderr


def test_argument_validation_survives_the_new_preamble(worker: Worker) -> None:
    result = _run(worker, "--nope")

    assert result.returncode == 64
    assert "Usage: run-public-refresh.sh [--force]" in result.stderr
