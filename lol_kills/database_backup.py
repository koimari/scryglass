"""Create and retain verified logical backups of the private Supabase database."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


class DatabaseBackupError(RuntimeError):
    """A logical backup could not be created and verified."""


SUPABASE_EXPORT_RE = re.compile(r'^export (PG[A-Z]+)="([^"]+)"$', re.MULTILINE)


def _database_connection(value: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(value.strip())
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.hostname
        or not parsed.username
        or not parsed.password
        or not parsed.path.strip("/")
        or not parsed.hostname.endswith((".supabase.co", ".supabase.com"))
    ):
        raise DatabaseBackupError("database URL must be a complete Supabase Postgres URL")
    username = urllib.parse.quote(urllib.parse.unquote(parsed.username), safe="")
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    safe_url = urllib.parse.urlunsplit(
        (parsed.scheme, f"{username}@{host}{port}", parsed.path, parsed.query, "")
    )
    return safe_url, urllib.parse.unquote(parsed.password)


def _supabase_cli_connection(workdir: Path) -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["npx", "supabase", "db", "dump", "--linked", "--dry-run"],
            cwd=workdir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()[:500]
        raise DatabaseBackupError(
            f"Supabase short-lived database login failed: {detail}"
        ) from error
    values = dict(SUPABASE_EXPORT_RE.findall(result.stdout))
    required = {"PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"}
    if set(values) < required:
        raise DatabaseBackupError("Supabase CLI did not return a complete short-lived login")
    username = urllib.parse.quote(values["PGUSER"], safe="")
    password = urllib.parse.quote(values["PGPASSWORD"], safe="")
    database = urllib.parse.quote(values["PGDATABASE"], safe="")
    return _database_connection(
        f"postgresql://{username}:{password}@{values['PGHOST']}:{values['PGPORT']}/{database}"
    )


def _retain(paths: list[Path], keep: int) -> list[str]:
    removed: list[str] = []
    for path in sorted(paths, key=lambda item: item.name, reverse=True)[keep:]:
        path.unlink()
        removed.append(path.name)
    return removed


def create_backup(
    destination_root: Path,
    *,
    database_url: str | None = None,
    supabase_workdir: Path | None = None,
    now: datetime | None = None,
    pg_dump: str = "pg_dump",
    pg_restore: str = "pg_restore",
) -> dict[str, object]:
    checked_at = now or datetime.now(timezone.utc)
    if database_url:
        url, password = _database_connection(database_url)
    elif supabase_workdir is not None:
        url, password = _supabase_cli_connection(supabase_workdir.resolve())
    else:
        raise DatabaseBackupError("a database URL or Supabase work directory is required")
    command_environment = {**os.environ, "PGPASSWORD": password}
    daily = destination_root.expanduser().resolve() / "daily"
    weekly = destination_root.expanduser().resolve() / "weekly"
    daily.mkdir(parents=True, exist_ok=True)
    weekly.mkdir(parents=True, exist_ok=True)
    stamp = checked_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = daily / f"scryglass-{stamp}.dump"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=daily,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                pg_dump,
                "--dbname",
                url,
                "--format=custom",
                "--schema=public",
                "--role=postgres",
                "--no-owner",
                "--no-acl",
                "--file",
                str(temporary),
            ],
            check=True,
            env=command_environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600,
        )
        if temporary.stat().st_size < 1024:
            raise DatabaseBackupError("logical backup is unexpectedly small")
        subprocess.run(
            [pg_restore, "--list", str(temporary)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
        os.replace(temporary, destination)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()[:500]
        raise DatabaseBackupError(f"logical backup command failed: {detail}") from error
    finally:
        temporary.unlink(missing_ok=True)

    weekly_path: Path | None = None
    if checked_at.astimezone(timezone.utc).weekday() == 6:
        weekly_path = weekly / destination.name
        if not weekly_path.exists():
            try:
                os.link(destination, weekly_path)
            except OSError:
                shutil.copy2(destination, weekly_path)
    removed = _retain(list(daily.glob("scryglass-*.dump")), 7)
    removed.extend(_retain(list(weekly.glob("scryglass-*.dump")), 4))
    return {
        "status": "success",
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "weekly_path": str(weekly_path) if weekly_path else None,
        "removed": removed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--supabase-workdir", type=Path)
    arguments = parser.parse_args(argv)
    database_url = os.environ.get("SCRYGLASS_DATABASE_URL", "").strip()
    result = create_backup(
        arguments.destination,
        database_url=database_url or None,
        supabase_workdir=arguments.supabase_workdir,
        pg_dump=os.environ.get("SCRYGLASS_PG_DUMP", "pg_dump"),
        pg_restore=os.environ.get("SCRYGLASS_PG_RESTORE", "pg_restore"),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
