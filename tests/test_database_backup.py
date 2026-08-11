from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from lol_kills.database_backup import create_backup


def test_backup_is_verified_and_retention_is_bounded(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    weekly = tmp_path / "weekly"
    daily.mkdir()
    weekly.mkdir()
    for day in range(8):
        (daily / f"scryglass-2026080{day}T032000Z.dump").write_bytes(b"old")
    for week in range(5):
        (weekly / f"scryglass-2026070{week}T032000Z.dump").write_bytes(b"old")

    def run(command, **_kwargs):
        if "--file" in command:
            Path(command[command.index("--file") + 1]).write_bytes(b"x" * 2048)

    with patch("lol_kills.database_backup.subprocess.run", side_effect=run) as command:
        result = create_backup(
            tmp_path,
            database_url="postgresql://worker:secret@db.example.supabase.co:5432/postgres",
            now=datetime(2026, 8, 16, 3, 20, tzinfo=timezone.utc),
        )

    assert result["status"] == "success"
    assert result["weekly_path"]
    assert len(list(daily.glob("*.dump"))) == 7
    assert len(list(weekly.glob("*.dump"))) == 4
    assert command.call_count == 2
    dump_command = command.call_args_list[0].args[0]
    assert "--schema=public" in dump_command
    assert "--role=postgres" in dump_command
    assert "secret" not in " ".join(dump_command)
    assert command.call_args_list[0].kwargs["env"]["PGPASSWORD"] == "secret"


def test_backup_can_use_a_short_lived_supabase_cli_login(tmp_path: Path) -> None:
    login = "\n".join(
        (
            'export PGHOST="aws-0-sa-east-1.pooler.supabase.com"',
            'export PGPORT="5432"',
            'export PGUSER="cli_login_postgres.project"',
            'export PGPASSWORD="temporary-secret"',
            'export PGDATABASE="postgres"',
        )
    )

    def run(command, **kwargs):
        if "--dry-run" in command:
            return type("Result", (), {"stdout": login, "stderr": ""})()
        if "--file" in command:
            Path(command[command.index("--file") + 1]).write_bytes(b"x" * 2048)
        return type("Result", (), {"stdout": "", "stderr": ""})()

    with patch("lol_kills.database_backup.subprocess.run", side_effect=run) as command:
        result = create_backup(tmp_path, supabase_workdir=tmp_path)

    assert result["status"] == "success"
    assert command.call_count == 3
    dump = command.call_args_list[1]
    assert "temporary-secret" not in " ".join(dump.args[0])
    assert dump.kwargs["env"]["PGPASSWORD"] == "temporary-secret"
