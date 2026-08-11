from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lol_kills.etl import oe_database


ROLES = ("top", "jng", "mid", "bot", "sup")


class FakeDatabase:
    def __init__(self, current: dict[str, str] | None = None) -> None:
        self.current = dict(current or {})
        self.versions: dict[tuple[str, str], dict] = {}
        self.imports: list[dict] = []

    def current_hashes(self, _year: int) -> dict[str, str]:
        return dict(self.current)

    def import_receipt(
        self, year: int, source_file_sha256: str, transform_version: str
    ) -> dict | None:
        return next(
            (
                row
                for row in self.imports
                if row["source_year"] == year
                and row["source_file_sha256"] == source_file_sha256
                and row["transform_version"] == transform_version
            ),
            None,
        )

    def append_versions(self, rows: list[dict]) -> None:
        for row in rows:
            self.versions[(row["canonical_game_id"], row["payload_sha256"])] = row

    def upsert_current(self, rows: list[dict]) -> None:
        for row in rows:
            key = (row["canonical_game_id"], row["payload_sha256"])
            assert key in self.versions
            self.current[row["canonical_game_id"]] = row["payload_sha256"]

    def record_import(self, row: dict) -> None:
        self.imports.append(row)


def _game_rows(game_id: str, date: str, *, player_count: int = 10) -> list[dict]:
    rows: list[dict] = []
    players = []
    for side_index, side in enumerate(("Blue", "Red")):
        result = 1 if side == "Blue" else 0
        team = f"Team {side}"
        for role_index, role in enumerate(ROLES):
            players.append(
                {
                    "gameid": game_id,
                    "league": "LCK",
                    "date": date,
                    "patch": "16.15",
                    "side": side,
                    "position": role,
                    "teamname": team,
                    "playername": f"{side}-{role}",
                    "champion": f"Champion-{side_index}-{role_index}",
                    "result": result,
                    "kills": role_index + 1,
                    "deaths": 2,
                    "assists": 8,
                    "teamkills": 20,
                    "gamelength": 1800,
                    "dpm": 400 + role_index,
                    "damageshare": 0.2,
                    "totalgold": 9000 + role_index,
                    "cspm": 6.0,
                    "wpm": 0.5,
                    "wcpm": 0.2,
                    "datacompleteness": "complete",
                    "padding": "x" * 500,
                }
            )
        rows.append(
            {
                "gameid": game_id,
                "league": "LCK",
                "date": date,
                "patch": "16.15",
                "side": side,
                "position": "team",
                "teamname": team,
                "playername": None,
                "champion": None,
                "result": result,
                "kills": 20,
                "deaths": 10,
                "assists": 40,
                "teamkills": 20,
                "gamelength": 1800,
                "dpm": 2000,
                "damageshare": 1.0,
                "totalgold": 50000,
                "cspm": 30.0,
                "wpm": 2.5,
                "wcpm": 1.0,
                "datacompleteness": "complete",
                "padding": "x" * 500,
            }
        )
    return players[:player_count] + rows


def _write_csv(path: Path, *, malformed: bool = False) -> None:
    rows = _game_rows("game-1", "2026-08-11T10:00:00Z")
    rows.extend(
        _game_rows(
            "game-2",
            "2026-08-11T11:00:00Z",
            player_count=9 if malformed else 10,
        )
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    assert path.stat().st_size >= 10_000


def test_prepare_import_accepts_complete_games_and_quarantines_bad_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    _write_csv(path, malformed=True)

    prepared = oe_database.prepare_import(path, 2026)

    assert list(prepared.games) == ["game-1"]
    assert prepared.games["game-1"].statistics_complete is True
    assert prepared.quarantined_game_ids == ("game-2",)
    payload = oe_database.payload_for_game(prepared, "game-1")
    assert len(payload["team_rows"]) == 2
    assert len(payload["player_rows"]) == 10


def test_sync_is_incremental_idempotent_and_records_corrections(tmp_path: Path) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    parquet = tmp_path / "parquet"
    _write_csv(path)
    database = FakeDatabase()

    first = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=parquet,
        client=database,
    )
    second = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=parquet,
        client=database,
    )

    assert first["new_games"] == 2
    assert first["corrected_games"] == 0
    assert second["new_games"] == 0
    assert second["corrected_games"] == 0
    assert second["unchanged_games"] == 2
    assert second["cache"]["replaced_games"] == 0
    assert len(database.versions) == 2

    frame = pd.read_csv(path)
    frame.loc[
        frame["gameid"].eq("game-2") & frame["playername"].eq("Blue-top"),
        "kills",
    ] = 9
    frame.to_csv(path, index=False)
    corrected = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=parquet,
        client=database,
    )

    assert corrected["new_games"] == 0
    assert corrected["corrected_games"] == 1
    assert corrected["cache"]["replaced_games"] == 1
    assert len(database.versions) == 3
    players = pd.read_parquet(parquet / "oe_player_games.parquet")
    changed = players.loc[
        players["gameid"].eq("game-2") & players["playername"].eq("Blue-top")
    ]
    assert changed.iloc[0]["kills"] == 9


def test_sync_rejects_disappearance_before_database_writes(tmp_path: Path) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    _write_csv(path)
    prepared = oe_database.prepare_import(path, 2026)
    database = FakeDatabase(
        {
            **{game_id: game.payload_sha256 for game_id, game in prepared.games.items()},
            "missing-game": "0" * 64,
        }
    )

    with pytest.raises(oe_database.OeDatabaseError, match="lost 1 stored games"):
        oe_database.sync_csv(
            path,
            2026,
            project_url="https://example.supabase.co",
            secret_key="sb_secret_unused_in_fake_database",
            parquet_dir=tmp_path / "parquet",
            client=database,
        )

    assert database.versions == {}
    assert database.imports == []


def test_database_migration_keeps_oe_tables_private() -> None:
    migration_dir = (
        Path(__file__).resolve().parents[1]
        / "supabase/migrations"
    )
    migration = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(migration_dir.glob("*.sql"))
    )
    for table in (
        "scryglass_oe_game_versions",
        "scryglass_oe_games",
        "scryglass_oe_imports",
    ):
        assert f"alter table public.{table} enable row level security" in migration
        assert f"revoke all on public.{table} from public, anon, authenticated" in migration
        assert f"revoke all on public.{table} from service_role" in migration
    assert "grant select, insert on public.scryglass_oe_game_versions to service_role" in migration
    assert "scryglass_oe_games_version_fk_idx" in migration
    assert "transform_version" in migration


def test_client_repr_redacts_secret_key() -> None:
    client = oe_database.SupabaseOeDatabase(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    assert "abcdefghijklmnopqrstuvwxyz" not in repr(client)
    assert "<redacted>" in repr(client)
