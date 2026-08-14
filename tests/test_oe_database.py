from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lol_kills.etl import oe_database
from lol_kills.etl.riot_patch_receipts import receipt_from_response_bytes


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
    assert prepared.quarantined_games["game-2"]
    payload = oe_database.payload_for_game(prepared, "game-1")
    assert len(payload["team_rows"]) == 2
    assert len(payload["player_rows"]) == 10


def test_prepare_import_applies_official_patch_receipt_without_date_rewrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    _write_csv(path)
    receipt = receipt_from_response_bytes(
        "game-1",
        b'{"gameMetadata":{"patchVersion":"16.16.801.5000"}}',
        retrieved_at_utc="2026-08-14T12:00:00Z",
    )

    prepared = oe_database.prepare_import(
        path,
        2026,
        patch_receipts={"game-1": receipt},
    )

    assert prepared.games["game-1"].patch == "16.16"
    assert prepared.team_rows.loc[prepared.team_rows["gameid"] == "game-1", "patch"].eq(
        "16.16"
    ).all()
    assert prepared.player_rows.loc[prepared.player_rows["gameid"] == "game-1", "patch"].eq(
        "16.16"
    ).all()


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
    assert len(first["cache"]["canonical_game_identity_digest"]) == 64
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


def test_sync_normalizes_numeric_patch_tokens_before_parquet(tmp_path: Path) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    rows = _game_rows("game-1", "2026-08-11T10:00:00Z")
    rows.extend(_game_rows("game-2", "2026-08-11T11:00:00Z"))
    for row in rows:
        row["patch"] = 16.16
    pd.DataFrame(rows).to_csv(path, index=False)

    result = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=tmp_path / "parquet",
        client=FakeDatabase(),
    )

    assert result["accepted_games"] == 2
    players = pd.read_parquet(tmp_path / "parquet/oe_player_games.parquet")
    teams = pd.read_parquet(tmp_path / "parquet/oe_team_games.parquet")
    assert str(players["patch"].dtype) == "string"
    assert str(teams["patch"].dtype) == "string"
    assert set(players["patch"].dropna()) == {"16.16"}
    assert set(teams["patch"].dropna()) == {"16.16"}


def test_sync_preserves_patch_minor_zero_before_string_normalization(tmp_path: Path) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    rows = _game_rows("game-1", "2026-08-11T10:00:00Z")
    rows.extend(_game_rows("game-2", "2026-08-11T11:00:00Z"))
    for row in rows:
        row["patch"] = "16.10"
    pd.DataFrame(rows).to_csv(path, index=False)

    result = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=tmp_path / "parquet",
        client=FakeDatabase(),
    )

    assert result["accepted_games"] == 2
    players = pd.read_parquet(tmp_path / "parquet/oe_player_games.parquet")
    assert set(players["patch"].dropna()) == {"16.10"}


def test_sync_rewrites_legacy_numeric_patch_cache_and_accepts_blank_patch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    rows = _game_rows("game-1", "2026-08-11T10:00:00Z")
    rows.extend(_game_rows("game-2", "2026-08-11T11:00:00Z"))
    for row in rows:
        row["patch"] = 16.15 if row["gameid"] == "game-1" else None
    pd.DataFrame(rows).to_csv(path, index=False)
    parquet = tmp_path / "parquet"
    database = FakeDatabase()

    result = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=parquet,
        client=database,
    )

    assert result["accepted_games"] == 2
    players = pd.read_parquet(parquet / "oe_player_games.parquet")
    teams = pd.read_parquet(parquet / "oe_team_games.parquet")
    assert str(players["patch"].dtype) == "string"
    assert str(teams["patch"].dtype) == "string"
    assert set(players["patch"].dropna()) == {"16.15"}

    # Simulate a pre-v2 cache. The next source pass must normalize it before
    # concatenating replacement rows.
    players["patch"] = players["patch"].astype(float)
    teams["patch"] = teams["patch"].astype(float)
    players.to_parquet(parquet / "oe_player_games.parquet", index=False)
    teams.to_parquet(parquet / "oe_team_games.parquet", index=False)
    frame = pd.read_csv(path)
    frame.loc[frame["gameid"].eq("game-2"), "kills"] = 9
    frame.to_csv(path, index=False)

    corrected = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=parquet,
        client=database,
    )

    assert corrected["corrected_games"] == 1
    players = pd.read_parquet(parquet / "oe_player_games.parquet")
    assert str(players["patch"].dtype) == "string"


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


def test_sync_accepts_reviewed_removed_games_and_drops_them_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    parquet = tmp_path / "parquet"
    rows = _game_rows("game-1", "2026-08-11T10:00:00Z")
    rows.extend(_game_rows("game-2", "2026-08-11T11:00:00Z"))
    rows.extend(_game_rows("game-3", "2026-08-11T12:00:00Z"))
    pd.DataFrame(rows).to_csv(path, index=False)
    database = FakeDatabase()

    first = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=parquet,
        client=database,
    )
    assert first["accepted_games"] == 3

    monkeypatch.setattr(
        oe_database, "REVIEWED_REMOVED_GAME_IDS", {"game-3": "test revision"}
    )
    rows = _game_rows("game-1", "2026-08-11T10:00:00Z")
    rows.extend(_game_rows("game-2", "2026-08-11T11:00:00Z"))
    pd.DataFrame(rows).to_csv(path, index=False)

    second = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=parquet,
        client=database,
    )

    assert second["accepted_games"] == 2
    assert second["new_games"] == 0
    players = pd.read_parquet(parquet / "oe_player_games.parquet")
    teams = pd.read_parquet(parquet / "oe_team_games.parquet")
    assert "game-3" not in set(players["gameid"].astype(str))
    assert "game-3" not in set(teams["gameid"].astype(str))


def test_sync_concurrent_upload_stores_all_games(tmp_path: Path) -> None:
    """The parallel upload path (>WRITE_BATCH_SIZE changed games) stores everything."""
    path = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    rows = []
    for index in range(120):
        rows.extend(
            _game_rows(
                f"game-{index}",
                f"2026-08-11T{10 + index // 60:02d}:{index % 60:02d}:00Z",
            )
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    assert path.stat().st_size >= 10_000
    database = FakeDatabase()
    result = oe_database.sync_csv(
        path,
        2026,
        project_url="https://example.supabase.co",
        secret_key="sb_secret_unused_in_fake_database",
        parquet_dir=tmp_path / "parquet",
        client=database,
    )
    assert result["new_games"] == 120
    assert result["accepted_games"] == 120
    assert len(database.current) == 120
    assert len(database.versions) == 120


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
    assert "riot_patch_receipts" in migration


def test_client_repr_redacts_secret_key() -> None:
    client = oe_database.SupabaseOeDatabase(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    assert "abcdefghijklmnopqrstuvwxyz" not in repr(client)
    assert "<redacted>" in repr(client)
