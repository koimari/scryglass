from __future__ import annotations

from collections import Counter

from lol_kills.research.mechanics_composite import (
    CompositePrediction,
    GridCheckpointReceipt,
    InteractionKey,
    RosterInterval,
    TemporalRosterRegistry,
    evaluate_winner_gate,
    interaction_backoff_chain,
    iter_interaction_keys,
    wilson_interval,
)


SHA = "b" * 64


def _draft() -> dict[str, dict[str, str]]:
    return {
        "blue": {"top": "A", "jng": "B", "mid": "C", "bot": "D", "sup": "E"},
        "red": {"top": "F", "jng": "G", "mid": "H", "bot": "I", "sup": "J"},
    }


def test_interactions_cover_orders_one_through_nine_for_each_focal() -> None:
    keys = iter_interaction_keys(_draft())
    assert len(keys) == 10 * (2**9 - 1)
    counts = Counter(key.order for key in keys)
    assert counts == {order: 10 * __import__("math").comb(9, order - 1) for order in range(1, 10)}


def test_backoff_never_replaces_an_exact_key_with_a_number() -> None:
    item = InteractionKey("top", "A", 3, (("jng", "B"),), (("mid", "H"),))
    chain = interaction_backoff_chain(item)
    assert chain[0] == item.key
    assert len(chain) > 1
    assert all(isinstance(value, str) for value in chain)


def test_interaction_key_is_patch_and_effect_scoped() -> None:
    item = InteractionKey(
        "top",
        "A",
        2,
        (("jng", "B"),),
        (),
        "26.13",
        "pregame",
        "damage",
        "single_target",
    )
    assert "patch=26.13" in item.key
    assert "effect=damage" in item.key
    assert "target=single_target" in item.key
    assert interaction_backoff_chain(item)[-1].startswith("IH|patch=26.13|")


def test_roster_interval_switches_from_starter_to_substitute_without_retroactive_leakage() -> None:
    registry = TemporalRosterRegistry()
    base = "2026-07-01T00:00:00Z"
    for role, player in (("top", "top-a"), ("mid", "mid-a"), ("bot", "bot-a"), ("sup", "sup-a")):
        registry.add(RosterInterval("FLY", player, role, base, None, base, "starter", SHA))
    registry.add(RosterInterval("FLY", "Inspired", "jng", base, "2026-07-10T00:00:00Z", base, "starter", SHA))
    registry.add(RosterInterval("FLY", "Armao", "jng", "2026-07-10T00:00:00Z", None, "2026-07-09T00:00:00Z", "substitute", SHA))
    before = registry.resolve(fixture_id="before", team_id="FLY", as_of="2026-07-09T12:00:00Z", event_start="2026-07-09T13:00:00Z")
    after = registry.resolve(fixture_id="after", team_id="FLY", as_of="2026-07-10T12:00:00Z", event_start="2026-07-10T13:00:00Z")
    assert dict(before.players)["jng"] == "Inspired"
    assert dict(after.players)["jng"] == "Armao"


def test_grid_checkpoint_fails_closed_on_stale_or_gapped_state() -> None:
    receipt = GridCheckpointReceipt("series", "game", 1000, 4, "2026-07-31T12:00:00Z", SHA, SHA, True, sequence_gap=True)
    eligible, blockers = receipt.validate_checkpoint(cutoff_game_time_ms=2000, maximum_age_ms=5000)
    assert not eligible
    assert blockers == ("grid_sequence_gap",)
    eligible, blockers = receipt.validate_checkpoint(cutoff_game_time_ms=7000, maximum_age_ms=5000)
    assert not eligible
    assert "grid_state_stale" in blockers


def test_grid_receipt_rejects_observation_after_pregame_cutoff() -> None:
    receipt = GridCheckpointReceipt(
        "series",
        "game",
        1000,
        4,
        "2026-07-31T12:00:01Z",
        SHA,
        SHA,
        True,
    )
    eligible, blockers = receipt.validate_checkpoint(
        cutoff_game_time_ms=1000,
        maximum_age_ms=5000,
        cutoff_observed_at="2026-07-31T12:00:00Z",
        maximum_observed_age_ms=5000,
    )
    assert not eligible
    assert blockers == ("grid_observation_after_cutoff",)
    assert len(receipt.receipt_sha256) == 64


def test_winner_gate_counts_unavailable_as_non_success() -> None:
    available = CompositePrediction("g1", "2026-07-31T12:00:00Z", "blue", 0.7, 0.2, 0.1, {}, "available", (), "test", SHA)
    unavailable = CompositePrediction("g2", "2026-07-31T12:00:00Z", None, None, None, None, {}, "unavailable", ("blocked",), "test", SHA)
    report = evaluate_winner_gate([available, unavailable], {"g1": "blue", "g2": "red"})
    assert report["total_verified_games"] == 2
    assert report["correct_primary_gate"] == 1
    assert report["primary_accuracy"] == 0.5
    assert report["coverage"] == 0.5


def test_wilson_interval_has_five_percent_gate_shape() -> None:
    lower, upper, half = wilson_interval(80, 100)
    assert lower < 0.8 < upper
    assert half > 0.05
