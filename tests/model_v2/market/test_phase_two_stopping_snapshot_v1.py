from __future__ import annotations

from lol_kills.v2.market import match_winner_future_protocol_v1 as protocol
from lol_kills.v2.market import phase_two_stopping_snapshot_v1 as snapshot


def _rows(*, quoted: int, total: int, shadow: int) -> list[dict]:
    domestic = ["LCS", "LEC", "LCK", "LPL"]
    rows: list[dict] = []
    for index in range(total):
        is_quoted = index < quoted
        if index < 300:
            league = domestic[index // 75]
        else:
            league = "MSI" if index % 2 == 0 else "EWC"
        patch = "26.17" if index < 200 else "26.18" if index < 400 else "26.19"
        rows.append(
            {
                "event_id": f"event-{index}",
                "series_id": f"series-{index // 4}",
                "game_number": index % 4 + 1,
                "league": league,
                "patch": patch,
                "roster_change_stratum": "CHANGED" if index < 50 else "UNCHANGED",
                "sparse_or_new_champion_map": index < 50,
                "qualified_quote": is_quoted,
                "quote_response_too_late": False,
                "response_to_actual_start_seconds": 10.0 if is_quoted else None,
                "prediction_to_response_seconds": 2.0 if is_quoted else None,
                "failure_code": None if is_quoted else "TRANSPORT_OR_SOURCE_UNAVAILABLE",
                "shadow_signal": {
                    "exactly_one_side_qualifies": index < shadow,
                    "both_sides_qualify_inconsistent": False,
                }
                if is_quoted
                else None,
            }
        )
    return rows


def test_exact_locked_support_floor_uses_all_plans_as_coverage_denominator() -> None:
    rule = protocol._phase_two_contract()["metadata_only_stopping_rule"]
    support = snapshot._support(_rows(quoted=500, total=600, shadow=100), rule)
    assert support["eligible_quoted_maps"] == 500
    assert support["otherwise_eligible_maps"] == 600
    assert support["quote_coverage"] == 500 / 600
    assert support["eligible_series"] == 125
    assert support["shadow_policy_qualifying_maps"] == 100
    assert support["quote_received_after_map_start_maps"] == 0
    assert support["support_met"] is True
    assert support["terminal_shadow_support_failure"] is False


def test_thousand_quotes_without_shadow_support_is_terminal_failure() -> None:
    rule = protocol._phase_two_contract()["metadata_only_stopping_rule"]
    support = snapshot._support(_rows(quoted=1000, total=1000, shadow=99), rule)
    assert support["support_met"] is False
    assert support["terminal_shadow_support_failure"] is True


def test_after_start_count_is_distinct_from_five_second_safety_buffer() -> None:
    rule = protocol._phase_two_contract()["metadata_only_stopping_rule"]
    rows = _rows(quoted=500, total=600, shadow=100)
    rows[0]["quote_response_too_late"] = True
    rows[0]["response_to_actual_start_seconds"] = 3.0
    rows[1]["quote_response_too_late"] = True
    rows[1]["response_to_actual_start_seconds"] = -0.1
    support = snapshot._support(rows, rule)
    assert support["quote_response_too_late_maps"] == 2
    assert support["quote_received_after_map_start_maps"] == 1
