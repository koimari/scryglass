from __future__ import annotations

from lol_kills.etl.riot_timelines import parse_grub_events, summarize_map_grubs


def _participants(*, red_near: bool) -> dict[str, dict]:
    frame = {}
    for participant_id in range(1, 11):
        near = participant_id <= 5 or red_near
        frame[str(participant_id)] = {
            "position": {
                "x": 5_000 if near else 12_000,
                "y": 5_000 if near else 12_000,
            }
        }
    return frame


def _timeline(frame_timestamp: int, *, red_near: bool) -> dict:
    return {
        "info": {
            "frames": [
                {
                    "timestamp": frame_timestamp,
                    "participantFrames": _participants(red_near=red_near),
                    "events": [],
                },
                {
                    "timestamp": 480_000,
                    "participantFrames": {},
                    "events": [
                        {
                            "type": "ELITE_MONSTER_KILL",
                            "monsterType": "HORDE",
                            "killerId": 1,
                            "timestamp": 480_000,
                            "position": {"x": 5_000, "y": 5_000},
                        }
                    ],
                },
            ]
        }
    }


def test_stale_position_frame_is_unknown_not_uncontested() -> None:
    timeline = _timeline(420_000, red_near=False)
    event = parse_grub_events(timeline)[0]
    assert event["contested"] is None
    assert event["contest_reason"] == "unknown_stale_or_incomplete_position_frame"
    summary = summarize_map_grubs(timeline)
    assert summary["n_uncontested"] == 0
    assert summary["n_contest_unknown"] == 1
    assert summary["any_contested"] is None


def test_complete_nearby_frame_distinguishes_verified_states() -> None:
    uncontested = parse_grub_events(_timeline(480_000, red_near=False))[0]
    contested = parse_grub_events(_timeline(480_000, red_near=True))[0]
    assert uncontested["contested"] is False
    assert contested["contested"] is True
