#!/usr/bin/env python3
"""
Parse Riot esports live-stats JSONL (rfc461 schemas) into fight-environment tables.

This is NOT a .rofl binary parse. Axword-style dumps are already decoded Live Stats
events (LOLTMNT*) with ~1s participant positions inside stats_update.

  python3 -m lol_kills.etl.riot_esports_events \
    --jsonl ~/Desktop/events_2970115_1_riot.jsonl \
    --out data/lol/warehouse/esports_events/426848
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from lol_kills.etl.paths import WAREHOUSE_DIR

DEFAULT_OUT = WAREHOUSE_DIR / "esports_events"
CONTEST_RADIUS = 2200.0  # match ranked HORDE parser convention


def _mmss(ms: float | int | None) -> str:
    if ms is None:
        return "?"
    s = int(ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


def _dist(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["z"]) - float(b["z"]))


def load_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_roster(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for o in events:
        if o.get("rfc461Schema") == "game_info":
            out = []
            for p in o.get("participants") or []:
                out.append(
                    {
                        "participantID": p["participantID"],
                        "teamID": p["teamID"],
                        "role": p.get("role"),
                        "championName": p.get("championName"),
                        "summonerName": p.get("summonerName")
                        or (p.get("riotId") or {}).get("displayName"),
                    }
                )
            return sorted(out, key=lambda r: r["participantID"])
    return []


def extract_positions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (stats_update, participant) with map position."""
    rows: list[dict[str, Any]] = []
    for o in events:
        if o.get("rfc461Schema") != "stats_update":
            continue
        t = o.get("gameTime")
        if t is None:
            continue
        for p in o.get("participants") or []:
            pos = p.get("position") or {}
            if "x" not in pos or "z" not in pos:
                continue
            rows.append(
                {
                    "gameTime_ms": int(t),
                    "participantID": int(p["participantID"]),
                    "teamID": int(p["teamID"]),
                    "championName": p.get("championName"),
                    "playerName": p.get("playerName"),
                    "role": p.get("role"),
                    "alive": bool(p.get("alive", True)),
                    "x": float(pos["x"]),
                    "z": float(pos["z"]),
                    "totalGold": float(p.get("totalGold") or 0),
                    "currentGold": float(p.get("currentGold") or 0),
                    "level": int(p.get("level") or 0),
                    "health": float(p.get("health") or 0),
                    "healthMax": float(p.get("healthMax") or 0),
                }
            )
    return rows


def extract_epic_kills(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for o in events:
        if o.get("rfc461Schema") != "epic_monster_kill":
            continue
        pos = o.get("position") or {}
        rows.append(
            {
                "gameTime_ms": int(o["gameTime"]),
                "monsterType": o.get("monsterType"),
                "killer": o.get("killer"),
                "killerTeamID": o.get("killerTeamID"),
                "assistants": list(o.get("assistants") or []),
                "x": float(pos.get("x") or 0),
                "z": float(pos.get("z") or 0),
                "killType": o.get("killType"),
                "localGold": o.get("localGold"),
                "globalGold": o.get("globalGold"),
            }
        )
    return rows


def extract_champion_kills(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for o in events:
        if o.get("rfc461Schema") != "champion_kill":
            continue
        pos = o.get("position") or {}
        rows.append(
            {
                "gameTime_ms": int(o["gameTime"]),
                "killer": o.get("killer"),
                "victim": o.get("victim"),
                "assistants": list(o.get("assistants") or []),
                "x": float(pos.get("x") or 0),
                "z": float(pos.get("z") or 0),
            }
        )
    return rows


def nearest_frame(
    positions: list[dict[str, Any]], t_ms: int, max_skew_ms: int = 1500
) -> dict[int, dict[str, Any]]:
    """Map participantID -> position row closest to t_ms."""
    by_p: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in positions:
        by_p[r["participantID"]].append(r)
    out: dict[int, dict[str, Any]] = {}
    for pid, rows in by_p.items():
        best = min(rows, key=lambda r: abs(r["gameTime_ms"] - t_ms))
        if abs(best["gameTime_ms"] - t_ms) <= max_skew_ms:
            out[pid] = best
    return out


def analyze_voidgrub_window(
    positions: list[dict[str, Any]],
    epic_kills: list[dict[str, Any]],
    champ_kills: list[dict[str, Any]],
    roster: list[dict[str, Any]],
    *,
    radius: float = CONTEST_RADIUS,
    pre_ms: int = 30_000,
    post_ms: int = 15_000,
) -> dict[str, Any]:
    grubs = [k for k in epic_kills if k["monsterType"] == "VoidGrub"]
    if not grubs:
        return {"n_grubs": 0, "note": "no VoidGrub kills in feed"}

    first = min(grubs, key=lambda k: k["gameTime_ms"])
    last = max(grubs, key=lambda k: k["gameTime_ms"])
    pit = {"x": first["x"], "z": first["z"]}
    t0, t1 = first["gameTime_ms"] - pre_ms, last["gameTime_ms"] + post_ms

    roster_by_id = {r["participantID"]: r for r in roster}
    snapshots = []
    # sample every ~2s across window from available frames
    times = sorted({r["gameTime_ms"] for r in positions if t0 <= r["gameTime_ms"] <= t1})
    # thin to ~2s
    sampled: list[int] = []
    for t in times:
        if not sampled or t - sampled[-1] >= 2000:
            sampled.append(t)

    for t in sampled:
        frame = nearest_frame(positions, t)
        near = []
        for pid, row in frame.items():
            d = _dist(row, pit)
            if d <= radius:
                meta = roster_by_id.get(pid, {})
                near.append(
                    {
                        "participantID": pid,
                        "teamID": row["teamID"],
                        "championName": row.get("championName") or meta.get("championName"),
                        "playerName": row.get("playerName") or meta.get("summonerName"),
                        "role": row.get("role") or meta.get("role"),
                        "dist": round(d, 1),
                        "alive": row["alive"],
                        "totalGold": row["totalGold"],
                    }
                )
        snapshots.append(
            {
                "gameTime_ms": t,
                "clock": _mmss(t),
                "n_near": len(near),
                "n_blue": sum(1 for x in near if x["teamID"] == 100),
                "n_red": sum(1 for x in near if x["teamID"] == 200),
                "near": near,
            }
        )

    # peak contest (max both teams simultaneously near)
    contested_snaps = [s for s in snapshots if s["n_blue"] > 0 and s["n_red"] > 0]
    peak = max(snapshots, key=lambda s: (min(s["n_blue"], s["n_red"]), s["n_near"]), default=None)

    kills_in = [k for k in champ_kills if t0 <= k["gameTime_ms"] <= t1]

    # gold deltas for botlaners over window (leave-farm lens)
    bots = [r for r in roster if r.get("role") in ("Bottom", "Support")]
    gold_paths = []
    for b in bots:
        pid = b["participantID"]
        pre = nearest_frame(positions, first["gameTime_ms"] - pre_ms).get(pid)
        mid = nearest_frame(positions, first["gameTime_ms"]).get(pid)
        post = nearest_frame(positions, last["gameTime_ms"] + post_ms).get(pid)
        gold_paths.append(
            {
                **b,
                "gold_pre": None if not pre else pre["totalGold"],
                "gold_at_first_grub": None if not mid else mid["totalGold"],
                "gold_post": None if not post else post["totalGold"],
                "dist_at_first_grub": None
                if not mid
                else round(_dist(mid, pit), 1),
                "near_pit_at_first": None
                if not mid
                else _dist(mid, pit) <= radius,
            }
        )

    taking_team = {g["killerTeamID"] for g in grubs}
    return {
        "n_grubs": len(grubs),
        "taking_teamIDs": sorted(taking_team),
        "sweep": len(taking_team) == 1 and len(grubs) == 3,
        "first_grub_ms": first["gameTime_ms"],
        "first_grub_clock": _mmss(first["gameTime_ms"]),
        "last_grub_ms": last["gameTime_ms"],
        "last_grub_clock": _mmss(last["gameTime_ms"]),
        "pit": pit,
        "radius": radius,
        "window_ms": [t0, t1],
        "window_clock": [_mmss(t0), _mmss(t1)],
        "grubs": grubs,
        "contested": len(contested_snaps) > 0,
        "n_contested_snapshots": len(contested_snaps),
        "peak_snapshot": peak,
        "champion_kills_in_window": kills_in,
        "botlane_gold_paths": gold_paths,
        "snapshots": snapshots,
        "limits": {
            "position_cadence": "~1s stats_update frames (not continuous tick)",
            "skill_used": "no target / no hit list — AoE multi-hit not in this feed",
            "source": "Riot Live Stats JSONL (rfc461), not .rofl packet decode",
        },
    }


def summarize_feed(events: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    schemas = Counter(o.get("rfc461Schema") for o in events)
    gi = next((o for o in events if o.get("rfc461Schema") == "game_info"), {})
    ge = next((o for o in events if o.get("rfc461Schema") == "game_end"), {})
    return {
        "n_events": len(events),
        "gameID": events[0].get("gameID") if events else None,
        "gameName": next((o.get("gameName") for o in events if o.get("gameName")), None),
        "platformID": events[0].get("platformID") if events else None,
        "gameVersion": gi.get("gameVersion"),
        "statsUpdateInterval_ms": gi.get("statsUpdateInterval"),
        "winningTeam": ge.get("winningTeam"),
        "gameEnd_ms": ge.get("gameTime"),
        "schemas": dict(schemas.most_common()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (list, dict)):
                v = json.dumps(v, separators=(",", ":"))
            s = "" if v is None else str(v)
            if "," in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def run(jsonl: Path, out_dir: Path) -> dict[str, Any]:
    events = load_events(jsonl)
    roster = extract_roster(events)
    positions = extract_positions(events)
    epic = extract_epic_kills(events)
    kills = extract_champion_kills(events)
    feed = summarize_feed(events)
    grub = analyze_voidgrub_window(positions, epic, kills, roster)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feed_summary.json").write_text(json.dumps(feed, indent=2) + "\n")
    (out_dir / "roster.json").write_text(json.dumps(roster, indent=2) + "\n")
    (out_dir / "voidgrub_contest.json").write_text(json.dumps(grub, indent=2) + "\n")
    write_csv(out_dir / "positions.csv", positions)
    write_csv(out_dir / "epic_monster_kills.csv", epic)
    write_csv(out_dir / "champion_kills.csv", kills)

    # compact snapshot table for chat/canvas
    snap_rows = [
        {
            "clock": s["clock"],
            "n_blue": s["n_blue"],
            "n_red": s["n_red"],
            "near": ", ".join(
                f"{x['championName']}({x['dist']:.0f})" for x in s["near"]
            ),
        }
        for s in grub.get("snapshots") or []
    ]
    write_csv(out_dir / "voidgrub_snapshots.csv", snap_rows)

    report = {
        "source_jsonl": str(jsonl),
        "out_dir": str(out_dir),
        "feed": feed,
        "roster": roster,
        "n_position_rows": len(positions),
        "voidgrub": {
            k: grub[k]
            for k in grub
            if k != "snapshots"  # keep report lean; full in voidgrub_contest.json
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    events_preview = load_events(args.jsonl)
    gid = events_preview[0].get("gameID") if events_preview else "unknown"
    out = args.out or (DEFAULT_OUT / str(gid))
    report = run(args.jsonl, out)
    vg = report["voidgrub"]
    print(f"Wrote {out}")
    print(
        f"positions={report['n_position_rows']}  "
        f"grubs={vg.get('n_grubs')}  contested={vg.get('contested')}  "
        f"first={vg.get('first_grub_clock')}  sweep_team={vg.get('taking_teamIDs')}"
    )
    peak = vg.get("peak_snapshot") or {}
    if peak:
        print(
            f"peak@{peak.get('clock')}: blue={peak.get('n_blue')} red={peak.get('n_red')} "
            f"near={[x.get('championName') for x in peak.get('near') or []]}"
        )


if __name__ == "__main__":
    main()
