#!/usr/bin/env python3
"""
Riot Match-V5 timeline fetch + void-grub (HORDE) contest parsing.

OE gameids like LOLTMNT05_171038 are used as Match-V5 ids when the key
has esports/tournament access. Cache under data/lol/warehouse/timelines/.

Requires env RIOT_API_KEY (or RIOT_API_TOKEN). Without a key, helpers
still parse any already-cached JSON so the contest study can merge
partial timeline coverage.

  python3 -m lol_kills.etl.riot_timelines --gameids LOLTMNT05_171038 --fetch
  python3 -m lol_kills.etl.riot_timelines --from-oe --limit 50 --fetch
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from lol_kills.etl.paths import RAW_OE_DIR, WAREHOUSE_DIR

TIMELINE_DIR = WAREHOUSE_DIR / "timelines"
HORDE = "HORDE"  # void grubs in Match-V5
CONTEST_RADIUS = 2200
CONTEST_WINDOW_MS = 35_000
CONTEST_FRAME_MAX_SKEW_MS = 5_000

# Personal API key defaults (dev): 20 / 1s and 100 / 2min per routing value.
RATE_PER_SEC = 20
RATE_PER_2MIN = 100
WINDOW_2MIN = 120.0


class RiotRateLimiter:
    """Enforce 20/1s and 100/2min for a single routing host."""

    def __init__(self) -> None:
        self._ts: list[float] = []

    def wait(self) -> None:
        while True:
            now = time.monotonic()
            self._ts = [t for t in self._ts if now - t < WINDOW_2MIN]
            if sum(1 for t in self._ts if now - t < 1.0) >= RATE_PER_SEC:
                time.sleep(0.05)
                continue
            if len(self._ts) >= RATE_PER_2MIN:
                sleep_for = WINDOW_2MIN - (now - self._ts[0]) + 0.05
                time.sleep(max(sleep_for, 0.05))
                continue
            self._ts.append(time.monotonic())
            return


_LIMITERS: dict[str, RiotRateLimiter] = {}


def _limiter_for(host: str) -> RiotRateLimiter:
    if host not in _LIMITERS:
        _LIMITERS[host] = RiotRateLimiter()
    return _LIMITERS[host]


def api_key() -> str | None:
    return os.environ.get("RIOT_API_KEY") or os.environ.get("RIOT_API_TOKEN") or None


def routing_hosts() -> list[str]:
    # Prefer americas for most tournament ids; fall back if 404.
    return [
        "https://americas.api.riotgames.com",
        "https://europe.api.riotgames.com",
        "https://asia.api.riotgames.com",
        "https://sea.api.riotgames.com",
    ]


def cache_path(gameid: str) -> Path:
    safe = str(gameid).replace("/", "_")
    return TIMELINE_DIR / f"{safe}.json"


def fetch_timeline(gameid: str, *, preferred_host: str | None = None) -> dict | None:
    """Download Match-V5 timeline; None on auth/404/fail. Caches successes."""
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(gameid)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    key = api_key()
    if not key:
        return None
    headers = {"X-Riot-Token": key}
    hosts = routing_hosts()
    if preferred_host and preferred_host in hosts:
        hosts = [preferred_host] + [h for h in hosts if h != preferred_host]
    last_err = None
    for host in hosts:
        url = f"{host}/lol/match/v5/matches/{gameid}/timeline"
        _limiter_for(host).wait()
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                path.write_text(json.dumps(data))
                # remember working host via sidecar
                (TIMELINE_DIR / "_preferred_host.txt").write_text(host)
                return data
            last_err = f"{host} → {r.status_code}"
            if r.status_code == 429:
                # honor Retry-After if present
                ra = float(r.headers.get("Retry-After") or 2.0)
                print(f"[timeline] 429 on {host}; sleep {ra}s")
                time.sleep(ra + 0.1)
                continue
            if r.status_code == 404:
                continue
            if r.status_code in (401, 403):
                print(f"[timeline] auth failed ({r.status_code}); check RIOT_API_KEY")
                return None
        except Exception as e:
            last_err = str(e)
    print(f"[timeline] miss {gameid}: {last_err}")
    return None


def load_cached(gameid: str) -> dict | None:
    path = cache_path(gameid)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def parse_grub_events(timeline: dict) -> list[dict[str, Any]]:
    """Extract HORDE (void grub) kills + contest flags from a timeline JSON."""
    info = timeline.get("info") or timeline
    frames = info.get("frames") or []
    # participant → team (1 blue, 2 red) from metadata if present
    meta = timeline.get("metadata") or {}
    # Match-V5: participantId 1-5 blue, 6-10 red typically
    events_out: list[dict[str, Any]] = []
    # Collect champion kills for contest window
    champ_kills: list[dict] = []
    positions_by_ts: list[tuple[int, dict[int, tuple[float, float]]]] = []

    for fr in frames:
        ts = int(fr.get("timestamp") or 0)
        pf = fr.get("participantFrames") or {}
        pos = {}
        for pid, pdata in pf.items():
            try:
                pidi = int(pid)
            except Exception:
                continue
            pp = (pdata or {}).get("position") or {}
            if "x" in pp and "y" in pp:
                pos[pidi] = (float(pp["x"]), float(pp["y"]))
        if pos:
            positions_by_ts.append((ts, pos))
        for ev in fr.get("events") or []:
            et = ev.get("type")
            if et == "CHAMPION_KILL":
                champ_kills.append(ev)
            if et == "ELITE_MONSTER_KILL" and str(ev.get("monsterType", "")).upper() == HORDE:
                killer = int(ev.get("killerId") or 0)
                x = float((ev.get("position") or {}).get("x") or 0)
                y = float((ev.get("position") or {}).get("y") or 0)
                team = 100 if 1 <= killer <= 5 else (200 if 6 <= killer <= 10 else None)
                # Contested: enemy death near pit in window, or both teams near pit
                ets = int(ev.get("timestamp") or ts)
                # The HORDE event itself is the authoritative camp location.
                # Fixed map coordinates previously miscentered the classifier
                # and materially changed contest labels across patches/maps.
                pit = (x, y) if x or y else None
                enemy_dead = False
                for ck in champ_kills:
                    cts = int(ck.get("timestamp") or 0)
                    if abs(cts - ets) > CONTEST_WINDOW_MS:
                        continue
                    vic = int(ck.get("victimId") or 0)
                    # enemy of killer's team
                    if team == 100 and not (6 <= vic <= 10):
                        continue
                    if team == 200 and not (1 <= vic <= 5):
                        continue
                    vpos = (ck.get("position") or {})
                    if pit is not None and "x" in vpos:
                        if _dist((float(vpos["x"]), float(vpos["y"])), pit) <= CONTEST_RADIUS:
                            enemy_dead = True
                            break
                both_near = False
                frame_usable = False
                # nearest frame positions
                nearest = min(positions_by_ts, key=lambda z: abs(z[0] - ets), default=None)
                if (
                    nearest
                    and pit is not None
                    and abs(nearest[0] - ets) <= CONTEST_FRAME_MAX_SKEW_MS
                    and set(nearest[1]) >= set(range(1, 11))
                ):
                    frame_usable = True
                    blue_near = any(
                        _dist(xy, pit) <= CONTEST_RADIUS for pid, xy in nearest[1].items() if 1 <= pid <= 5
                    )
                    red_near = any(
                        _dist(xy, pit) <= CONTEST_RADIUS for pid, xy in nearest[1].items() if 6 <= pid <= 10
                    )
                    both_near = blue_near and red_near
                contested: bool | None
                if enemy_dead or both_near:
                    contested = True
                elif frame_usable:
                    contested = False
                else:
                    contested = None
                events_out.append(
                    {
                        "timestamp_ms": ets,
                        "minute": round(ets / 60000.0, 2),
                        "killer_id": killer,
                        "killer_team": team,  # 100 blue / 200 red
                        "x": x,
                        "y": y,
                        "contested": contested,
                        "contest_reason": (
                            "enemy_death_near_pit"
                            if enemy_dead
                            else (
                                "both_teams_near"
                                if both_near
                                else (
                                    "verified_no_enemy_presence"
                                    if frame_usable
                                    else "unknown_stale_or_incomplete_position_frame"
                                )
                            )
                        ),
                    }
                )
    return events_out


def summarize_map_grubs(timeline: dict) -> dict[str, Any]:
    evs = parse_grub_events(timeline)
    blue = sum(1 for e in evs if e.get("killer_team") == 100)
    red = sum(1 for e in evs if e.get("killer_team") == 200)
    contested = [e for e in evs if e.get("contested") is True]
    uncontested = [e for e in evs if e.get("contested") is False]
    unknown = [e for e in evs if e.get("contested") is None]
    return {
        "n_horde_events": len(evs),
        "blue_grubs_timeline": blue,
        "red_grubs_timeline": red,
        "n_contested": len(contested),
        "n_uncontested": len(uncontested),
        "n_contest_unknown": len(unknown),
        "any_contested": (
            True
            if contested
            else (False if uncontested and not unknown else None)
        ),
        "first_grub_minute": evs[0]["minute"] if evs else None,
        "events": evs,
    }


def oe_gameids(*, year_min: int = 2025, limit: int | None = None) -> list[str]:
    files = sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    ids: list[str] = []
    seen = set()
    for fp in files:
        y = int(fp.name[:4])
        if y < year_min:
            continue
        import pandas as pd

        df = pd.read_csv(fp, usecols=["gameid", "position"], low_memory=False)
        df = df[df["position"].astype(str).str.lower() == "team"]
        for g in df["gameid"].astype(str):
            if g in seen:
                continue
            # Prefer Riot-looking tournament ids
            if "_" not in g or g[0].isdigit():
                continue
            seen.add(g)
            ids.append(g)
            if limit and len(ids) >= limit:
                return ids
    return ids


def coverage_report(gameids: Iterable[str]) -> dict:
    ids = list(gameids)
    cached = sum(1 for g in ids if cache_path(g).exists())
    return {"n_ids": len(ids), "n_cached": cached, "frac": cached / max(len(ids), 1), "dir": str(TIMELINE_DIR)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gameids", nargs="*", default=[])
    ap.add_argument("--from-oe", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fetch", action="store_true", help="Hit Riot API (needs RIOT_API_KEY)")
    ap.add_argument("--year-min", type=int, default=2025)
    args = ap.parse_args(argv)

    ids = list(args.gameids)
    if args.from_oe:
        ids.extend(oe_gameids(year_min=args.year_min, limit=args.limit))
    ids = list(dict.fromkeys(ids))
    print(f"[timeline] ids={len(ids)} key={'yes' if api_key() else 'NO'}")
    print("[timeline] coverage", coverage_report(ids))

    if args.fetch:
        pref = None
        pref_path = TIMELINE_DIR / "_preferred_host.txt"
        if pref_path.exists():
            pref = pref_path.read_text().strip() or None
        ok = miss = skip = 0
        todo = [g for g in ids if not cache_path(g).exists()]
        print(f"[timeline] to_fetch={len(todo)} already_cached={len(ids)-len(todo)} preferred={pref}")
        t0 = time.time()
        for i, g in enumerate(todo):
            data = fetch_timeline(g, preferred_host=pref)
            if data:
                ok += 1
                if pref_path.exists():
                    pref = pref_path.read_text().strip() or pref
                s = summarize_map_grubs(data)
                if (ok <= 5) or (ok % 25 == 0):
                    print(
                        f"  ok#{ok} {g}: horde={s['n_horde_events']} contested={s['n_contested']} "
                        f"elapsed={time.time()-t0:.0f}s"
                    )
            else:
                miss += 1
                if miss <= 10 or miss % 50 == 0:
                    print(f"  miss#{miss} {g}")
            if (i + 1) % 50 == 0:
                print(
                    f"  … progress {i+1}/{len(todo)} ok={ok} miss={miss} "
                    f"rate≈{ok/max(time.time()-t0,1)*60:.1f}/min"
                )
        print(f"[timeline] done ok={ok} miss={miss} skip_cached={len(ids)-len(todo)} cache={TIMELINE_DIR}")
    else:
        # summarize cache only
        n_c = 0
        for g in ids:
            data = load_cached(g)
            if not data:
                continue
            n_c += 1
            s = summarize_map_grubs(data)
            if s["n_horde_events"]:
                print(f"  {g}: {s['n_horde_events']} horde, contested={s['n_contested']}")
        print(f"[timeline] cached_parsed={n_c}")


if __name__ == "__main__":
    main()
