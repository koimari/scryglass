#!/usr/bin/env python3
"""
Ranked SOLO queue HORDE contest-EV proof (personal Riot API key only).

Pro OE gameids (LOLTMNT*) are NOT in public Match-V5. Until Grid/partner access,
the Riot-timeline layer is deliberately a separate high-elo SOLO queue study:
Diamond and Masters+ anchor cohorts, reported separately, across every supported
platform. It must not be pooled with the Oracle's Elixir professional calibration
or with ranks below Diamond.

Rate limits (personal key): 20/1s and 100/2min per routing value.

  export RIOT_API_KEY=RGAPI-...
  python3 -m lol_kills.research.grubs_ranked_contest_proof --n-matches 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import requests

from lol_kills.etl.paths import MODELS_DIR
from lol_kills.etl.riot_timelines import (
    RiotRateLimiter,
    TIMELINE_DIR,
    cache_path,
    summarize_map_grubs,
)
from lol_kills.research.grubs_intrinsic_value import (
    _wald_covariance,
    contest_certainty_atlas,
    cross_validated_gold_diagnostics,
    delta_pp,
    fit_logit,
    univariate_delta_sampling_ci,
)

# Riot's current LoL platform / Match-V5 regional routing map.  Keep the
# platform list exhaustive so estimates are not silently NA/EUW/KR-only.
PLATFORMS = {
    "br1": "americas", "la1": "americas", "la2": "americas", "na1": "americas",
    "eun1": "europe", "euw1": "europe", "ru": "europe", "tr1": "europe",
    "jp1": "asia", "kr": "asia",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
}
DIAMOND_DIVISIONS = ("I", "II", "III", "IV")
RANK_BUCKETS = ("diamond", "masters_plus")


def _headers() -> dict:
    key = os.environ.get("RIOT_API_KEY") or os.environ.get("RIOT_API_TOKEN")
    if not key:
        raise SystemExit("Set RIOT_API_KEY")
    return {"X-Riot-Token": key}


def _sample_entries(entries: list[dict], n: int) -> list[dict]:
    """Deterministic spread across a ladder page; avoids a pure top-of-ladder grab."""
    if len(entries) <= n:
        return entries
    idx = np.linspace(0, len(entries) - 1, num=n, dtype=int)
    return [entries[i] for i in np.unique(idx)]


def _ranked_anchor_puuids(
    platform: str,
    bucket: str,
    headers: dict,
    lim: RiotRateLimiter,
    per_platform: int,
) -> list[str]:
    """Return current-rank anchors for one strict D or M+ cohort.

    A game is initially sampled through one of these anchors.  This is not a
    historical all-ten-player rank reconstruction, so the report labels it as
    an *anchor cohort* and never calls it a whole-lobby rank guarantee.
    """
    urls: list[str] = []
    if bucket == "diamond":
        urls = [
            f"https://{platform}.api.riotgames.com/lol/league/v4/entries/"
            f"RANKED_SOLO_5x5/DIAMOND/{division}"
            for division in DIAMOND_DIVISIONS
        ]
    elif bucket == "masters_plus":
        urls = [
            f"https://{platform}.api.riotgames.com/lol/league/v4/{tier}leagues/"
            "by-queue/RANKED_SOLO_5x5"
            for tier in ("master", "grandmaster", "challenger")
        ]
    else:
        raise ValueError(f"Unknown rank bucket: {bucket}")

    entries: list[dict] = []
    for url in urls:
        lim.wait()
        try:
            response = requests.get(url, headers=headers, timeout=(5, 15))
        except requests.RequestException as exc:
            print(f"[ranked] {platform} {bucket} unavailable: {type(exc).__name__}")
            # A platform DNS/connect failure applies to every tier endpoint;
            # skip its remaining calls and retain the rest of the global panel.
            return []
        if response.status_code != 200:
            print(f"[ranked] {platform} {bucket} -> {response.status_code}")
            continue
        payload = response.json()
        if isinstance(payload, list):
            entries.extend(payload)
        else:
            entries.extend(payload.get("entries") or [])
    return [e["puuid"] for e in _sample_entries(entries, per_platform) if e.get("puuid")]


def collect_match_ids(
    n: int,
    headers: dict,
    *,
    anchors_per_bucket_platform: int,
    matches_per_anchor: int,
    platforms: dict[str, str] | None = None,
) -> list[tuple[str, str, str, str, str]]:
    """Return (routing, match_id, rank_bucket, platform, anchor) tuples.

    Each platform x bucket cell receives an initial equal quota before any cell
    can grow. Masters+ is processed first within each platform, so an overlapping
    match is assigned to that more selective cohort. This keeps D and M+ samples
    disjoint while retaining all-geography coverage.
    """
    out: list[tuple[str, str, str, str, str]] = []
    seen = set()
    limiters: dict[str, RiotRateLimiter] = {}

    def limiter_for(host: str) -> RiotRateLimiter:
        if host not in limiters:
            limiters[host] = RiotRateLimiter()
        return limiters[host]

    platforms = platforms or PLATFORMS
    n_cells = len(platforms) * len(RANK_BUCKETS)
    target_per_cell = max(1, math.ceil(n / n_cells))
    for platform, routing in platforms.items():
        for bucket in ("masters_plus", "diamond"):
            cell_n = 0
            anchors = _ranked_anchor_puuids(
                platform, bucket, headers, limiter_for(platform), anchors_per_bucket_platform
            )
            for puuid in anchors:
                limiter_for(routing).wait()
                try:
                    mr = requests.get(
                        f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids",
                        headers=headers,
                        params={"start": 0, "count": matches_per_anchor, "queue": 420},
                        timeout=(5, 15),
                    )
                except requests.RequestException as exc:
                    print(f"[ranked] {platform} matchlist unavailable: {type(exc).__name__}")
                    continue
                if mr.status_code != 200:
                    continue
                for mid in mr.json():
                    if mid in seen:
                        continue
                    seen.add(mid)
                    out.append((routing, mid, bucket, platform, puuid))
                    cell_n += 1
                    if len(out) >= n:
                        return out
                    if cell_n >= target_per_cell:
                        break
                if cell_n >= target_per_cell:
                    break
    return out


def gold_at_minute(timeline: dict, minute: float, team100: bool = True) -> float | None:
    """Sum participant gold from nearest frame ≤ minute."""
    frames = (timeline.get("info") or {}).get("frames") or []
    target_ms = minute * 60_000
    best = None
    best_dt = 1e18
    for fr in frames:
        ts = int(fr.get("timestamp") or 0)
        dt = abs(ts - target_ms)
        if dt < best_dt:
            best_dt = dt
            best = fr
    if not best:
        return None
    total = 0.0
    n = 0
    for pid, pdata in (best.get("participantFrames") or {}).items():
        try:
            pidi = int(pid)
        except Exception:
            continue
        if team100 and not (1 <= pidi <= 5):
            continue
        if (not team100) and not (6 <= pidi <= 10):
            continue
        total += float((pdata or {}).get("totalGold") or 0)
        n += 1
    return total if n else None


def match_winner_blue(match: dict) -> bool | None:
    info = match.get("info") or {}
    for t in info.get("teams") or []:
        if int(t.get("teamId") or 0) == 100:
            return bool(t.get("win"))
    return None


def timeline_winner_blue(timeline: dict) -> bool | None:
    """Read the winning team from the terminal GAME_END timeline event."""
    for frame in reversed((timeline.get("info") or {}).get("frames") or []):
        for event in reversed(frame.get("events") or []):
            if event.get("type") == "GAME_END" and event.get("winningTeam") is not None:
                return int(event["winningTeam"]) == 100
    return None


def analyze_timeline(
    routing: str, mid: str, headers: dict, lim: RiotRateLimiter, *, rank_bucket: str, platform: str
) -> dict | None:
    path = cache_path(mid)
    if path.exists():
        timeline = json.loads(path.read_text())
    else:
        lim.wait()
        try:
            r = requests.get(
                f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{mid}/timeline",
                headers=headers,
                timeout=(5, 20),
            )
        except requests.RequestException:
            return None
        if r.status_code == 429:
            ra = float(r.headers.get("Retry-After") or 2)
            time.sleep(ra + 0.1)
            return analyze_timeline(routing, mid, headers, lim, rank_bucket=rank_bucket, platform=platform)
        if r.status_code != 200:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
        timeline = r.json()

    win_blue = timeline_winner_blue(timeline)
    # Older timelines can lack GAME_END. Use a cached/fetched match DTO only as
    # a fallback, avoiding one redundant API request for normal modern matches.
    if win_blue is None:
        mpath = path.with_suffix(".match.json")
        if mpath.exists():
            match = json.loads(mpath.read_text())
        else:
            lim.wait()
            try:
                r = requests.get(
                    f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{mid}",
                    headers=headers,
                    timeout=(5, 20),
                )
            except requests.RequestException:
                return None
            if r.status_code == 429:
                ra = float(r.headers.get("Retry-After") or 2)
                time.sleep(ra + 0.1)
                return analyze_timeline(
                    routing, mid, headers, lim,
                    rank_bucket=rank_bucket, platform=platform,
                )
            if r.status_code != 200:
                return None
            mpath.write_text(r.text)
            match = r.json()
        win_blue = match_winner_blue(match)
    if win_blue is None:
        return None
    gsum = summarize_map_grubs(timeline)
    if gsum["n_horde_events"] < 1:
        return None

    g8_b = gold_at_minute(timeline, 8.0, True)
    g8_r = gold_at_minute(timeline, 8.0, False)
    g10_b = gold_at_minute(timeline, 10.0, True)
    g10_r = gold_at_minute(timeline, 10.0, False)
    gold8_diff = (g8_b - g8_r) if g8_b is not None and g8_r is not None else None
    gold10_diff = (g10_b - g10_r) if g10_b is not None and g10_r is not None else None

    # death cost: champ kills in contest window of any contested HORDE
    death_near = 0
    for ev in gsum["events"]:
        if not ev.get("contested"):
            continue
        # recount from timeline for team of killer vs enemy
        death_near += 1  # contested flag already encodes enemy death or both near

    sweeper_blue = gsum["blue_grubs_timeline"] > gsum["red_grubs_timeline"]
    sweeper_won = (win_blue and sweeper_blue) or ((not win_blue) and (not sweeper_blue))
    all3_blue = gsum["blue_grubs_timeline"] >= 3 and gsum["red_grubs_timeline"] == 0
    all3_red = gsum["red_grubs_timeline"] >= 3 and gsum["blue_grubs_timeline"] == 0

    return {
        "match_id": mid,
        "routing": routing,
        "platform": platform,
        "rank_bucket": rank_bucket,
        "win_blue": win_blue,
        "n_horde": gsum["n_horde_events"],
        "blue_grubs": gsum["blue_grubs_timeline"],
        "red_grubs": gsum["red_grubs_timeline"],
        "any_contested": gsum["any_contested"],
        "n_contested": gsum["n_contested"],
        "first_grub_minute": gsum["first_grub_minute"],
        "gold8_diff_blue": gold8_diff,
        "gold10_diff_blue": gold10_diff,
        "all3_blue": all3_blue,
        "all3_red": all3_red,
        "sweeper_won": sweeper_won,
        "events": gsum["events"],
    }


def bin_gold(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "missing"
    if x < -2000:
        return "behind2k"
    if x < -500:
        return "behind"
    if x <= 500:
        return "even"
    if x <= 2000:
        return "ahead"
    return "ahead2k"


def gold10_calibration(rows: list[dict]) -> dict | None:
    """Fit the same bounded gold@10 conversion used by the pro study."""
    if len(rows) < 20:
        return None
    gold = np.asarray(
        [r.get("gold10_diff_blue", np.nan) for r in rows], dtype=float
    )
    won = np.asarray([1.0 if r["win_blue"] else 0.0 for r in rows], dtype=float)
    intercept, coef, n_fit = fit_logit(gold, won, abs_cap=3000)
    fit_mask = np.isfinite(gold) & np.isfinite(won) & (np.abs(gold) <= 3000)
    fit_X = np.column_stack(
        [np.ones(int(fit_mask.sum())), gold[fit_mask] / 1000.0]
    )
    fit_covariance_scaled = _wald_covariance(
        fit_X, intercept, np.asarray([coef * 1000.0])
    )
    diagnostics = cross_validated_gold_diagnostics(
        gold, won, abs_cap=3000, folds=10
    )
    cash_ci = univariate_delta_sampling_ci(
        gold,
        won,
        abs_cap=3000,
        intercept=intercept,
        coef=coef,
        base=0.0,
        bump=90.0,
    )
    atlas = contest_certainty_atlas(
        intercept,
        coef,
        touch_gold=(192.0 / 900.0) * 120.0,
        covariance_scaled=fit_covariance_scaled,
    )
    pstars = [
        float(cell["breakeven_p_win_fight"])
        for package in atlas["packages"].values()
        for cell in package["cells"]
    ]
    return {
        "gold_minute": 10,
        "abs_gold_cap": 3000,
        "n_fit": int(n_fit),
        "intercept": float(intercept),
        "coef_per_gold": float(coef),
        "cash_90g_pp_at_even": float(delta_pp(intercept, coef, 0.0, 90.0)),
        "cash_90g_wald_95_ci": cash_ci,
        "diagnostics_10fold": diagnostics,
        "certainty_atlas_common_state_specification": atlas,
        "breakeven_min": float(min(pstars)),
        "breakeven_max": float(max(pstars)),
        "comparison_note": (
            "Uses the same gold cap, 90g cash state, symmetric +/-600g fight "
            "sensitivity, and leave-state grid as the competitive-map model. "
            "This aligns the conversion scale; it does not equate populations."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-matches", type=int, default=200, help="Target match IDs to attempt")
    ap.add_argument("--anchors-per-bucket-platform", type=int, default=4)
    ap.add_argument("--matches-per-anchor", type=int, default=12)
    ap.add_argument(
        "--platforms",
        default=",".join(PLATFORMS),
        help="Comma-separated Riot platform IDs; defaults to every supported platform",
    )
    args = ap.parse_args()

    requested_platforms = [p.strip().lower() for p in args.platforms.split(",") if p.strip()]
    unknown_platforms = [p for p in requested_platforms if p not in PLATFORMS]
    if unknown_platforms:
        raise SystemExit(f"Unknown Riot platform IDs: {', '.join(unknown_platforms)}")
    selected_platforms = {p: PLATFORMS[p] for p in requested_platforms}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    headers = _headers()
    route_limiters = {
        routing: RiotRateLimiter() for routing in set(selected_platforms.values())
    }

    print(f"[ranked] collecting up to {args.n_matches} match ids…")
    ids = collect_match_ids(
        args.n_matches,
        headers,
        anchors_per_bucket_platform=args.anchors_per_bucket_platform,
        matches_per_anchor=args.matches_per_anchor,
        platforms=selected_platforms,
    )
    print(f"[ranked] got {len(ids)} ids; fetching timelines (100/2min)…")

    # Round-robin regional hosts so each independent Riot app-rate window is
    # used evenly instead of exhausting one route while the others sit idle.
    by_route: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_match_ids": int(args.n_matches),
        "matches": [
            {
                "routing": routing,
                "match_id": mid,
                "rank_bucket": rank_bucket,
                "platform": platform,
                "anchor_cluster": hashlib.sha256(puuid.encode("utf-8")).hexdigest()[:20],
            }
            for routing, mid, rank_bucket, platform, puuid in ids
        ],
    }
    (MODELS_DIR / "grubs_ranked_contest_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    for item in ids:
        by_route[item[0]].append(item)
    interleaved_ids: list[tuple[str, str, str, str, str]] = []
    while any(by_route.values()):
        for routing in sorted(by_route):
            if by_route[routing]:
                interleaved_ids.append(by_route[routing].pop(0))

    rows = []
    t0 = time.time()
    for i, (routing, mid, rank_bucket, platform, puuid) in enumerate(interleaved_ids):
        row = analyze_timeline(
            routing, mid, headers, route_limiters[routing],
            rank_bucket=rank_bucket, platform=platform
        )
        if row:
            row["anchor_cluster"] = hashlib.sha256(
                puuid.encode("utf-8")
            ).hexdigest()[:20]
            rows.append(row)
        if (i + 1) % 20 == 0 or i == 0:
            print(
                f"  … {i+1}/{len(interleaved_ids)} parsed={len(rows)} "
                f"elapsed={time.time()-t0:.0f}s"
            )

    if not rows:
        raise SystemExit("No HORDE timelines parsed — check key / queue")

    (MODELS_DIR / "grubs_ranked_contest_rows.json").write_text(
        json.dumps(rows, indent=2)
    )

    n = len(rows)
    contested = [r for r in rows if r["any_contested"]]
    free = [r for r in rows if not r["any_contested"]]
    all3 = [r for r in rows if r["all3_blue"] or r["all3_red"]]

    def wr(rs):
        if not rs:
            return None
        return float(np.mean([1.0 if r["sweeper_won"] else 0.0 for r in rs]))

    # Contested vs free sweeper WR
    by_bin = defaultdict(lambda: {"contested": [], "free": []})
    for r in all3:
        # gold from sweeper POV
        if r["all3_blue"] and r.get("gold8_diff_blue") is not None:
            g = r["gold8_diff_blue"]
        elif r["all3_red"] and r.get("gold8_diff_blue") is not None:
            g = -r["gold8_diff_blue"]
        else:
            g = None
        b = bin_gold(g)
        if r["any_contested"]:
            by_bin[b]["contested"].append(r)
        else:
            by_bin[b]["free"].append(r)

    bin_table = []
    for b, parts in by_bin.items():
        if b == "missing":
            continue
        c, f = parts["contested"], parts["free"]
        bin_table.append(
            {
                "bin_sweeper_gold8": b,
                "n_contested": len(c),
                "n_free": len(f),
                "wr_sweeper_contested": wr(c),
                "wr_sweeper_free": wr(f),
                "dpp_contested_minus_free": (
                    (wr(c) - wr(f)) * 100 if wr(c) is not None and wr(f) is not None else None
                ),
            }
        )

    # Dog steal: sweeper was behind @8 but got all3
    dog_steal = []
    fav_take = []
    for r in all3:
        if r["all3_blue"] and r.get("gold8_diff_blue") is not None:
            lead = r["gold8_diff_blue"]
        elif r["all3_red"] and r.get("gold8_diff_blue") is not None:
            lead = -r["gold8_diff_blue"]
        else:
            continue
        if lead < -500:
            dog_steal.append(r)
        elif lead > 500:
            fav_take.append(r)

    report = {
        "version": 2,
        "population": "ranked_solo_HORDE_high_elo_anchor_cohorts",
        "purpose": (
            "Riot-timeline contest analysis for strict Diamond and Masters+ SOLO queue "
            "anchor cohorts across all supported platforms. Pro OE LOLTMNT* Match-V5 "
            "is inaccessible without Grid/partner; do not pool or directly equate this "
            "ranked layer with the separate professional OE calibration."
        ),
        "rank_policy": {
            "included_anchor_buckets": ["diamond", "masters_plus"],
            "excluded_below_diamond": True,
            "bucket_rule": "Masters+ wins duplicates; Diamond receives D-tier anchors only.",
            "caveat": "Anchor current rank is not historical all-lobby rank at match time.",
        },
        "platforms_requested": list(selected_platforms),
        "platforms_observed": sorted({r["platform"] for r in rows}),
        "platforms_without_usable_horde_rows": sorted(
            set(selected_platforms) - {r["platform"] for r in rows}
        ),
        "target_match_ids": int(args.n_matches),
        "n_match_ids_attempted": int(len(ids)),
        "n_matches_with_horde": n,
        "contest_rate": float(len(contested) / n),
        "n_contested": len(contested),
        "n_free": len(free),
        "n_all3": len(all3),
        "sweeper_wr_all3": wr(all3),
        "sweeper_wr_contested_all3": wr([r for r in all3 if r["any_contested"]]),
        "sweeper_wr_free_all3": wr([r for r in all3 if not r["any_contested"]]),
        "dog_steal_behind8": {
            "n": len(dog_steal),
            "contest_rate": float(np.mean([r["any_contested"] for r in dog_steal])) if dog_steal else None,
            "sweeper_wr": wr(dog_steal),
        },
        "fav_take_ahead8": {
            "n": len(fav_take),
            "contest_rate": float(np.mean([r["any_contested"] for r in fav_take])) if fav_take else None,
            "sweeper_wr": wr(fav_take),
        },
        "by_gold8_bin": bin_table,
        "mean_first_grub_minute": float(np.mean([r["first_grub_minute"] for r in rows if r["first_grub_minute"]])),
        "rate_limits": {"per_1s": 20, "per_2min": 100},
        "note_for_IMLS": (
            "Contest flags = enemy death near pit OR both teams near pit within 35s of HORDE kill. "
            "Ranked is not pro; the event schema is identical (ELITE_MONSTER_KILL / HORDE), "
            "but any cross-population generalization remains a hypothesis."
        ),
    }
    report["by_rank_bucket"] = {
        bucket: {
            "n_matches_with_horde": int(len(part)),
            "n_contested": int(sum(r["any_contested"] for r in part)),
            "contest_rate": float(np.mean([r["any_contested"] for r in part])) if part else None,
            "n_all3": int(sum(r["all3_blue"] or r["all3_red"] for r in part)),
            "gold10_calibration": gold10_calibration(part),
        }
        for bucket in RANK_BUCKETS
        for part in [[r for r in rows if r["rank_bucket"] == bucket]]
    }
    report["gold10_calibration"] = gold10_calibration(rows)

    # Simple contest EV sketch for dog: p_win_fight unknown; use empirical
    # If dog contested and got all3 vs free fav take — selection still present but flagged
    c_all3 = [r for r in all3 if r["any_contested"]]
    f_all3 = [r for r in all3 if not r["any_contested"]]
    if wr(c_all3) is not None and wr(f_all3) is not None:
        report["delta_sweeper_wr_contested_vs_free_pp"] = (wr(c_all3) - wr(f_all3)) * 100

    out = MODELS_DIR / "grubs_ranked_contest_proof.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    brief = MODELS_DIR / "grubs_ranked_contest_proof.md"
    lines = [
        "# High-elo ranked SOLO queue HORDE contest proof (personal Riot key)",
        "",
        f"**n={n}** ranked matches with ≥1 void grub (HORDE) · contest rate **{report['contest_rate']:.1%}**",
        "",
        "## Why ranked",
        "",
        "".join(
            (
                "OE pro `LOLTMNT*` gameids are unavailable through a personal Match-V5 key. ",
                "This is therefore a distinct ranked SOLO queue layer, not a pro dataset. ",
                "It samples Diamond and Masters+ anchors separately across all supported platforms; ",
                "ranks below Diamond are excluded from the sampling frame.",
            )
        ),
        "",
        "## Results",
        "",
        f"- All-3 sweeper WR: **{(report['sweeper_wr_all3'] or 0)*100:.1f}%** (n={len(all3)})",
        f"- Contested all-3 sweeper WR: **{(report['sweeper_wr_contested_all3'] or 0)*100:.1f}%**",
        f"- Free all-3 sweeper WR: **{(report['sweeper_wr_free_all3'] or 0)*100:.1f}%**",
        f"- Δ contested−free: **{report.get('delta_sweeper_wr_contested_vs_free_pp')}** pp",
        f"- Dog steal (behind@8, got all3): n={report['dog_steal_behind8']['n']} · "
        f"contest_rate={report['dog_steal_behind8']['contest_rate']} · "
        f"WR={report['dog_steal_behind8']['sweeper_wr']}",
        f"- Fav take (ahead@8, got all3): n={report['fav_take_ahead8']['n']} · "
        f"contest_rate={report['fav_take_ahead8']['contest_rate']} · "
        f"WR={report['fav_take_ahead8']['sweeper_wr']}",
        f"- Mean first grub minute: **{report['mean_first_grub_minute']:.2f}**",
        f"- Diamond anchor cohort: n={report['by_rank_bucket']['diamond']['n_matches_with_horde']}"
        f"; Masters+ anchor cohort: n={report['by_rank_bucket']['masters_plus']['n_matches_with_horde']}",
        "",
        "## Gold@8 bins (sweeper POV)",
        "",
        "| Bin | n contested | n free | WR contested | WR free | Δpp |",
        "|-----|-------------|--------|--------------|---------|-----|",
    ]
    for b in bin_table:
        lines.append(
            f"| {b['bin_sweeper_gold8']} | {b['n_contested']} | {b['n_free']} | "
            f"{b['wr_sweeper_contested']} | {b['wr_sweeper_free']} | {b['dpp_contested_minus_free']} |"
        )
    lines += [
        "",
        "## Next",
        "",
        "1. Scale `--n-matches` while retaining the separate Diamond and Masters+ reports.",
        "2. Add historical all-lobby rank reconstruction before making a lobby-wide tier claim.",
        "3. When Grid is justified: use a separate pro series rather than blending pro and ranked estimates.",
        "",
    ]
    brief.write_text("\n".join(lines))
    print(f"[ranked] wrote {out}")
    print(f"[ranked] wrote {brief}")
    print(json.dumps({k: report[k] for k in report if k != "by_gold8_bin"}, indent=2, default=str))


if __name__ == "__main__":
    main()
