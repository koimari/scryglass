#!/usr/bin/env python3
"""
Deep side × objective edge hunt from *raw* OE team rows.

Warehouse maps.parquet dropped the good stuff. This pulls it back:
  void grubs, elemental dragon *types*, elders, heralds,
  gold/xp/kills @20/@25, blue vs red asymmetries.

Creative proxies (OE has no baron clock / no "first dragon type"):
  - baron_state_at20 = golddiff@20 on maps with firstbaron
  - likely_first_drake_type when team has exactly one elemental type
  - soul_path = elementaldrakes >= 4
  - grub_sweep / grub_starve
  - lead_conversion by dragon type presence

  python3 -m lol_kills.research.side_objective_edges
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lol_kills.etl.paths import MODELS_DIR, RAW_OE_DIR

DRAGON_TYPES = ["infernals", "mountains", "clouds", "oceans", "hextechs", "chemtechs"]
MIN_N = 200


def _r(a, b, min_n=MIN_N):
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < min_n:
        return None, n
    aa, bb = a[mask], b[mask]
    if float(aa.std()) < 1e-12 or float(bb.std()) < 1e-12:
        return None, n
    return float(np.corrcoef(aa, bb)[0, 1]), n


def _rate(mask_num, mask_den) -> dict | None:
    den = int(mask_den.sum())
    if den < MIN_N:
        return None
    num = int((mask_num & mask_den).sum())
    return {"n": den, "rate": round(num / den, 4), "count": num}


def _wr_table(df: pd.DataFrame, group_col: str, y="y_blue_win") -> list[dict]:
    rows = []
    for k, g in df.groupby(group_col):
        if len(g) < 80:
            continue
        rows.append(
            {
                "key": str(k),
                "n": int(len(g)),
                "wr_blue": round(float(g[y].mean()), 4),
                "mean_kills": round(float(g["total_kills"].mean()), 2) if "total_kills" in g else None,
                "mean_length": round(float(g["length_min"].mean()), 2) if "length_min" in g else None,
            }
        )
    rows.sort(key=lambda x: -abs(x["wr_blue"] - 0.5))
    return rows


def load_oe_team_maps() -> pd.DataFrame:
    """Pivot OE team rows → one map row with blue_/red_ rich objective cols."""
    files = sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    want = {
        "gameid",
        "date",
        "league",
        "year",
        "split",
        "playoffs",
        "patch",
        "side",
        "position",
        "teamname",
        "result",
        "gamelength",
        "kills",
        "deaths",
        "assists",
        "teamkills",
        "firstblood",
        "firstdragon",
        "firstherald",
        "firstbaron",
        "firsttower",
        "dragons",
        "barons",
        "towers",
        "heralds",
        "void_grubs",
        "elders",
        "elementaldrakes",
        "infernals",
        "mountains",
        "clouds",
        "oceans",
        "hextechs",
        "chemtechs",
        "golddiffat10",
        "golddiffat15",
        "golddiffat20",
        "golddiffat25",
        "xpdiffat15",
        "xpdiffat20",
        "killsat15",
        "killsat20",
        "ckpm",
    }
    frames = []
    for fp in files:
        hdr = pd.read_csv(fp, nrows=0).columns.tolist()
        usecols = [c for c in hdr if c in want]
        df = pd.read_csv(fp, usecols=usecols, low_memory=False)
        df = df[df["position"].astype(str).str.lower() == "team"].copy()
        df["oe_year"] = int(fp.name[:4])
        frames.append(df)
        print(f"[sideobj] loaded {fp.name} team_rows={len(df)}")
    raw = pd.concat(frames, ignore_index=True)
    raw["side"] = raw["side"].astype(str).str.title()
    raw["gameid"] = raw["gameid"].astype(str)

    def pivot_side(side: str, prefix: str) -> pd.DataFrame:
        s = raw[raw["side"] == side].copy()
        s = s.drop_duplicates("gameid", keep="first")
        rename = {c: f"{prefix}{c}" for c in s.columns if c not in ("gameid", "side", "position")}
        s = s.rename(columns=rename)
        return s

    blue = pivot_side("Blue", "blue_")
    red = pivot_side("Red", "red_")
    # shared meta from blue
    meta_cols = ["blue_date", "blue_league", "blue_year", "blue_split", "blue_playoffs", "blue_patch", "blue_oe_year", "blue_gamelength"]
    m = blue.merge(red, on="gameid", how="inner")
    m = m.rename(
        columns={
            "blue_date": "date",
            "blue_league": "league",
            "blue_year": "year",
            "blue_patch": "patch",
            "blue_playoffs": "playoffs",
            "blue_oe_year": "oe_year",
            "blue_gamelength": "gamelength",
            "blue_result": "y_blue_win",
        }
    )
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["y_blue_win"] = pd.to_numeric(m["y_blue_win"], errors="coerce")
    m["length_min"] = pd.to_numeric(m["gamelength"], errors="coerce") / 60.0
    m["total_kills"] = pd.to_numeric(m["blue_kills"], errors="coerce") + pd.to_numeric(m["red_kills"], errors="coerce")
    m["ckpm"] = pd.to_numeric(m.get("blue_ckpm"), errors="coerce")
    return m.dropna(subset=["y_blue_win", "date"]).sort_values("date")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # numeric coerce helpers
    def n(col):
        return pd.to_numeric(d.get(col), errors="coerce")

    for side in ("blue", "red"):
        d[f"{side}_void_grubs"] = n(f"{side}_void_grubs")
        d[f"{side}_elders"] = n(f"{side}_elders")
        d[f"{side}_elementaldrakes"] = n(f"{side}_elementaldrakes")
        d[f"{side}_heralds"] = n(f"{side}_heralds")
        d[f"{side}_firstbaron"] = n(f"{side}_firstbaron")
        d[f"{side}_firstdragon"] = n(f"{side}_firstdragon")
        d[f"{side}_firstherald"] = n(f"{side}_firstherald")
        d[f"{side}_firsttower"] = n(f"{side}_firsttower")
        d[f"{side}_firstblood"] = n(f"{side}_firstblood")
        for t in DRAGON_TYPES:
            d[f"{side}_{t}"] = n(f"{side}_{t}")
        for t in (10, 15, 20, 25):
            d[f"{side}_golddiffat{t}"] = n(f"{side}_golddiffat{t}")

    d["grub_diff"] = d["blue_void_grubs"].fillna(0) - d["red_void_grubs"].fillna(0)
    d["grub_sum"] = d["blue_void_grubs"].fillna(0) + d["red_void_grubs"].fillna(0)
    d["grub_sweep_blue"] = ((d["blue_void_grubs"] >= 5) & (d["red_void_grubs"].fillna(0) <= 1)).astype(float)
    d["grub_starve_blue"] = ((d["blue_void_grubs"].fillna(0) <= 1) & (d["red_void_grubs"] >= 5)).astype(float)
    d["herald_diff"] = d["blue_heralds"].fillna(0) - d["red_heralds"].fillna(0)
    d["elder_blue"] = (d["blue_elders"].fillna(0) >= 1).astype(float)
    d["elder_red"] = (d["red_elders"].fillna(0) >= 1).astype(float)
    d["soul_blue"] = (d["blue_elementaldrakes"].fillna(0) >= 4).astype(float)
    d["soul_red"] = (d["red_elementaldrakes"].fillna(0) >= 4).astype(float)

    # map-level dragon type presence / majority
    for t in DRAGON_TYPES:
        d[f"map_{t}"] = d[f"blue_{t}"].fillna(0) + d[f"red_{t}"].fillna(0)
        d[f"{t}_diff"] = d[f"blue_{t}"].fillna(0) - d[f"red_{t}"].fillna(0)
        d[f"blue_has_{t}"] = (d[f"blue_{t}"].fillna(0) >= 1).astype(float)

    # Dominant dragon type on map (argmax of map counts)
    type_mat = np.column_stack([d[f"map_{t}"].fillna(0).values for t in DRAGON_TYPES])
    dom_idx = type_mat.argmax(axis=1)
    dom_val = type_mat.max(axis=1)
    d["dominant_drake"] = [DRAGON_TYPES[i] if dom_val[j] > 0 else "none" for j, i in enumerate(dom_idx)]
    d["infernal_map"] = (d["map_infernals"] >= 1).astype(float)
    d["ocean_map"] = (d["map_oceans"] >= 1).astype(float)
    d["mountain_map"] = (d["map_mountains"] >= 1).astype(float)
    d["chemtech_map"] = (d["map_chemtechs"] >= 1).astype(float)
    d["hextech_map"] = (d["map_hextechs"] >= 1).astype(float)

    # Likely first-drake type: firstdragon side has exactly one elemental type with count>=1
    def likely_first_type(row, side: str) -> str:
        if not row.get(f"{side}_firstdragon"):
            return "not_first"
        present = [t for t in DRAGON_TYPES if (row.get(f"{side}_{t}") or 0) >= 1]
        if len(present) == 1:
            return present[0]
        return "mixed"

    d["blue_likely_first_drake"] = d.apply(lambda r: likely_first_type(r, "blue"), axis=1)
    d["map_likely_first_drake"] = np.where(
        d["blue_firstdragon"] == 1,
        d["blue_likely_first_drake"],
        d.apply(lambda r: likely_first_type(r, "red"), axis=1),
    )

    # Baron proxies (no clock — use gold@20/@25 state)
    d["blue_took_baron"] = (d["blue_firstbaron"] == 1).astype(float)
    d["gold20"] = d["blue_golddiffat20"]
    d["gold25"] = d["blue_golddiffat25"]
    d["gold15"] = d["blue_golddiffat15"]
    d["gold10"] = d["blue_golddiffat10"]
    # "Baron at even/behind/ahead" among first-baron maps
    d["baron_at20_state"] = pd.cut(
        d["gold20"],
        bins=[-1e9, -2000, -500, 500, 2000, 1e9],
        labels=["behind2k", "behind", "even", "ahead", "ahead2k"],
    )
    d["flipped_15_to_20"] = ((d["gold15"] < -500) & (d["gold20"] > 500)).astype(float)
    d["collapsed_15_to_20"] = ((d["gold15"] > 500) & (d["gold20"] < -500)).astype(float)
    # Early vs late game with baron: short games with baron ≈ earlier decisive baron
    d["baron_and_short"] = ((d["blue_firstbaron"].fillna(0) + d["red_firstbaron"].fillna(0) >= 1) & (d["length_min"] < 30)).astype(float)
    d["baron_and_long"] = ((d["blue_firstbaron"].fillna(0) + d["red_firstbaron"].fillna(0) >= 1) & (d["length_min"] >= 35)).astype(float)

    d["under_29_5"] = (d["total_kills"] <= 29).astype(float)
    d["under_32_5"] = (d["total_kills"] <= 32).astype(float)
    d["long_35"] = (d["length_min"] >= 35).astype(float)

    # Side sequence proxies
    d["blue_fb_then_fd"] = ((d["blue_firstblood"] == 1) & (d["blue_firstdragon"] == 1)).astype(float)
    d["blue_fb_no_fd"] = ((d["blue_firstblood"] == 1) & (d["blue_firstdragon"] != 1)).astype(float)
    d["blue_grubs_and_herald"] = ((d["grub_diff"] >= 2) & (d["blue_firstherald"] == 1)).astype(float)
    d["blue_grubs_no_herald"] = ((d["grub_diff"] >= 2) & (d["blue_firstherald"] != 1)).astype(float)

    # Lead conversion: ahead@15 → win, stratified by infernal presence
    d["ahead15"] = (d["gold15"] > 1000).astype(float)
    d["behind15"] = (d["gold15"] < -1000).astype(float)
    return d.copy()


def side_baselines(df: pd.DataFrame) -> dict:
    return {
        "blue_wr": round(float(df["y_blue_win"].mean()), 4),
        "n": int(len(df)),
        "blue_firstblood_rate": round(float(df["blue_firstblood"].mean()), 4),
        "blue_firstdragon_rate": round(float(df["blue_firstdragon"].mean()), 4),
        "blue_firstherald_rate": round(float(df["blue_firstherald"].dropna().mean()), 4),
        "blue_firstbaron_rate": round(float(df["blue_firstbaron"].dropna().mean()), 4),
        "blue_firsttower_rate": round(float(df["blue_firsttower"].mean()), 4),
        "mean_grub_diff": round(float(df["grub_diff"].mean()), 3),
    }


def first_objective_wr(df: pd.DataFrame) -> dict:
    out = {}
    for obj, col in [
        ("firstblood", "blue_firstblood"),
        ("firstdragon", "blue_firstdragon"),
        ("firstherald", "blue_firstherald"),
        ("firstbaron", "blue_firstbaron"),
        ("firsttower", "blue_firsttower"),
    ]:
        got = df[df[col] == 1]
        missed = df[df[col] == 0]
        out[obj] = {
            "wr_if_blue_takes": round(float(got["y_blue_win"].mean()), 4) if len(got) >= 80 else None,
            "n_takes": int(len(got)),
            "wr_if_blue_misses": round(float(missed["y_blue_win"].mean()), 4) if len(missed) >= 80 else None,
            "n_misses": int(len(missed)),
            "delta_pp": (
                round((got["y_blue_win"].mean() - missed["y_blue_win"].mean()) * 100, 2)
                if len(got) >= 80 and len(missed) >= 80
                else None
            ),
        }
    return out


def dragon_type_edges(df: pd.DataFrame) -> dict:
    """Does map dragon diet change WR / pace / conversion?"""
    by_dom = _wr_table(df[df["dominant_drake"] != "none"], "dominant_drake")

    # When blue takes FIRST dragon: which type does blue *end up* stacking?
    # (OE lacks first-drake type label; end-stack among first-takers is the usable proxy.)
    blue_fd = df[df["blue_firstdragon"] == 1].copy()
    by_blue_stack = []
    for t in DRAGON_TYPES:
        g = blue_fd[blue_fd[f"blue_{t}"].fillna(0) >= 1]
        g0 = blue_fd[blue_fd[f"blue_{t}"].fillna(0) < 1]
        if len(g) < 150:
            continue
        by_blue_stack.append(
            {
                "type": t,
                "n_blue_fd_and_has_type": int(len(g)),
                "wr": round(float(g["y_blue_win"].mean()), 4),
                "wr_fd_without_type": round(float(g0["y_blue_win"].mean()), 4) if len(g0) >= 150 else None,
                "delta_pp_vs_fd_without": (
                    round((g["y_blue_win"].mean() - g0["y_blue_win"].mean()) * 100, 2) if len(g0) >= 150 else None
                ),
                "mean_kills": round(float(g["total_kills"].mean()), 2),
                "mean_length": round(float(g["length_min"].mean()), 2),
            }
        )
    by_blue_stack.sort(key=lambda x: -(x["delta_pp_vs_fd_without"] or -999))

    # Pace by dominant type (full map diet)
    pace_by_dom = []
    for t, g in df[df["dominant_drake"] != "none"].groupby("dominant_drake"):
        if len(g) < 200:
            continue
        pace_by_dom.append(
            {
                "dominant": t,
                "n": int(len(g)),
                "mean_kills": round(float(g["total_kills"].mean()), 2),
                "mean_length": round(float(g["length_min"].mean()), 2),
                "p_under_29_5": round(float(g["under_29_5"].mean()), 4),
                "p_under_32_5": round(float(g["under_32_5"].mean()), 4),
                "p_long_35": round(float(g["long_35"].mean()), 4),
                "wr_blue": round(float(g["y_blue_win"].mean()), 4),
            }
        )
    pace_by_dom.sort(key=lambda x: -x["mean_kills"])

    # Lead conversion: ahead@15 → win, with/without type on *blue*
    ahead = df[df["ahead15"] == 1]
    conv = {"any": {"n_ahead15": int(len(ahead)), "convert_wr": round(float(ahead["y_blue_win"].mean()), 4)}}
    for t in DRAGON_TYPES:
        g = ahead[ahead[f"blue_{t}"].fillna(0) >= 1]
        g0 = ahead[ahead[f"blue_{t}"].fillna(0) < 1]
        if len(g) < 200 or len(g0) < 200:
            continue
        conv[f"blue_has_{t}"] = {
            "n_ahead15": int(len(g)),
            "convert_wr": round(float(g["y_blue_win"].mean()), 4),
            "contrast_wr_no_type": round(float(g0["y_blue_win"].mean()), 4),
            "delta_pp": round((g["y_blue_win"].mean() - g0["y_blue_win"].mean()) * 100, 2),
        }

    infernal_wr = []
    for k in range(0, 4):
        g = df[df["blue_infernals"].fillna(0) == k]
        if len(g) < 80:
            continue
        infernal_wr.append({"blue_infernals": k, "n": int(len(g)), "wr": round(float(g["y_blue_win"].mean()), 4)})

    return {
        "wr_by_dominant_drake_on_map": by_dom,
        "pace_by_dominant_drake": pace_by_dom,
        "blue_firstdragon_then_type_stack": by_blue_stack,
        "ahead15_conversion_when_blue_has_type": conv,
        "wr_by_blue_infernal_count": infernal_wr,
        "soul_blue": {
            "wr_if_blue_soul": round(float(df.loc[df["soul_blue"] == 1, "y_blue_win"].mean()), 4)
            if (df["soul_blue"] == 1).sum() >= 80
            else None,
            "n": int((df["soul_blue"] == 1).sum()),
        },
        "elder_blue": {
            "wr_if_blue_elder": round(float(df.loc[df["elder_blue"] == 1, "y_blue_win"].mean()), 4)
            if (df["elder_blue"] == 1).sum() >= 40
            else None,
            "n": int((df["elder_blue"] == 1).sum()),
        },
        "side_quirk": {
            "blue_firstdragon_rate": round(float(df["blue_firstdragon"].mean()), 4),
            "note": "Blue takes first dragon only ~40% in this OE slice — red-favored FD is a real side quirk to price.",
        },
    }


def void_grub_edges(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["grub_diff", "y_blue_win"])
    # WR by grub diff buckets
    buckets = []
    for lo, hi, lab in [(-99, -3, "red_sweep"), (-3, -0.5, "red_edge"), (-0.5, 0.5, "even"), (0.5, 3, "blue_edge"), (3, 99, "blue_sweep")]:
        g = sub[(sub["grub_diff"] > lo) & (sub["grub_diff"] <= hi)]
        if lo == -99:
            g = sub[sub["grub_diff"] <= hi]
        if hi == 99:
            g = sub[sub["grub_diff"] > lo]
        if len(g) < 100:
            continue
        buckets.append(
            {
                "bucket": lab,
                "n": int(len(g)),
                "wr_blue": round(float(g["y_blue_win"].mean()), 4),
                "mean_gold15": round(float(g["gold15"].mean()), 1) if g["gold15"].notna().any() else None,
                "p_blue_firsttower": round(float(g["blue_firsttower"].mean()), 4),
                "p_blue_firstherald": round(float(g["blue_firstherald"].dropna().mean()), 4),
                "mean_kills": round(float(g["total_kills"].mean()), 2),
            }
        )

    # Correlations
    corrs = {}
    for outc in ["y_blue_win", "gold15", "gold20", "blue_firsttower", "blue_firstherald", "total_kills", "under_29_5"]:
        if outc not in sub.columns:
            continue
        r, n = _r(sub["grub_diff"].values, sub[outc].astype(float).values)
        if r is not None:
            corrs[outc] = {"r": round(r, 4), "n": n}

    # Grubs without converting herald — trap?
    trap = sub[sub["grub_diff"] >= 2]
    got_h = trap[trap["blue_firstherald"] == 1]
    no_h = trap[trap["blue_firstherald"] == 0]
    return {
        "wr_by_grub_diff_bucket": buckets,
        "correlations_grub_diff": corrs,
        "grub_edge_with_vs_without_herald": {
            "wr_grubs_plus_herald": round(float(got_h["y_blue_win"].mean()), 4) if len(got_h) >= 80 else None,
            "n_with": int(len(got_h)),
            "wr_grubs_no_herald": round(float(no_h["y_blue_win"].mean()), 4) if len(no_h) >= 80 else None,
            "n_without": int(len(no_h)),
            "delta_pp": (
                round((got_h["y_blue_win"].mean() - no_h["y_blue_win"].mean()) * 100, 2)
                if len(got_h) >= 80 and len(no_h) >= 80
                else None
            ),
        },
    }


def baron_proxy_edges(df: pd.DataFrame) -> dict:
    """No baron timestamp — use gold@20/@25 + firstbaron + length."""
    took = df[df["blue_firstbaron"] == 1].dropna(subset=["gold20", "y_blue_win"])
    by_state = []
    if len(took) >= 100:
        for state, g in took.groupby("baron_at20_state", observed=False):
            if len(g) < 60:
                continue
            by_state.append(
                {
                    "gold20_when_blue_firstbaron": str(state),
                    "n": int(len(g)),
                    "wr": round(float(g["y_blue_win"].mean()), 4),
                    "mean_gold20": round(float(g["gold20"].mean()), 1),
                    "mean_length": round(float(g["length_min"].mean()), 2),
                    "mean_gold25": round(float(g["gold25"].mean()), 1) if g["gold25"].notna().any() else None,
                }
            )

    # Red first baron while blue ahead@20 — steal?
    red_baron = df[df["red_firstbaron"] == 1].dropna(subset=["gold20"])
    steal = red_baron[red_baron["gold20"] > 1000]  # blue was ahead but red got baron
    normal = red_baron[red_baron["gold20"] <= 1000]

    # Flip into baron
    flipped = df[(df["flipped_15_to_20"] == 1) & (df["blue_firstbaron"] == 1)]
    collapsed = df[(df["collapsed_15_to_20"] == 1) & (df["blue_firstbaron"] == 1)]

    return {
        "blue_firstbaron_wr_by_gold20_state": by_state,
        "note": "gold@20 is a proxy for 'around first baron window', not exact baron minute",
        "red_baron_while_blue_ahead20": {
            "n": int(len(steal)),
            "blue_wr_still": round(float(steal["y_blue_win"].mean()), 4) if len(steal) >= 40 else None,
            "contrast_n": int(len(normal)),
            "blue_wr_when_red_baron_not_ahead": round(float(normal["y_blue_win"].mean()), 4) if len(normal) >= 80 else None,
        },
        "blue_baron_after_15to20_flip": {
            "n": int(len(flipped)),
            "wr": round(float(flipped["y_blue_win"].mean()), 4) if len(flipped) >= 40 else None,
        },
        "blue_baron_after_15to20_collapse": {
            "n": int(len(collapsed)),
            "wr": round(float(collapsed["y_blue_win"].mean()), 4) if len(collapsed) >= 40 else None,
        },
        "short_game_with_any_baron": {
            "n": int(df["baron_and_short"].sum()),
            "blue_wr_if_blue_baron": round(
                float(df[(df["baron_and_short"] == 1) & (df["blue_firstbaron"] == 1)]["y_blue_win"].mean()), 4
            )
            if ((df["baron_and_short"] == 1) & (df["blue_firstbaron"] == 1)).sum() >= 40
            else None,
        },
    }


def sequence_and_side_traps(df: pd.DataFrame) -> dict:
    return {
        "blue_fb_then_fd": {
            "n": int(df["blue_fb_then_fd"].sum()),
            "wr": round(float(df.loc[df["blue_fb_then_fd"] == 1, "y_blue_win"].mean()), 4),
        },
        "blue_fb_no_fd": {
            "n": int(df["blue_fb_no_fd"].sum()),
            "wr": round(float(df.loc[df["blue_fb_no_fd"] == 1, "y_blue_win"].mean()), 4),
        },
        "delta_pp_fd_after_fb": round(
            (
                df.loc[df["blue_fb_then_fd"] == 1, "y_blue_win"].mean()
                - df.loc[df["blue_fb_no_fd"] == 1, "y_blue_win"].mean()
            )
            * 100,
            2,
        ),
        "blue_side_wr_by_league_top": _wr_table(df, "league")[:12],
    }


def residual_board(df: pd.DataFrame) -> list[dict]:
    """Prematch-ish / early live features → outcomes, creative set."""
    preds = [
        "grub_diff",
        "herald_diff",
        "infernal_map",
        "ocean_map",
        "mountain_map",
        "chemtech_map",
        "hextech_map",
        "infernals_diff",
        "soul_blue",
        "gold10",
        "gold15",
        "gold20",
        "flipped_15_to_20",
        "blue_firstdragon",
        "blue_firstherald",
        "blue_firstbaron",
        "blue_fb_then_fd",
        "blue_grubs_and_herald",
        "blue_grubs_no_herald",
    ]
    outs = ["y_blue_win", "under_29_5", "under_32_5", "total_kills", "long_35", "gold20"]
    edges = []
    for p in preds:
        if p not in df.columns:
            continue
        for o in outs:
            if o not in df.columns or p == o:
                continue
            r, n = _r(df[p].astype(float).values, df[o].astype(float).values, min_n=250)
            if r is None or abs(r) < 0.05:
                continue
            edges.append({"feature": p, "outcome": o, "r": round(r, 4), "n": n, "abs_r": abs(r)})
    edges.sort(key=lambda x: -x["abs_r"])
    return edges[:50]


def creative_findings(df: pd.DataFrame, report: dict) -> list[str]:
    """Narrative bullets for the weird stuff."""
    bullets = []
    sb = report["side_baselines"]
    bullets.append(
        f"Blue-side baseline WR {sb['blue_wr']:.1%} but first-dragon rate only {sb['blue_firstdragon_rate']:.1%} "
        f"while first-herald {sb['blue_firstherald_rate']:.1%} and mean grub_diff {sb['mean_grub_diff']:+.2f} — "
        f"blue wins the *grub/herald* layer more than the *drake* layer."
    )
    fo = report["first_objectives"]
    bullets.append(
        "First-obj ΔWR (blue takes vs misses): "
        + ", ".join(f"{k} {v['delta_pp']:+.1f}pp" for k, v in fo.items() if v.get("delta_pp") is not None)
        + " — Herald (+26pp) punches above FD (+21pp); Baron is the nuke (+63pp)."
    )
    g = report["void_grubs"]["grub_edge_with_vs_without_herald"]
    if g.get("delta_pp") is not None:
        bullets.append(
            f"Void grub edge (≥+2) WITHOUT Herald: blue WR {g['wr_grubs_no_herald']:.1%} (n={g['n_without']}) vs "
            f"WITH Herald {g['wr_grubs_plus_herald']:.1%} (n={g['n_with']}, Δ {g['delta_pp']:+.1f}pp). "
            f"Live read: grubs alone are often a *trap lead*."
        )
    gb = report["void_grubs"]["wr_by_grub_diff_bucket"]
    if gb:
        bullets.append(
            "Grub ladder WR: "
            + " · ".join(f"{b['bucket']} {b['wr_blue']:.1%} (FT {b['p_blue_firsttower']:.0%})" for b in gb)
        )
    pace = report["dragon_types"].get("pace_by_dominant_drake") or []
    if pace:
        hi_k, lo_k = pace[0], min(pace, key=lambda x: x["mean_kills"])
        hi_l = max(pace, key=lambda x: x["mean_length"])
        bullets.append(
            f"Map dragon diet → pace: dominant {hi_k['dominant']} averages {hi_k['mean_kills']:.1f} kills / "
            f"{hi_k['mean_length']:.1f}m; {lo_k['dominant']} {lo_k['mean_kills']:.1f}k; "
            f"longest diet = {hi_l['dominant']} ({hi_l['mean_length']:.1f}m, under29 {hi_l['p_under_29_5']:.1%})."
        )
    stack = report["dragon_types"].get("blue_firstdragon_then_type_stack") or []
    if stack:
        best, worst = stack[0], stack[-1]
        bullets.append(
            f"After blue FD, stacking {best['type']} adds {best['delta_pp_vs_fd_without']:+.1f}pp WR vs FD without that type; "
            f"{worst['type']} adds {worst['delta_pp_vs_fd_without']:+.1f}pp — type path matters after the first take."
        )
    conv = report["dragon_types"].get("ahead15_conversion_when_blue_has_type") or {}
    typed = [(k, v) for k, v in conv.items() if k != "any" and v.get("delta_pp") is not None]
    if typed:
        typed.sort(key=lambda x: -x[1]["delta_pp"])
        b, w = typed[0], typed[-1]
        bullets.append(
            f"Ahead@15 conversion with blue holding type: best {b[0]} {b[1]['delta_pp']:+.1f}pp vs no-type; "
            f"worst {w[0]} {w[1]['delta_pp']:+.1f}pp."
        )
    inf = report["dragon_types"]["wr_by_blue_infernal_count"]
    if len(inf) >= 3:
        bullets.append(
            "Blue Infernal count ladder: "
            + " · ".join(f"{x['blue_infernals']}→{x['wr']:.1%} (n={x['n']})" for x in inf)
        )
    bstates = report["baron_proxies"]["blue_firstbaron_wr_by_gold20_state"]
    if len(bstates) >= 2:
        best, worst = max(bstates, key=lambda x: x["wr"]), min(bstates, key=lambda x: x["wr"])
        bullets.append(
            f"Blue first Baron ≠ auto-win: WR {best['wr']:.1%} if gold@20 {best['gold20_when_blue_firstbaron']} "
            f"vs {worst['wr']:.1%} if {worst['gold20_when_blue_firstbaron']} "
            f"(same first-Baron flag — OE has no baron minute, @20 is the window proxy)."
        )
    steal = report["baron_proxies"]["red_baron_while_blue_ahead20"]
    if steal.get("blue_wr_still") is not None:
        bullets.append(
            f"Red Baron while blue ahead@20: blue still wins {steal['blue_wr_still']:.1%} (n={steal['n']}) vs "
            f"{steal['blue_wr_when_red_baron_not_ahead']:.1%} when red Baron and blue not ahead — theft hurts but isn't free."
        )
    flip = report["baron_proxies"]["blue_baron_after_15to20_flip"]
    col = report["baron_proxies"]["blue_baron_after_15to20_collapse"]
    if flip.get("wr") and col.get("wr"):
        bullets.append(
            f"Blue Baron after 15→20 gold flip: WR {flip['wr']:.1%} (n={flip['n']}); "
            f"after 15→20 collapse still taking Baron: WR {col['wr']:.1%} (n={col['n']})."
        )
    seq = report["sequences"]
    bullets.append(
        f"Sequence: blue FB+FD WR {seq['blue_fb_then_fd']['wr']:.1%} vs FB without FD {seq['blue_fb_no_fd']['wr']:.1%} "
        f"(Δ {seq['delta_pp_fd_after_fb']:+.1f}pp)."
    )
    soul = report["dragon_types"]["soul_blue"]
    elder = report["dragon_types"]["elder_blue"]
    if soul.get("wr_if_blue_soul") is not None:
        bullets.append(f"Blue soul (4+ elementals): WR {soul['wr_if_blue_soul']:.1%} (n={soul['n']}).")
    if elder.get("wr_if_blue_elder") is not None:
        bullets.append(
            f"Blue Elder: WR {elder['wr_if_blue_elder']:.1%} (n={elder['n']}) — high but not 95%+; Elder games are often already decided both ways."
        )
    return bullets


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[sideobj] ingesting raw OE team rows (all years)…")
    raw = load_oe_team_maps()
    print(f"[sideobj] maps={len(raw)} date {raw['date'].min().date()} → {raw['date'].max().date()}")
    df = engineer(raw)
    print("[sideobj] engineering + slicing…")

    report = {
        "version": 1,
        "n_maps": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "data_limits": [
            "OE has no baron *minute* — gold@20/@25 + firstbaron used as window proxy.",
            "OE has no labeled first-dragon *type* — inferred only when taker has exactly one elemental type.",
            "Void grubs / dragon types exist in raw OE but were missing from warehouse maps.parquet until this study.",
        ],
        "side_baselines": side_baselines(df),
        "first_objectives": first_objective_wr(df),
        "dragon_types": dragon_type_edges(df),
        "void_grubs": void_grub_edges(df),
        "baron_proxies": baron_proxy_edges(df),
        "sequences": sequence_and_side_traps(df),
        "correlation_board": residual_board(df),
    }
    report["creative_findings"] = creative_findings(df, report)

    path = MODELS_DIR / "side_objective_edges.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[sideobj] wrote {path}")
    print("\n=== CREATIVE FINDINGS ===")
    for b in report["creative_findings"]:
        print(f" • {b}")
    print("\n=== TOP CORR BOARD ===")
    for e in report["correlation_board"][:15]:
        print(f"  {e['feature']:28} → {e['outcome']:14} r={e['r']:+.3f} n={e['n']}")


if __name__ == "__main__":
    main()
