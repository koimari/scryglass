/**
 * Browser port of lol_kills.live_win softcap / OE-matrix path + ablation Δpp.
 * Coefs: ./coefs/draft_live_coefs.json (+ optional grubs_decision_numbers.json).
 */
(function (global) {
  "use strict";

  const FALLBACK = {
    kill_diff: 0.1,
    gold_diff_k: 0.18,
    dragon: 0.22,
    infernal_extra: 0.05,
    void_grub: 0.06,
    tower: 0.18,
    time_decay: 0.012,
    adv_cap: 1.35,
  };

  const GRUB_CLOCK_END = 14.75;

  let _coefs = null;
  let _grubsArt = null;
  let _loadPromise = null;

  function logit(p) {
    p = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
    return Math.log(p / (1 - p));
  }

  function sigmoid(x) {
    if (x > 30) return 1;
    if (x < -30) return 0;
    return 1 / (1 + Math.exp(-x));
  }

  function phaseOf(minute) {
    if (minute < 14) return "early";
    if (minute < 25) return "mid";
    return "late";
  }

  function goldBin(g) {
    const edges = [-1e9, -3000, -2000, -1000, -500, 500, 1000, 2000, 3000, 1e9];
    const labels = [
      "le-3k", "-3k--2k", "-2k--1k", "-1k--500", "even",
      "+500-1k", "+1k-2k", "+2k-3k", "ge+3k",
    ];
    for (let i = 0; i < edges.length - 1; i++) {
      if (edges[i] <= g && g < edges[i + 1]) return labels[i];
    }
    return "even";
  }

  async function loadCoefs(base = ".") {
    if (_coefs) return _coefs;
    if (_loadPromise) return _loadPromise;
    _loadPromise = (async () => {
      const [cRes, gRes] = await Promise.all([
        fetch(`${base}/coefs/draft_live_coefs.json`, { cache: "force-cache" }),
        fetch(`${base}/coefs/grubs_decision_numbers.json`, { cache: "force-cache" }),
      ]);
      _coefs = cRes.ok ? await cRes.json() : {};
      _grubsArt = gRes.ok ? await gRes.json() : null;
      return _coefs;
    })();
    return _loadPromise;
  }

  function liveWinProb(opts) {
    const {
      p_pre,
      minute,
      kill_diff,
      gold_diff,
      dragons = 0,
      opp_dragons = 0,
      infernal = false,
      void_grubs = 0,
      towers = 0,
      opp_towers = 0,
      draft_edge = null,
      kill_conc_diff = null,
      scaling_flag = null,
      blue_hypercarry = null,
      draft_q = null,
      first_dragon = null,
      first_herald = null,
      first_tower = null,
    } = opts;

    const phase = phaseOf(minute);
    const coefs = _coefs || {};
    const phase_c = (coefs.phase_coefs || {})[phase];

    const dragon_diff = dragons - opp_dragons;
    const tower_diff = towers - opp_towers;
    const gold_k = gold_diff / 1000;
    const edge = draft_edge != null ? +draft_edge : 0;
    const conc = kill_conc_diff != null ? +kill_conc_diff : 0;
    const scal = +(scaling_flag || 0);
    const bhc = blue_hypercarry != null ? +blue_hypercarry : scal;

    let x = logit(p_pre);
    x *= Math.max(0.4, 1 - FALLBACK.time_decay * minute);

    let method = "softcap_fallback";
    let adv = 0;
    let matrix_cell = null;

    if (phase_c && draft_edge != null) {
      method = `oe_matrix:${phase}`;
      const priors = coefs.live_obj_priors || {};
      adv += +(phase_c.draft_edge || 0) * edge * 0.35;
      adv += +(phase_c.gold_k || 0.18) * gold_k;
      if (first_dragon != null) adv += +(phase_c.first_dragon || 0) * +first_dragon;
      if (first_herald != null) adv += +(phase_c.first_herald || 0) * +first_herald;
      if (first_tower != null) adv += +(phase_c.first_tower || 0) * +first_tower;
      adv += +(phase_c.draft_x_gold || 0) * edge * gold_k;
      adv += +(phase_c.conc_x_gold || 0) * conc * gold_k;
      adv += +(phase_c.scaling_x_gold || 0) * scal * gold_k;
      adv += +(phase_c.blue_carry_x_gold || 0) * bhc * gold_k;
      adv += +(priors.dragon_diff ?? FALLBACK.dragon) * dragon_diff;
      adv += +(priors.tower_diff ?? FALLBACK.tower) * tower_diff;
      adv += +(priors.void_grub ?? FALLBACK.void_grub) * +void_grubs;
      adv += +(priors.kill_diff ?? FALLBACK.kill_diff) * +kill_diff;
      if (infernal && dragons > 0) {
        adv += +(priors.infernal_extra ?? FALLBACK.infernal_extra);
      }
      const cap = +(coefs.adv_cap || FALLBACK.adv_cap);
      adv = cap * Math.tanh(adv / cap);
      matrix_cell = {
        phase,
        gold_bin: goldBin(gold_diff),
        draft_q: draft_q != null ? +draft_q : 2,
        draft_edge: Math.round(edge * 10000) / 10000,
      };
    } else {
      adv += FALLBACK.kill_diff * kill_diff;
      adv += FALLBACK.gold_diff_k * gold_k;
      adv += FALLBACK.dragon * dragon_diff;
      if (infernal && dragons > 0) adv += FALLBACK.infernal_extra;
      adv += FALLBACK.void_grub * +void_grubs;
      adv += FALLBACK.tower * tower_diff;
      const cap = FALLBACK.adv_cap;
      adv = cap * Math.tanh(adv / cap);
      matrix_cell = { phase, gold_bin: goldBin(gold_diff), mode: "fallback" };
    }

    x += adv;
    const p = sigmoid(x);
    return {
      p_win: Math.round(p * 10000) / 10000,
      p_pre: Math.round(p_pre * 10000) / 10000,
      logit: Math.round(x * 10000) / 10000,
      adv_logit: Math.round(adv * 10000) / 10000,
      minute,
      phase,
      matrix_cell,
      features: {
        kill_diff,
        gold_diff,
        dragons,
        opp_dragons,
        infernal,
        void_grubs,
        towers,
        opp_towers,
        draft_edge,
      },
      method,
    };
  }

  function grubsResearch(minute, voidNet) {
    if (minute > GRUB_CLOCK_END || !_grubsArt) return null;
    const dpp = (_grubsArt.deltas_pp || {}).win_minus_leave_mix;
    if (dpp == null) return null;
    return {
      label: "grubs_research",
      estimand: "win_minus_leave_mix (trailing contest vs leave-mix)",
      delta_pp: Math.round(+dpp * 100) / 100,
      note:
        "Contest research only — not added into live p_win. " +
        `Live void_grub prior stays separate (net=${voidNet}).`,
      breakeven_p_win_fight_vs_leave:
        (_grubsArt.breakeven_p_win_fight || {}).vs_leave_mix ?? null,
    };
  }

  function decideCashout({ p_win, stake, odds, cashout }) {
    const payout = stake * odds;
    const ev_hold = p_win * payout;
    const ev_cashout = cashout;
    const edge = ev_hold - ev_cashout;
    let verdict;
    let reason;
    if (edge > stake * 0.15) {
      verdict = "HOLD";
      reason = `Hold EV exceeds cashout by R$${edge.toFixed(2)} (>15% of stake).`;
    } else if (edge < -stake * 0.1) {
      verdict = "CASHOUT";
      reason = `Cashout beats hold EV by R$${(-edge).toFixed(2)}.`;
    } else if (Math.abs(edge) <= stake * 0.05) {
      verdict = edge >= 0 ? "HOLD" : "CASHOUT";
      reason = `Too close (edge R$${edge >= 0 ? "+" : ""}${edge.toFixed(2)}); default to ${
        edge >= 0 ? "hold locked price" : "take cash"
      }.`;
    } else {
      verdict = edge > 0 ? "HOLD" : "CASHOUT";
      reason = `Edge R$${edge >= 0 ? "+" : ""}${edge.toFixed(2)} vs cashout.`;
    }
    return {
      verdict,
      reason,
      ev_hold: Math.round(ev_hold * 100) / 100,
      ev_cashout,
      edge_hold_vs_cashout: Math.round(edge * 100) / 100,
      fair_cashout: Math.round(ev_hold * 100) / 100,
      implied_by_cashout: Math.round((cashout / payout) * 10000) / 10000,
    };
  }

  /**
   * @param {object} opts — same fields as liveWinProb, plus optional stake/odds/cashout
   *   and void_grubs_blue / void_grubs_red (converted to net).
   */
  function objectiveDeltaPpBreakdown(opts) {
    let voidNet = +(opts.void_grubs || 0);
    if (opts.void_grubs_blue != null || opts.void_grubs_red != null) {
      voidNet = +(opts.void_grubs_blue || 0) - +(opts.void_grubs_red || 0);
    }

    const base = {
      p_pre: +opts.p_pre,
      minute: +opts.minute,
      kill_diff: +opts.kill_diff,
      gold_diff: +opts.gold_diff,
      dragons: +(opts.dragons || 0),
      opp_dragons: +(opts.opp_dragons || 0),
      infernal: !!opts.infernal,
      void_grubs: voidNet,
      towers: +(opts.towers || 0),
      opp_towers: +(opts.opp_towers || 0),
      draft_edge: opts.draft_edge != null ? +opts.draft_edge : null,
      kill_conc_diff: opts.kill_conc_diff != null ? +opts.kill_conc_diff : null,
      scaling_flag: opts.scaling_flag != null ? +opts.scaling_flag : null,
      blue_hypercarry: opts.blue_hypercarry != null ? +opts.blue_hypercarry : null,
      draft_q: opts.draft_q != null ? +opts.draft_q : null,
      first_dragon: opts.first_dragon != null ? +opts.first_dragon : null,
      first_herald: opts.first_herald != null ? +opts.first_herald : null,
      first_tower: opts.first_tower != null ? +opts.first_tower : null,
    };

    const full = liveWinProb(base);
    const pFull = full.p_win;

    const channels = {
      gold: { gold_diff: 0 },
      kills: { kill_diff: 0 },
      dragons: { dragons: 0, opp_dragons: 0, infernal: false },
      towers: { towers: 0, opp_towers: 0 },
      void_grubs: { void_grubs: 0 },
    };
    if (base.first_dragon != null) channels.first_dragon = { first_dragon: 0 };
    if (base.first_herald != null) channels.first_herald = { first_herald: 0 };
    if (base.first_tower != null) channels.first_tower = { first_tower: 0 };

    const deltas = {};
    for (const [name, ov] of Object.entries(channels)) {
      const ab = liveWinProb({ ...base, ...ov });
      deltas[name] = Math.round(100 * (pFull - ab.p_win) * 100) / 100;
    }

    const top = Object.entries(deltas)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .filter(([, v]) => Math.abs(v) >= 0.05)
      .slice(0, 6)
      .map(([channel, delta_pp]) => ({ channel, delta_pp }));

    const out = {
      p_win: pFull,
      p_pre: base.p_pre,
      fair_odds: Math.round((1 / Math.max(pFull, 1e-6)) * 1000) / 1000,
      fair_odds_opp: Math.round((1 / Math.max(1 - pFull, 1e-6)) * 1000) / 1000,
      delta_vs_pre_pp: Math.round(100 * (pFull - base.p_pre) * 100) / 100,
      phase: full.phase,
      minute: base.minute,
      method: full.method,
      deltas_pp: deltas,
      top,
      features: { ...full.features, void_grubs_net: voidNet },
      grubs_research: grubsResearch(base.minute, voidNet),
      note:
        "Δpp via ablation of live_win softcap model. grubs_research is contest estimand — not in p_win.",
    };

    if (opts.stake != null && opts.odds != null) {
      const stake = +opts.stake;
      const odds = +opts.odds;
      const payout = stake * odds;
      out.ticket = {
        stake,
        odds,
        payout: Math.round(payout * 100) / 100,
        fair_cashout: Math.round(pFull * payout * 100) / 100,
        hold_ev: Math.round((pFull * payout - stake) * 100) / 100,
      };
      if (opts.cashout != null) {
        out.cashout = decideCashout({
          p_win: pFull,
          stake,
          odds,
          cashout: +opts.cashout,
        });
      }
    }
    return out;
  }

  /** Parse mm:ss → fractional minute. */
  function clockToMinute(clock) {
    if (clock == null || clock === "") return null;
    if (typeof clock === "number") return clock;
    const m = String(clock).trim().match(/^(\d{1,2}):(\d{2})$/);
    if (!m) return null;
    return +m[1] + +m[2] / 60;
  }

  global.LiveWin = {
    loadCoefs,
    liveWinProb,
    objectiveDeltaPpBreakdown,
    decideCashout,
    clockToMinute,
    FALLBACK,
  };
})(typeof window !== "undefined" ? window : globalThis);
