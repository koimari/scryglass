## Learned User Preferences

- Prefers Cursor chat over CLI for LoL board/betting; never push `python -m …` commands for day-to-day board usage
- Fair-odds-first: on draft, lead with model fair odds; book decimal > fair ⇒ +EV; when book odds are present, show Actual alongside Fair so edge size is obvious; odds only later for GRADE/Kelly
- Prefer Slip Composer text paste over screenshot OCR for drafts and live state; screenshots often misread champs and dragons
- Trust the user’s stated draft over tab/HUD OCR; when they correct a read, acknowledge, restate, and recompute — do not argue
- Count dragon icons carefully per side (do not round; 3 ≠ 2); do not confuse grub shields with dragons; current-season void grubs max is 3 per side (not 6)
- Full Kelly is OK on small recreational bankrolls when the user explicitly asks for it
- Cashout advice should be data-driven (live survival / model EV vs offered cashout), not gut-only
- When odds are moving, give actionable thresholds (“if book ≥ X → take/hold”), not only a static table
- Simples / bet-slip paste → one-view board (favorite %, fair odds, kills bucket/median); when grading a ticket, order by edge
- Research claims should be isolation-controlled with high-precision numbers plus context — not vague single-% summaries

## Learned Workspace Facts

- LoL offline betting research engine lives in `lol_kills`; draft boards via `build_board` / `format_fair_odds` in `lol_kills/board.py`
- Slip Composer UI is at `web/composer/` (`index.html`); paste block is ground truth for chat
- GRADE / Kelly / combo grading lives in `lol_kills/econ.py` (`grade_bet` / `grade_combo`)
- Draft Score is league-calibrated (v3 classic + **v4 composite**): `lol_kills.draft_phase_score.draft_score_composite` emits @10/@15/@20/@25 curve + Flores beatdown/control roles; fit via `lol_kills.research.draft_phase_beatdown` → `data/lol/models/draft_phase_beatdown.json`. Classic scalar still in `lol_kills.draft_score` / `draft_wr_calibration.json`
- Beatdown framing (Mike Flores, “Who’s the Beatdown?”): the early-damage side must convert by ~15–20; the inevitability/control side must weather and win late — misassignment of role is a known loss mode; live_win uses the phase-bucket `draft_edge` for the current minute
- Dual Elo (regional μ + international/meta μ) supports EWC and other intl events (`lol_kills/ratings/dual_elo.py`)
- Team aliases include KC → Karmine Corp and DK → Dplus Kia (`web/composer/teams.json`, `lol_kills/etl/aliases.py`)
- Primary warehouse is Oracle’s Elixir CSVs; agent-only refresh via `lol_kills.refresh_warehouse` / `lol_kills.pipeline`
- Bet-slip one-view skill: `.cursor/skills/bet-slip-view` → `lol_kills.skills.slip_view`
- Draft × time × objectives advantage matrix: `lol_kills/research/draft_advantage_matrix.py` → `data/lol/models/draft_advantage_matrix.json`
- Objective / void-grub WR isolation: `lol_kills/research/side_objective_edges.py` and `grubs_isolation_study.py`
- First-inhib board lean is a known overconfident head (board shop leans exclude it)
