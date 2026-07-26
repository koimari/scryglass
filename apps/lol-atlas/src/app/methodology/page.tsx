import Link from "next/link";
import { promises as fs } from "fs";
import path from "path";
import type { ReactNode } from "react";

async function loadChangelog(): Promise<string> {
  try {
    return await fs.readFile(path.join(process.cwd(), "CHANGELOG.md"), "utf8");
  } catch {
    return "_No changelog entries yet._";
  }
}

type DraftEvaluation = {
  selected_alpha?: number;
  selected_half_life_days?: number;
  rolling_validation_mean_brier?: number;
  interaction_gates?: {
    role_residual?: number;
    ally_synergy?: number;
    cross_counter?: number;
    composition_counter?: number;
    same_role?: number;
  };
  baseline_holdout?: { n?: number; brier?: number; log_loss?: number; auc?: number };
  interaction_holdout?: { n?: number; brier?: number; log_loss?: number; auc?: number };
};

async function loadDraftEvaluation(): Promise<DraftEvaluation | null> {
  try {
    const raw = await fs.readFile(
      path.join(process.cwd(), "data", "draft", "runtime.json"),
      "utf8",
    );
    return (JSON.parse(raw) as { evaluation?: DraftEvaluation }).evaluation ?? null;
  } catch {
    return null;
  }
}

function MethodSection({
  id,
  title,
  children,
  defaultOpen = false,
}: {
  id: string;
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details id={id} className="method-details" open={defaultOpen}>
      <summary className="method-h">{title}</summary>
      <div className="method-body mt-3">{children}</div>
    </details>
  );
}

export default async function MethodologyPage() {
  const [changelog, draftEvaluation] = await Promise.all([
    loadChangelog(),
    loadDraftEvaluation(),
  ]);

  return (
    <article className="page-prose">
      <header className="page-header">
        <p className="blog-kicker">Method · Estimands</p>
        <h1 className="font-display mt-2 text-[2.25rem] leading-tight text-[var(--ink)]">
          Methodology
        </h1>
        <p className="lede">
          Written for people who will try to break it — statisticians, coaches, analysts. Every
          public number on Scryglass is meant to survive that kind of reading.
        </p>
        <div className="micro-log mt-4">
          <a className="row-link" href="/method/formulas.tex">
            Download formulas (.tex)
          </a>
          <Link className="row-link" href="/articles/void-grubs-contest-or-leave">
            Void-grubs article
          </Link>
        </div>
      </header>

      <aside className="author-stub">
        <strong>By whom</strong>
        <p>
          koi — independent LoL research. Author sidebar with bio lands in a later pass; the work is
          the citation for now.
        </p>
      </aside>

      <MethodSection id="hierarchical-ladder" title="Current public ladder" defaultOpen>
        <p>
          The published team ladder uses a regularized hierarchical Bradley–Terry fit. It estimates
          one organization effect across every event, plus a partially pooled home-league effect.
          The same Team Liquid identity is therefore used in LCS, MSI, and EWC; an event label never
          creates a second team.
        </p>
        <p>
          Bo3 and Bo5 maps are collapsed to one series observation so a long series cannot count as
          five independent matches. Historical LTA source labels are retained for audit: LTA North
          maps to LCS, LTA South maps to CBLOL, and an unqualified LTA row is treated as an Americas
          cross-region event. International events are classified from their source competition,
          so a title such as “LCK Road to MSI” remains LCK.
        </p>
        <p>
          The default adjusted rating is the one-sided 90% lower bound of the fitted rating. Teams
          without an international bridge receive wider uncertainty; this is why a thin domestic
          win streak does not automatically outrank an established LPL or LCK team. The fit is a
          penalized MAP estimate with a local Laplace uncertainty approximation, not a claim of a
          fully sampled Bayesian posterior.
        </p>
        <p>
          Player and team affiliation filters use the latest observed non-international competition,
          not career league history. Tier 1 is the six major regional leagues (LCK, LPL, LEC, LCS,
          CBLOL, and LCP); Tier 2 contains established secondary and challenger circuits such as
          TCL, LJL, CD, and NACL; Tier 3 is the remaining domestic or developmental population. A
          Tier 2 player therefore cannot enter a Tier 1 league view merely because they played there
          in an older season. On Ratings, tier chips set the current competitive level while league,
          cross-region, and international-event chips narrow the evidence scope; filters across groups
          are conjunctive.
        </p>
        <p>
          The validation contract is chronological, series-level holdouts: report log loss, Brier
          score, calibration, interval coverage, and rank stability before changing the published
          hyperparameters. Random map splits are not used because maps within a series are
          correlated. The design follows the dynamic Bradley–Terry and state-space literature: <a className="row-link" href="https://arxiv.org/abs/2003.00083">dynamic BT</a>,
          <a className="row-link" href="https://arxiv.org/abs/2308.02414"> state-space skill models</a>, and
          <a className="row-link" href="https://arxiv.org/abs/2106.11397"> team-skill aggregation evidence</a>.
        </p>
      </MethodSection>

      <MethodSection id="dual-elo" title="Sequential Dual Elo benchmark">
        <p>
          The sequential Dual Elo track remains available as a time-safe pre-match feature benchmark.
          Each team carries a regional component and an international (meta) component. They sum to
          a total rating μ with a spread σ. Outcomes in the Oracle&apos;s Elixir pack update both.
          League chips on the Ratings page filter who appears while μ stays shared.
        </p>
        <p>
          σ shrinks toward a floor as informative games arrive. Team floor is 25. When σ sits on that
          floor, the Evidence label reads <em>Settled</em> — the rating is as tight as this model allows. Headroom
          above the floor is what the adjusted rating penalizes:
        </p>
        <p className="font-mono text-sm">
          benchmark adjusted rating = μ − max(0, σ − σ_min)
        </p>
        <p>
          That soft penalty stops a thin regional spike from outranking a settled major org on the
          default ladder sort. Full LaTeX is in the formulas download.
        </p>
      </MethodSection>

      <MethodSection id="player-elo" title="Player Dual Elo">
        <p>
          Players are rated on their own Dual Elo track (player floor σ_min = 28). A team&apos;s
          player-aggregated strength is a role-weighted blend of the five on the rift. Prefer player
          ladders when rosters move. Evidence on a player at 28 is usually Settled because 28 is the
          floor.
        </p>
      </MethodSection>

      <MethodSection id="draft-score" title="Draft Sandbox evaluation">
        <p>
          The sandbox ranks a legal next pick by its expected match-win probability after that pick.
          It is not a global champion tier list. Every candidate is rescored against the champions
          already selected, the open role, the two teams, and the assigned player. Changing an ally,
          opponent, role, or lineup can therefore change both the order and the explanation.
          Repeating the same state returns the same ranking for auditability; deterministic here
          means reproducible, not context-free.
        </p>
        <p>
          The draft component is an additive log-odds model. It contains regularized champion main
          effects, validation-gated champion-by-role residuals, unordered allied-pair synergy,
          directed cross-team counters, lower-dimensional composition responses, same-role
          counters, and a player-champion comfort term. Exact
          champion interactions share evidence with overlapping composition tags such as engage,
          poke, scaling, frontline, and peel. During fitting, team and player identity enter as
          nuisance controls. Their coefficients are not relabeled as champion value. At serving
          time, current team Elo and the selected five-player lineup supply a separately calibrated
          strength prior:
        </p>
        <p className="font-mono text-sm">
          logit(Pblue) = strength prior + league scale × confidence ×
          (champions + ally synergy + counters + lane + comfort)
        </p>
        <p>
          The interaction model is fitted on 16,334 complete professional drafts from the 2025 and
          2026 Oracle&apos;s Elixir warehouse. Features are sparse and L2-regularized. Model
          selection uses three rolling-origin folds inside the first 85% of maps. Those folds choose
          the regularization level, recency half-life, and separate serving gates for allied
          champion-by-role residuals, synergy, exact counters, composition responses, and same-role
          evidence. The chronologically
          later 15% is then scored as the final temporal evaluation. Any family that does not reduce
          rolling-validation Brier error is withheld from EV with a gate of zero.
        </p>
        {draftEvaluation ? (
          <div className="method-note">
            <strong>Current serving contract</strong>
            <p>
              Recency half-life {draftEvaluation.selected_half_life_days ?? "—"} days; L2 α{" "}
              {draftEvaluation.selected_alpha ?? "—"}. Gates: champion-by-role{" "}
              {draftEvaluation.interaction_gates?.role_residual ?? 0}, synergy{" "}
              {draftEvaluation.interaction_gates?.ally_synergy ?? 0}, exact counter{" "}
              {draftEvaluation.interaction_gates?.cross_counter ?? 0}, composition response{" "}
              {draftEvaluation.interaction_gates?.composition_counter ?? 0}, same-role{" "}
              {draftEvaluation.interaction_gates?.same_role ?? 0}.
            </p>
            <p>
              Final temporal evaluation n = {draftEvaluation.interaction_holdout?.n ?? "—"}:
              interaction Brier{" "}
              {draftEvaluation.interaction_holdout?.brier?.toFixed(4) ?? "—"}, log loss{" "}
              {draftEvaluation.interaction_holdout?.log_loss?.toFixed(4) ?? "—"}, AUC{" "}
              {draftEvaluation.interaction_holdout?.auc?.toFixed(4) ?? "—"}. The no-interaction
              comparator Brier is{" "}
              {draftEvaluation.baseline_holdout?.brier?.toFixed(4) ?? "—"}.
            </p>
          </div>
        ) : null}
        <p>
          Allied synergy and response terms are fitted jointly with champion strength. This matters:
          a raw pair win rate can mistake a strong champion, team, or player for synergy. The terms
          remain predictive associations, not causal claims about what would happen if a team were
          forced onto a champion. The model design follows the separation of player preference and
          match interaction in{" "}
          <a className="row-link" href="https://arxiv.org/abs/2204.12750">DraftRec</a>, and the
          cooperation-versus-competition framing in{" "}
          <a className="row-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/16528">
            NeuralAC
          </a>
          . Full game-tree search, as in{" "}
          <a className="row-link" href="https://arxiv.org/abs/2012.10171">JueWuDraft</a>, is a
          future extension rather than something this page currently claims to run.
        </p>
        <p>
          The champion pool contains all 173 champions in the current client roster. A champion
          without professional evidence receives a zero-centered main effect and no invented
          interaction or role evidence. It remains available for manual analysis and is labeled
          <em> Neutral prior</em>. Missing evidence is never treated as proof that the champion is
          average.
        </p>
        <p>
          Player comfort is a recency-weighted champion result relative to that player&apos;s own
          baseline after the champion&apos;s global strength is also removed. It is shrunk with a
          Bayesian prior and capped before it reaches the draft model. The overall player Elo still
          enters through lineup strength. Comfort is therefore a modest player-specific adjustment,
          not a second unrestricted player rating or a duplicate champion prior.
        </p>
        <p>
          A zero interaction gate is a result, not missing UI. It means the evidence family did not
          generalize well enough to change professional recommendations in the current artifact.
          The sandbox still shows the gate so an analyst can distinguish a neutral estimate from an
          unsupported claim.
        </p>
        <p>
          Unfilled seats contribute the fitted average. Pick-to-pick changes compare legal branches
          at the same draft seat; partial states are not independently calibrated outcome
          probabilities. Recommendations cannot observe scrim plans, communication, private
          champion pools, planned role swaps, patch-day discoveries, or hidden flex intent. A
          professional analyst should use the decomposition and evidence counts as a challenge list,
          not treat the first row as an instruction.
        </p>
      </MethodSection>

      <MethodSection id="kills" title="Expected kills">
        <p>
          Match checklists use league-mean total kills in the pack year as a descriptive over/under
          prior (half-kill line).
        </p>
      </MethodSection>

      <MethodSection id="evidence" title="Evidence labels">
        <p>
          Evidence is a plain-language view of σ relative to its floor. Settled = at floor. Thin / Very thin =
          headroom above the floor. Games counts sit beside it. Method owns the formula; the ladder
          owns the sentence.
        </p>
      </MethodSection>

      <MethodSection id="blind-counter" title="Blind / Counter (if shown)">
        <p>
          Any Blind/Counter framing on Scryglass is an Oracle&apos;s Elixir matchup-shape proxy. It
          describes observed champion pairs; pick-order seats are outside the estimand.
        </p>
      </MethodSection>

      <MethodSection id="pack-years" title="Pack years">
        <p>
          Default pack years are <strong>2025–2026</strong>. Column-trimmed OE parquet, rating
          snapshots, and pinned calibrations live under{" "}
          <span className="font-mono">/packs/</span>. Cite the pack id when matching a published
          finding. Rebuild notes and essentials are on{" "}
          <Link href="/reproduce" className="row-link">
            Reproduce
          </Link>
          .
        </p>
      </MethodSection>

      <MethodSection id="freshness" title="Freshness and sources">
        <p>
          Scryglass publishes versioned packs, with each rating file preserved as a dated snapshot. OE is
          the canonical source when the same game appears in both feeds. GRID is a pro-only bridge
          for completed games that have already finished while the next OE export is pending.
          Scheduled or scrim-like series stay outside the result set.
        </p>
        <p>
          A new pack is built by the refresh workflow, then the public pointer is updated. The
          pack date is publication time; source metadata identifies the newest match. Use the details
          in Reproduce when the distinction matters.
        </p>
      </MethodSection>

      <MethodSection id="void-grubs" title="Void grubs">
        <p>
          The{" "}
          <Link href="/articles/void-grubs-contest-or-leave" className="row-link">
            void-grubs article
          </Link>{" "}
          leads with the opportunity-cost estimand. At even gold, leaving for the two-wave farm
          reference is worth more than contesting until estimated fight-win reaches the contest bar
          ≈ 58.9%. Gold@10 → map-win is an associational logit conversion. The OE trailing-team
          leave-mix (~24%) answers a different question.
        </p>
      </MethodSection>

      <MethodSection id="faq" title="FAQ">
        <p>
          <strong>Why is Evidence “Settled” at 28 for players?</strong> Because 28 is the player σ
          floor, the model&apos;s minimum spread.
        </p>
        <p>
          <strong>Do league chips change Elo?</strong> They filter the ladder roster while the shared
          Elo stays fixed.
        </p>
        <p>
          <strong>Where is model accuracy?</strong> Matches shows Dual Elo favorite hit rate for the
          selected year. Draft overlap with the actual winner is visible on match boards.
        </p>
      </MethodSection>

      <MethodSection id="changelog" title="Changelog">
        <pre className="changelog-pre">{changelog}</pre>
      </MethodSection>
    </article>
  );
}
