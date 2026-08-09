import Link from "next/link";
import { promises as fs } from "fs";
import path from "path";
import type { ReactNode } from "react";
import { OperationalHeader } from "@/components/OperationalHeader";

async function loadChangelog(): Promise<string> {
  try {
    return await fs.readFile(path.join(process.cwd(), "CHANGELOG.md"), "utf8");
  } catch {
    return "_No changelog entries yet._";
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
  const changelog = await loadChangelog();

  return (
    <article className="page-prose methodology-page">
      <OperationalHeader
        title="Methodology"
        description="Definitions, model assumptions, validation, and data provenance for the public estimates."
        meta={
          <>
            <a className="row-link" href="/method/formulas.tex">
              Formulas (.tex)
            </a>
            <Link className="row-link" href="/articles/void-grubs-contest-or-leave">
              Void-grubs article
            </Link>
          </>
        }
      />

      <nav className="method-toc" aria-label="Methodology sections">
        <strong>On this page</strong>
        <a href="#hierarchical-ladder">Team ratings</a>
        <a href="#dual-elo">Sequential benchmark</a>
        <a href="#player-elo">Player ratings</a>
        <a href="#draft-score">Draft model</a>
        <a href="#evidence">Evidence and uncertainty</a>
        <a href="#tier-list">Tier list calculation</a>
        <a href="#pack-years">Pack years</a>
        <a href="#freshness">Data sources</a>
        <a href="#void-grubs">Void grubs</a>
        <a href="#faq">FAQ</a>
        <a href="#changelog">Changelog</a>
      </nav>

      <div className="method-content">
        <MethodSection id="hierarchical-ladder" title="Team ratings: what the number means" defaultOpen>
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

      <MethodSection id="dual-elo" title="Sequential benchmark">
        <p>
          The sequential Dual Elo track remains available as a time-safe pre-match feature benchmark.
          Each team carries a regional component and an international (meta) component. They sum to
          a total rating μ with a spread σ. Outcomes in the Oracle&apos;s Elixir pack update both.
          League chips on the Ratings page filter who appears while μ stays shared.
        </p>
        <p>
          σ shrinks toward a floor as informative games arrive. Team floor is 25; σ is a diagnostic,
          not the evidence label. The Evidence label comes from the validated evidence contract:
          95% interval width, relative precision versus the tightest rating in the same scope,
          weekly stability, freshness, and support coverage. <em>Settled</em> requires strictly
          greater-than-95% relative precision, known bounded stability, a fresh and active row,
          and full support coverage. Anything missing or failing closes to Stale, Inactive,
          Disconnected, Wide interval, Fallback, Out of distribution, or Unsupported.
        </p>
        <p className="font-mono text-sm">
          benchmark adjusted rating = μ − max(0, σ − σ_min)
        </p>
        <p>
          That soft penalty stops a thin regional spike from outranking a settled major org on the
          default ladder sort. Full LaTeX is in the formulas download.
        </p>
      </MethodSection>

      <MethodSection id="player-elo" title="Player ratings">
        <p>
          Players are rated on their own Dual Elo track (player floor σ_min = 28; the floor is a
          diagnostic, not the evidence label). A team&apos;s player-aggregated strength is a
          role-weighted blend of the five on the rift. Prefer player ladders when rosters move. The
          Evidence label follows the same validated contract as teams: relative precision,
          per-game stability, freshness, support coverage, active eligibility, and fallback/league
          flags — a low σ alone never reads as Settled.
        </p>
      </MethodSection>

      <MethodSection id="draft-score" title="Draft model">
        <p>
          Draft Score is not available in the public MVP. The sandbox route is retained as a
          placeholder while the source-bound fit, held-out evaluation, and serving transform are
          independently reviewed.
        </p>
        <p>
          A future release may expose a partial-draft comparison, but it will require a separately
          authorized terminal and prefix evaluation. Until that work is accepted, no public score,
          probability, confidence, recommendation, or accuracy claim is emitted.
        </p>
      </MethodSection>

      <MethodSection id="evidence" title="Evidence and uncertainty">
        <p>
          Evidence is a fail-closed state derived from the validated contract, not from σ alone.
          Each rating row carries a 95% interval width, relative precision, stability, freshness,
          support coverage, and fallback/active/disconnected/OOD flags; the label is computed from
          those fields with exact thresholds. <strong>Settled</strong> = strictly greater-than-95%
          relative precision, known bounded stability, fresh and active inputs, and full support
          coverage. <strong>Observed</strong> = reasonably tight and supported but not Settled.
          <strong>Thin</strong> = still moving. Stale, Inactive, Disconnected, Wide interval,
          Fallback, Out of distribution, and Unsupported rows fail closed and never read as
          settled. σ and game/series counts remain visible as separate diagnostics beside the
          label.
        </p>
      </MethodSection>

      <MethodSection id="tier-list" title="Tier List Calculation">
        <p>
          Each role and competition gets its own board. The model reads completed professional maps
          from 01-01-2025 through the source watermark. One map supplies one result. All five role
          matchups enter that result together, alongside blue side and each team&apos;s pre-match rating.
          Recent maps carry more weight, with a 120-day half-life.
        </p>
        <p>
          <strong>Strength</strong> asks how a champion performs against 5 common, legal opponents in
          that role. The reference pool starts with the 6 most-picked champions, removes the champion
          being rated, then weights the remaining 5 by pick frequency. Tier Value is the expected
          matchup result relative to 50%. A through D follow this strength order.
        </p>
        <p>
          <strong>Blind</strong> looks at the ugly part of the matchup spread. For every uncertainty
          draw, Scryglass takes the weighted average of the weakest 20% of those 5 matchups. The
          published score uses a conservative lower estimate. A high Blind score means the champion&apos;s
          bad matchups still look playable.
        </p>
        <p>
          <strong>Counter</strong> measures matchup advantage left after general champion strength is
          accounted for. The score is the expected weighted number of reference opponents that the
          champion counters. A stricter count requires at least an 80% posterior chance that the
          matchup advantage exceeds 0.05 logit.
        </p>
        <p>
          Z and S labels need complete evidence against all 5 reference opponents. Every pair must
          include multiple maps and series, both observed outcomes, and sufficiently narrow
          uncertainty. The label must also survive repeated uncertainty draws with at least 65%
          membership probability. If those checks fail, the champion remains in A through D and the
          Blind or Counter claim stays unavailable.
        </p>
        <p>
          Champion atomization runs before every refresh. It records the champion state, source patch,
          and changed mechanics that need review. Oracle&apos;s Elixir patch labels remain in their own
          namespace until an exact patch mapping is verified. This keeps current-patch claims closed
          when the two sources cannot be joined safely.
        </p>
        <p>
          Rank arrows compare the current board with a previous weekly snapshot rebuilt under the same
          method. Blind and Counter describe observed matchup shape. They do not identify pick order or
          a causal counter-pick effect.
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

      <MethodSection id="freshness" title="Data freshness and sources">
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
          <strong>Where is model accuracy?</strong> The public MVP shows completed match records,
          not a predictive accuracy claim. Accuracy summaries remain development and review
          artifacts until an independently accepted serving contract is available.
        </p>
      </MethodSection>

      <MethodSection id="changelog" title="Changelog">
        <pre className="changelog-pre">{changelog}</pre>
      </MethodSection>
      </div>
    </article>
  );
}
