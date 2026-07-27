import Link from "next/link";
import { promises as fs } from "fs";
import path from "path";
import type { ReactNode } from "react";
import { readValidatedGrubsArticlePublication } from "@/lib/grubsArticlePublication.server";
import {
  draftProbabilityGateEvidence,
  teamRatingGateEvidence,
} from "@/lib/modelValidation";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

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
  let draftGate: ReturnType<typeof draftProbabilityGateEvidence> = null;
  let teamGate: ReturnType<typeof teamRatingGateEvidence> = null;
  let grubsPublication: Awaited<
    ReturnType<typeof readValidatedGrubsArticlePublication>
  > = null;
  try {
    const manifest = await readPackManifest();
    try {
      const artifact = await readPackJson(
        manifest,
        "models/model_validation_2026-07-27.json",
      );
      draftGate = draftProbabilityGateEvidence(artifact);
      teamGate = teamRatingGateEvidence(artifact);
    } catch {
      draftGate = null;
      teamGate = null;
    }
    try {
      grubsPublication = await readValidatedGrubsArticlePublication(manifest);
    } catch {
      grubsPublication = null;
    }
  } catch {
    draftGate = null;
    teamGate = null;
    grubsPublication = null;
  }

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

      <MethodSection id="dynamic-series-ladder" title="Current public ladder" defaultOpen>
        <p>
          The published team snapshot uses a time-varying Bradley–Terry state-space filter over
          verified completed series. One immutable organization identity persists across events;
          event labels never create a second team. A forecast is emitted at the verified series
          start and the result enters the state only at verified completion.
        </p>
        <p>
          Historical LTA source labels are retained for audit: LTA North maps to LCS, LTA South maps
          to CBLOL, and an unqualified LTA row is an Americas cross-region event. International
          events are classified from their source competition, so “LCK Road to MSI” remains LCK.
          Only series with verified Bo1, Bo3, or Bo5 format, contiguous maps, a compatible terminal
          score, and explicit completion provenance enter the rating. Ambiguous, tied, gapped, or
          incomplete groups remain quarantined.
        </p>
        <p>
          When the artifact declares <span className="font-mono">z = 1.64485</span>, the adjusted
          value is the filtered mean minus that multiple of its diagonal Gaussian approximation:
          a normal-approximation 5th percentile used for conservative ordering. It is not presented
          as an empirically calibrated 95% coverage bound and is withheld when the row, metadata,
          and arithmetic do not agree. Uncertainty grows during inactivity.
        </p>
        <p>
          Current affiliation comes only from the reviewed Riot tournament registry and matching
          row-level participation evidence. A last observed team or league cannot create current
          membership. The public current-membership view is Tier 1 only until an authoritative
          registry establishes lower-tier participation under the same contract.
        </p>
        <p>
          Model changes require chronological holdouts, immutable prediction rows, log loss, Brier
          score, calibration, dependence-aware intervals, and rank stability. A challenger is not
          promoted merely because its point estimate is better; the current immutable validation
          artifact records the selection, untouched test, uncertainty interval, and promotion
          decision. The frozen 1,771-series test also reports calibration separately for Bo1, Bo3,
          and Bo5. Ratings are comparable only inside one connected historical comparison component.
          The design follows the dynamic Bradley–Terry and state-space literature: <a className="row-link" href="https://doi.org/10.1111/j.1467-9876.2012.01046.x">dynamic BT</a>,
          <a className="row-link" href="https://arxiv.org/abs/2308.02414"> state-space skill models</a>, and
          <a className="row-link" href="https://www.remi-coulom.fr/WHR/WHR.pdf"> whole-history rating</a>.
        </p>
        {teamGate ? (
          <p>
            On the immutable {teamGate.finalTestSeries.toLocaleString()}-series final test,
            the dynamic model scored {teamGate.logLoss.toFixed(5)} log loss versus{" "}
            {teamGate.rollingEloLogLoss.toFixed(5)} for the selected rolling-Elo
            benchmark. The paired difference was {teamGate.logLossDifference.toFixed(5)},
            with a 95% circular moving-block bootstrap interval from{" "}
            {teamGate.confidenceInterval[0].toFixed(5)} to{" "}
            {teamGate.confidenceInterval[1].toFixed(5)}. Its Brier score was{" "}
            {teamGate.brier.toFixed(5)} and ten-bin ECE was{" "}
            {teamGate.ece.toFixed(5)}. The {teamGate.bo5Series}-series Bo5 slice had ECE{" "}
            {teamGate.bo5Ece.toFixed(3)}, so this release is not described as universally
            calibrated or state of the art.
          </p>
        ) : (
          <p>
            Team-rating test metrics are withheld because the current immutable pack does
            not provide a gate artifact whose model hashes, observation hash, benchmark
            comparison, interval, and format diagnostics reconcile.
          </p>
        )}
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
          The legacy player track applies the same team outcome to the five observed players. It can
          travel across organizations and provide a shared lineup signal, but team wins generally do
          not identify individual teammate effects. Missing identifiability metadata is therefore
          “unverified,” never individual evidence. Exact shared-exposure cohorts remain tied, and
          public individual rank ordering is withheld even when a player has a unique exposure
          history.
        </p>
        <p>
          A separately gated research model estimates role-relative 15-minute resource performance
          from gold, experience, and creep-score differentials without using map wins. It appears
          only when the current immutable pack contains a matching snapshot, metadata, and passing
          chronological validation artifact. Even then, it is not general skill, win contribution,
          or complete-game performance.
        </p>
      </MethodSection>

      <MethodSection id="draft-score" title="Draft Score">
        {draftGate ? (
          <p>
            The published draft win probability is withheld by the current immutable validation
            artifact. On its untouched {draftGate.finalTestMaps.toLocaleString()}-map chronological
            test window, the best role/champion composition candidate scored{" "}
            {draftGate.compositionLogLoss.toFixed(5)} log loss and{" "}
            {draftGate.compositionBrier.toFixed(5)} Brier, compared with{" "}
            {draftGate.overallBaseRateLogLoss.toFixed(5)} and{" "}
            {draftGate.overallBaseRateBrier.toFixed(5)} for the overall blue-side base rate.
            Those scores belong to the pre-final fitted pipeline evaluated on that untouched
            window. The experimental Sandbox runtime was refit on the full population, including
            those labels, and is exposed only as a composition utility; it is not the exact
            coefficient artifact scored above. Calling either output a calibrated win chance would
            overstate the evidence.
          </p>
        ) : (
          <p>
            Draft win probability is withheld. The current pack does not provide a valid immutable
            gate artifact from which this page can quote test metrics, so no numerical fallback is
            shown.
          </p>
        )}
        <p>
          For a complete, uniquely role-assigned draft, the frozen probability
          specification is{" "}
          <span className="font-mono">
            logit(p_blue) = a + b × (β_side + e_composition)
          </span>
          . The composition edge contains role-aware direct terms, within-team
          pair terms, and all cross-team interactions. Its sign reverses when
          the two role-labelled compositions swap. The blue-side probability
          itself need not sum to one with that swapped counterfactual because
          the fitted side baseline remains attached to blue. The interval uses
          the model-intercept variance, a diagonal approximation for active
          composition terms, and the full intercept/slope covariance from the
          chronological calibration fit; omitted feature covariance remains a
          stated limitation.
        </p>
        <p>
          The Draft Sandbox therefore uses a 0–100 experimental composition policy value at every
          draft state, including a complete board. It combines role-aware champion terms, observed
          ally pair terms, enemy interactions, and a bounded two-ply beam-minimax response search.
          It is useful for comparing branches inside this model, but it is not a win probability,
          an exhaustive best response, or evidence of optimal drafting. It also does not include
          private champion pools, scrim plans, or unannounced flex intent.
          Public patch labels are explicit contracts: public 25.xx maps to
          source key 15.xx and public 26.xx maps to source key 16.xx. Ambiguous
          one-digit minor strings are rejected.
        </p>
      </MethodSection>

      <MethodSection id="kills" title="Chronological kills benchmark">
        <p>
          A match may be compared with the same-league mean from strictly earlier maps in the pack
          year. This is a simple chronological benchmark, not a fitted kills forecast. Missing
          history or map kills suppresses the comparison.
        </p>
      </MethodSection>

      <MethodSection id="evidence" title="Evidence labels">
        <p>
          Evidence semantics are model-specific. Dynamic team rows use the model-declared local
          spread and bound contract; Player Dual Elo uses its own sigma floor only after outcome
          identifiability is known. A shared cohort or missing metadata cannot be labelled as
          individual evidence merely because the numeric spread is small.
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
          the canonical inclusion source when the same game appears in both feeds. Verified GRID
          results may bridge a completed-game gap while the next OE export is pending; GRID may
          also provide event detail for a map whose canonical result is already OE-backed. Those
          roles are published separately as <span className="font-mono">canonical_map_source</span>{" "}
          and <span className="font-mono">map_detail_source</span>. Scheduled or scrim-like series
          stay outside the result set.
        </p>
        <p>
          A new pack is built by the refresh workflow, then the public pointer is updated. The
          pack date is publication time; source metadata identifies the newest match. Use the details
          in Reproduce when the distinction matters.
        </p>
      </MethodSection>

      <MethodSection id="void-grubs" title="Void grubs">
        {grubsPublication ? (
          <p>
            The{" "}
            <Link href="/articles/void-grubs-contest-or-leave" className="row-link">
              void-grubs article
            </Link>{" "}
            leads with a Patch {grubsPublication.article.mechanics.patch} opportunity-cost
            sensitivity. At even gold, leaving for the two-wave farm reference is worth more than
            contesting until estimated fight-win reaches its{" "}
            {grubsPublication.article.p_star_pct.toFixed(2)}% contest bar. Gold@10 → map-win is an
            associational logit conversion; the result is not an identified action policy.
          </p>
        ) : (
          <p>
            The void-grubs numerical methodology is unavailable because the active immutable
            article artifact did not pass its manifest hash, current-mechanics, schema, and
            formula-parity checks.
          </p>
        )}
      </MethodSection>

      <MethodSection id="faq" title="FAQ">
        <p>
          <strong>Why can two teammates have the same score?</strong> When they have the same signed
          map exposure, team outcomes contain no information that separates them. The interface
          keeps that tie and states the identifiability limit.
        </p>
        <p>
          <strong>Do league chips change Elo?</strong> They filter the ladder roster while the shared
          Elo stays fixed.
        </p>
        <p>
          <strong>Where is probability quality?</strong> Favorite hit rate is only a threshold
          diagnostic. Release evidence uses chronological log loss, Brier score, calibration, and
          dependence-aware comparisons tied to an immutable model and pack.
        </p>
      </MethodSection>

      <MethodSection id="changelog" title="Changelog">
        <pre className="changelog-pre">{changelog}</pre>
      </MethodSection>
    </article>
  );
}
