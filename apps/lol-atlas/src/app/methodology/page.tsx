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

      <MethodSection id="dual-elo" title="Dual Elo (teams)" defaultOpen>
        <p>
          Each team carries a regional component and an international (meta) component. They sum to
          a total rating μ with a spread σ. Outcomes in the Oracle&apos;s Elixir pack update both.
          League chips on the Ratings page filter who appears while μ stays shared. The pack
          uses Oracle&apos;s Elixir as its reconciled baseline and can include a recent completed
          GRID game while the next OE export is pending.
        </p>
        <p>
          σ shrinks toward a floor as informative games arrive. Team floor is 25. When σ sits on that
          floor, the Evidence label reads <em>Settled</em> — the rating is as tight as this model allows. Headroom
          above the floor is what the adjusted rating penalizes:
        </p>
        <p className="font-mono text-sm">
          adjusted rating = μ − max(0, σ − σ_min)
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

      <MethodSection id="draft-score" title="Draft Score">
        <p>
          Draft Score maps five-on-five picks to a blue win probability with league-calibrated
          temperature. The Elo-controlled bump (wr_bump_pp) is the residual ridge × draft edge ×
          confidence. Draft favorites can disagree with Dual Elo favorites; Matches reports Elo hit
          rate, while match boards also show Draft WR.
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
