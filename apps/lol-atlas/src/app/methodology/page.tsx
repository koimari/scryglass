import Link from "next/link";

export default function MethodologyPage() {
  return (
    <article className="page-prose">
      <header className="page-header">
        <p className="blog-kicker">Method · Estimands</p>
        <h1 className="font-display mt-2 text-[2.25rem] leading-tight text-[var(--ink)]">
          Methodology
        </h1>
        <p className="lede">
          Strength estimates and a filtered match pack for reproduction. Not official Riot ranking.
        </p>
      </header>

      <section className="method-block">
        <h2 className="method-h">Dual Elo (teams)</h2>
        <div className="method-body">
          <p>
            Teams carry regional and meta skill components that combine into a total μ with
            uncertainty σ. Ratings update from professional map outcomes in the Oracle&apos;s Elixir
            warehouse. Published win probabilities use a logistic map from μ gaps (Elo→WR
            calibration), hotter than a classic 400-scale for player-aggregated gaps.
          </p>
        </div>
      </section>

      <section className="method-block">
        <h2 className="method-h">Player Dual Elo</h2>
        <div className="method-body">
          <p>
            Players are rated individually. A team&apos;s player-aggregated strength is the
            role-weighted blend of the five on the rift. Prefer player ladders when rosters move.
          </p>
        </div>
      </section>

      <section className="method-block">
        <h2 className="method-h">Reproduction pack</h2>
        <div className="method-body">
          <p>
            Default pack years are <strong className="text-[var(--ink)]">2025–2026</strong>. The pack
            contains column-trimmed OE parquet, rating snapshots, and pinned calibration files.
            Source attribution and SHA-256 hashes are recorded in{" "}
            <span className="font-mono text-[0.85em] text-[var(--ink)]">manifest.json</span>.
          </p>
        </div>
      </section>

      <section className="method-block">
        <h2 className="method-h">Void grubs</h2>
        <div className="method-body space-y-3">
          <p>
            The{" "}
            <Link href="/articles/void-grubs-contest-or-leave" className="underline decoration-[var(--line)] underline-offset-2 hover:decoration-[var(--ink)]">
              void-grubs article
            </Link>{" "}
            leads with the article&apos;s opportunity-cost estimand. At even gold, leaving
            to collect the two-wave farm reference is worth more than contesting until your estimated
            chance to win the fight reaches the{" "}
            <span className="font-mono text-[var(--ink)]">contest bar ≈ 58.9%</span>. A coin-flip
            fight therefore still favors leaving, by about 2.08 percentage points in the model.
          </p>
          <p>
            The conversion from gold at 10 minutes to map win probability is associational. The
            threshold changes with the gold already held and with the farm that can actually be
            collected, so 58.9% is the stated two-wave reference, not a universal contest rule.
          </p>
          <p>
            The Oracle&apos;s Elixir trailing-team leave-mix analysis is a separate sister study. Its
            roughly 24% breakeven compares contest outcomes with the observed leave-mix in that
            sample. It does not reproduce the article&apos;s opportunity-cost choice and should not
            appear as a second headline threshold.
          </p>
        </div>
      </section>
    </article>
  );
}
