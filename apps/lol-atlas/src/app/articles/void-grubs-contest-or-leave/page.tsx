import type { Metadata } from "next";
import { ArticleContestCharts } from "@/components/ArticleContestCharts";
import { ArticleShell, EvidenceFigure } from "@/components/ArticleShell";
import { getArticle } from "@/lib/articles";
import { blockMathHtml } from "@/lib/formulaHtml";
import { readValidatedGrubsArticlePublication } from "@/lib/grubsArticlePublication.server";
import { PSTAR_TEX } from "@/lib/pstar";
import { readPackManifest } from "@/lib/serverPack";
import "katex/dist/katex.min.css";

function pct(x: number, digits = 1) {
  return `${(100 * x).toFixed(digits)}%`;
}

function pp(x: number) {
  const sign = x > 0 ? "+" : "";
  return `${sign}${x.toFixed(2)}pp`;
}

const META = getArticle("void-grubs-contest-or-leave")!;

export const metadata: Metadata = {
  title: `${META.title} — Scryglass`,
  description: META.dek,
};

// The live pack may change independently of a deployment. Keep the article
// request-time validated rather than freezing a build-time copy.
export const dynamic = "force-dynamic";

function PublicationUnavailable({ packId }: { packId: string }) {
  return (
    <ArticleShell
      meta={META}
      packId={packId}
      kicker={`Article · ${META.topic}`}
      stats={
        <span>
          <strong>Publication</strong> unavailable
        </span>
      }
    >
      <section className="article-prose">
        <h2 className="article-section-title">Publication data unavailable</h2>
        <p>
          The current article artifact did not pass its manifest hash, schema, Patch 26.11+
          mechanics, and formula-parity checks. Numerical conclusions, downloads, and interactive
          figures are withheld until a valid bundle is published.
        </p>
      </section>
    </ArticleShell>
  );
}

export default async function VoidGrubsArticlePage() {
  const manifest = await readPackManifest();
  const publication = await readValidatedGrubsArticlePublication(manifest);
  if (!publication) {
    return <PublicationUnavailable packId={manifest.pack_id} />;
  }

  const { article, atFifty } = publication;

  const formulaHtml = {
    pStar: blockMathHtml(PSTAR_TEX.pStar),
    winProb: blockMathHtml(PSTAR_TEX.winProb),
    params: blockMathHtml(PSTAR_TEX.params),
  };

  return (
    <ArticleShell
      meta={META}
      packId={publication.packId}
      kicker={`Article · ${META.topic}`}
      stats={
        <>
          <span>
            <strong>Contest bar</strong> {article.p_star_pct}%
            <sup>
              <a href="#fn-contest-bar" className="article-fn-ref">
                1
              </a>
            </sup>
          </span>
          <span>
            <strong>At 50/50</strong> leave preferred
          </span>
          <span>
            <strong>Mechanics</strong> Patch {article.mechanics.patch}
          </span>
        </>
      }
    >
      <aside className="article-reader-brief" aria-label="Reader's brief">
        <p className="blog-kicker">Reader&apos;s brief</p>
        <dl>
          <div>
            <dt>Question</dt>
            <dd>When does contesting beat leaving for two waves of farm?</dd>
          </div>
          <div>
            <dt>Result</dt>
            <dd>At even gold, the contest bar is about {article.p_star_pct}%.</dd>
          </div>
          <div>
            <dt>Interpretation</dt>
            <dd>
              A fixed sensitivity comparison, not an identified live-action threshold.
            </dd>
          </div>
        </dl>
      </aside>
      <section className="article-prose">
        <p>
          How often do you need to win the river fight before contesting beats leaving for two waves
          of farm? At even gold that bar sits at{" "}
          <span className="font-mono text-[var(--ink)]">{article.p_star_pct}%</span>
          <sup>
            <a href="#fn-contest-bar" className="article-fn-ref">
              1
            </a>
          </sup>
          . At a coin-flip fight the model still prefers leave
          {" "}
          by{" "}
          <span className="font-mono text-[var(--danger)]">
            {pp(Math.abs(atFifty.edge_contest_minus_leave_pp))}
          </span>
          <sup>
            <a href="#fn-pp" className="article-fn-ref">
              2
            </a>
          </sup>
          .
        </p>
        <p>
          The Patch {article.mechanics.patch} objective term is{" "}
          <span className="font-mono text-[var(--ink)]">
            {article.mechanics.objective_gold_equivalent.toFixed(2)}g
          </span>
          <sup>
            <a href="#fn-objective" className="article-fn-ref">
              3
            </a>
          </sup>
          : 90g cash plus an upper-bound plate-progress equivalent. The progress term is not
          guaranteed immediate gold.
        </p>

        <aside className="article-callout">
          <p className="blog-kicker">Headline</p>
          <p>
            Under this fixed current-mechanics sensitivity, contest and a two-wave leave tie near{" "}
            {article.p_star_pct}% at even gold. This does not identify a universal shotcalling
            threshold.
          </p>
        </aside>

        <div className="mt-4 flex flex-wrap gap-2">
          <a href={publication.href} className="status-pill ghost pill-btn">
            Download the validated article JSON
          </a>
        </div>
      </section>

      <EvidenceFigure
        id="fig-contest-surface"
        caption="Figure 1. Current-mechanics sensitivity by fight-win chance, leave-farm package, and pre-contest gold lead. The dashed line is modeled indifference at even gold."
      >
        <ArticleContestCharts
          curve={article.curve}
          pStar={article.p_star}
          byLeaveFarm={article.by_leave_farm_F}
          byGoldB={article.by_precontest_gold_B_two_wave_leave}
          formulaHtml={formulaHtml}
        />
      </EvidenceFigure>

      <EvidenceFigure
        id="fig-article-curve"
        caption="Figure 2. Article curve table — contest value, leave value, and edge in percentage points of map win rate."
      >
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Fight-win chance</th>
                <th className="num">Contest value</th>
                <th className="num">Leave value</th>
                <th className="num">Edge</th>
                <th>Model preference</th>
              </tr>
            </thead>
            <tbody>
              {article.curve.map((row) => (
                <tr key={row.p_win_fight}>
                  <td className="font-mono">{pct(row.p_win_fight, 0)}</td>
                  <td className="num">{pp(row.ev_contest_pp)}</td>
                  <td className="num">{pp(row.ev_leave_pp)}</td>
                  <td className="num">{pp(row.edge_contest_minus_leave_pp)}</td>
                  <td>
                    {row.model_preference === "CONTEST"
                      ? "Contest higher in model"
                      : "Leave higher in model"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </EvidenceFigure>

      <section className="article-prose border-t border-[var(--line)] pt-5">
        <h2 className="article-section-title">Limits on the estimate</h2>
        <p>
          The gold-at-10 conversion behind the article estimand is associational
          <sup>
            <a href="#fn-assoc" className="article-fn-ref">
              4
            </a>
          </sup>
          — it converts the fixed gold scenarios into modeled map-win association. It does not
          make this comparison causal or estimate a live fight from champions, position, vision,
          cooldowns, or player form.
        </p>
        <ul>
          {article.limitations.map((limitation) => (
            <li key={limitation}>{limitation}</li>
          ))}
        </ul>
      </section>
    </ArticleShell>
  );
}
