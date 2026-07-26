import type { Metadata } from "next";
import { ArticleContestCharts } from "@/components/ArticleContestCharts";
import { ArticleShell, EvidenceFigure } from "@/components/ArticleShell";
import { PdfEmbed } from "@/components/PdfEmbed";
import { getArticle } from "@/lib/articles";
import { blockMathHtml } from "@/lib/formulaHtml";
import { packUrl } from "@/lib/pack";
import { PSTAR_TEX } from "@/lib/pstar";
import "katex/dist/katex.min.css";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

type ArticleEv = {
  p_star: number;
  p_star_pct: number;
  note: string;
  estimand_note: string;
  curve: Array<{
    p_win_fight: number;
    ev_contest_pp: number;
    ev_leave_pp: number;
    edge_contest_minus_leave_pp: number;
    verdict: string;
  }>;
  by_leave_farm_F: Array<{
    label: string;
    leave_farm_gold: number;
    p_star_at_parity: number;
    p_star_at_parity_pct: number;
    p_star_at_B_plus_1183: number;
    p_star_at_B_plus_1183_pct: number;
  }>;
  by_precontest_gold_B_two_wave_leave: Array<{
    B_gold: number;
    leave_farm_gold: number;
    objective_gold: number;
    p_star: number;
    p_star_pct: number;
  }>;
  oe_sister: {
    breakeven_vs_leave_mix: number;
    breakeven_vs_split: number;
    win_minus_leave_mix_pp: number;
    warning: string;
  };
};

type DecisionNumbers = {
  sample: {
    n_trailing_maps: number;
    n_era_3camp: number;
    date_min: string;
    date_max: string;
    filter: string;
  };
  outcomes: Record<
    string,
    { n: number; wr: number; mean_gold10: number; mean_gold_path_10_to_15: number }
  >;
  deltas_pp: {
    win_minus_lose: number;
    win_minus_split: number;
    lose_minus_split: number;
    win_minus_leave_mix: number;
  };
  breakeven_p_win_fight: { vs_leave_mix: number; vs_split: number };
  ev_curve: Array<{
    p_win_fight: number;
    expected_wr_if_contest: number;
    edge_vs_leave_mix_pp: number;
    verdict_vs_leave: string;
  }>;
};

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

export default async function VoidGrubsArticlePage() {
  const man = await readPackManifest();
  const article = await readPackJson<ArticleEv>(man, "studies/grubs/grubs_article_contest_ev.json");
  const numbers = await readPackJson<DecisionNumbers>(man, "studies/grubs/grubs_decision_numbers.json");

  const pdfHref = packUrl(man, "studies/grubs/void_grubs_scrap_value_and_contest_rationality.pdf");
  const articleHref = packUrl(man, "studies/grubs/grubs_article_contest_ev.json");
  const at50 = article.curve.find((c) => c.p_win_fight === 0.5);
  const formulaHtml = {
    pStar: blockMathHtml(PSTAR_TEX.pStar),
    winProb: blockMathHtml(PSTAR_TEX.winProb),
    params: blockMathHtml(PSTAR_TEX.params),
  };

  return (
    <ArticleShell
      meta={META}
      packId={man.pack_id}
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
            <dt>Decision rule</dt>
            <dd>Research estimate of the opportunity cost between contesting and taking two waves of farm.</dd>
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
          {at50 ? (
            <>
              {" "}
              by{" "}
              <span className="font-mono text-[var(--danger)]">
                {pp(Math.abs(at50.edge_contest_minus_leave_pp))}
              </span>
              <sup>
                <a href="#fn-pp" className="article-fn-ref">
                  2
                </a>
              </sup>
            </>
          ) : null}
          .
        </p>

        <aside className="article-callout">
          <p className="blog-kicker">Headline</p>
          <p>
            Quote this: contest bar ≈ {article.p_star_pct}% at even gold (two-wave leave). Below that
            fight-win chance, leave; above it, contest can be worth it.
          </p>
        </aside>

        <div className="mt-4 flex flex-wrap gap-2">
          <PdfEmbed src={pdfHref} title="Void grubs scrap value PDF" />
          <a href={articleHref} className="status-pill ghost pill-btn">
            Download the numbers (JSON)
          </a>
        </div>
      </section>

      <EvidenceFigure
        id="fig-contest-surface"
        caption="Figure 1. Leave vs contest by fight-win chance, leave-farm package, and pre-contest gold lead. The dashed line is the article contest bar at even gold."
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
                    {row.verdict === "CONTEST" ? "Contest preferred" : "Leave preferred"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </EvidenceFigure>

      <section className="article-prose border-t border-[var(--line)] pt-5">
        <h2 className="article-section-title">Why you also see ~24%</h2>
        <p>
          That number answers a different Oracle&apos;s Elixir question from the article contest bar
          <sup>
            <a href="#fn-24" className="article-fn-ref">
              3
            </a>
          </sup>
          . It comes from a different question on Oracle&apos;s Elixir trailing-team maps: among teams
          that actually contested while behind, what fight-win rate would make contesting break even
          versus the historical leave mix those teams took. That other break-even is about{" "}
          <span className="font-mono text-[var(--ink)]">
            {pct(numbers.breakeven_p_win_fight.vs_leave_mix)}
          </span>
          . Keep it alongside the two-wave contest bar ({article.p_star_pct}%) when you are studying
          the OE leave-mix sample.
        </p>
        <p>
          The gold-at-10 conversion behind the article estimand is associational
          <sup>
            <a href="#fn-assoc" className="article-fn-ref">
              4
            </a>
          </sup>
          — the conversion estimates map-win association with gold@10 under the article estimand.
        </p>
        <details className="mt-3">
          <summary className="cursor-pointer text-sm font-medium text-[var(--accent-ink)]">
            Show OE leave-mix curve (optional)
          </summary>
          <p className="mt-3 text-sm text-[var(--ink-muted)]">
            Trailing maps n={numbers.sample.n_trailing_maps.toLocaleString()} · vs split{" "}
            {pct(numbers.breakeven_p_win_fight.vs_split, 0)} · win−leave_mix{" "}
            {pp(numbers.deltas_pp.win_minus_leave_mix)}
          </p>
          <div className="mt-4 table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Fight-win chance</th>
                  <th className="num">Expected WR if contest</th>
                  <th className="num">vs leave-mix</th>
                  <th>Preference</th>
                </tr>
              </thead>
              <tbody>
                {numbers.ev_curve.map((row) => (
                  <tr key={row.p_win_fight}>
                    <td className="font-mono">{pct(row.p_win_fight, 0)}</td>
                    <td className="num">{pct(row.expected_wr_if_contest)}</td>
                    <td className="num">{pp(row.edge_vs_leave_mix_pp)}</td>
                    <td>{row.verdict_vs_leave === "CONTEST" ? "Contest" : "Leave"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>
    </ArticleShell>
  );
}
