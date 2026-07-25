import { promises as fs } from "fs";
import path from "path";
import { ArticleContestCharts } from "@/components/ArticleContestCharts";
import { PdfEmbed } from "@/components/PdfEmbed";
import { blockMathHtml } from "@/lib/formulaHtml";
import type { PackManifest } from "@/lib/pack";
import { packUrl } from "@/lib/pack";
import { PSTAR_TEX } from "@/lib/pstar";
import "katex/dist/katex.min.css";

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

export default async function GrubsPage() {
  const man = JSON.parse(
    await fs.readFile(path.join(process.cwd(), "public", "packs", "manifest.json"), "utf8"),
  ) as PackManifest;
  const base = path.join(process.cwd(), "public", "packs", man.pack_id, "studies", "grubs");
  const article = JSON.parse(
    await fs.readFile(path.join(base, "grubs_article_contest_ev.json"), "utf8"),
  ) as ArticleEv;
  const numbers = JSON.parse(
    await fs.readFile(path.join(base, "grubs_decision_numbers.json"), "utf8"),
  ) as DecisionNumbers;

  const pdfHref = packUrl(man, "studies/grubs/void_grubs_scrap_value_and_contest_rationality.pdf");
  const articleHref = packUrl(man, "studies/grubs/grubs_article_contest_ev.json");
  const at50 = article.curve.find((c) => c.p_win_fight === 0.5);
  const formulaHtml = {
    pStar: blockMathHtml(PSTAR_TEX.pStar),
    winProb: blockMathHtml(PSTAR_TEX.winProb),
    params: blockMathHtml(PSTAR_TEX.params),
  };

  return (
    <div className="space-y-[var(--space-5)]">
      <header className="page-header">
        <p className="blog-kicker">Study companion · Void grubs</p>
        <h1 className="font-display mt-3 text-[2.25rem] leading-tight text-[var(--ink)] sm:text-[2.75rem]">
          The <span className="olive">contest bar</span>
        </h1>
        <p className="lede">
          How often do you need to win the river fight before contesting beats leaving for two
          waves of farm? At even gold that bar sits at{" "}
          <span className="font-mono text-[var(--ink)]">{article.p_star_pct}%</span>. At a coin-flip
          fight the model still prefers leave
          {at50 ? (
            <>
              {" "}
              by{" "}
              <span className="font-mono text-[var(--danger)]">
                {pp(Math.abs(at50.edge_contest_minus_leave_pp))}
              </span>
            </>
          ) : null}
          .
        </p>
        <div className="mt-4">
          <PdfEmbed src={pdfHref} title="Void grubs scrap value PDF" />
        </div>
        <div className="mt-3">
          <a href={articleHref} className="status-pill ghost" style={{ padding: "0.55rem 0.9rem" }}>
            article_contest_ev.json
          </a>
        </div>
        <div className="micro-log mt-6">
          <span>
            <strong>Contest bar</strong> {article.p_star_pct}%
          </span>
          <span>
            <strong>Pack</strong> {man.pack_id}
          </span>
        </div>
      </header>

      <p className="max-w-[68ch] border-t border-[var(--line)] pt-4 text-sm text-[var(--ink-muted)]">
        <span className="status-pill" style={{ marginBottom: "0.5rem" }}>
          Headline
        </span>
        <br />
        Quote this: contest bar ≈ {article.p_star_pct}% at even gold (two-wave leave). Below that
        fight-win chance, leave; above it, contest can be worth it. (pp = percentage points of map
        win rate.)
      </p>

      <ArticleContestCharts
        curve={article.curve}
        pStar={article.p_star}
        byLeaveFarm={article.by_leave_farm_F}
        byGoldB={article.by_precontest_gold_B_two_wave_leave}
        formulaHtml={formulaHtml}
      />

      <section className="space-y-3">
        <h2 className="font-display text-xl">Article curve (table)</h2>
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
      </section>

      <section className="border-t border-[var(--line)] pt-5 space-y-3 max-w-[68ch]">
        <h2 className="font-display text-lg">Why you also see ~24%</h2>
        <p className="text-sm text-[var(--ink-muted)]">
          That number is <strong className="text-[var(--ink)]">not</strong> the article contest
          bar. It comes from a different question on Oracle&apos;s Elixir trailing-team maps: among
          teams that actually contested while behind, what fight-win rate would make contesting
          break even versus the historical leave mix those teams took. That other break-even is
          about{" "}
          <span className="font-mono text-[var(--ink)]">
            {pct(numbers.breakeven_p_win_fight.vs_leave_mix)}
          </span>
          . Use it only when you care about that OE leave-mix sample — never as a substitute for
          the two-wave contest bar ({article.p_star_pct}%).
        </p>
        <details>
          <summary className="cursor-pointer text-sm font-medium text-[var(--accent)]">
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

      <section className="border-t border-[var(--line)] pt-5 space-y-2 max-w-[68ch]">
        <h2 className="font-display text-lg">Reproduce</h2>
        <ol className="list-decimal pl-5 text-sm text-[var(--ink-muted)] space-y-1">
          <li>Pin pack {man.pack_id}</li>
          <li>Start at studies/grubs/grubs_article_contest_ev.json</li>
          <li>PDF above for scrap-value narrative</li>
        </ol>
      </section>
    </div>
  );
}
