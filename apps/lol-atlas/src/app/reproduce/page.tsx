import { formatMb, packUpdatedLabel, packUrl, type PackFile, type PackManifest } from "@/lib/pack";
import { readPackManifest } from "@/lib/serverPack";

/** Hard allowlist: if it is not cited on-site, it does not appear here. */
const ESSENTIALS: { group: string; paths: string[] }[] = [
  {
    group: "Data",
    paths: [
      "maps/year=2025/part.parquet",
      "maps/year=2026/part.parquet",
      "team_games/year=2025/part.parquet",
      "team_games/year=2026/part.parquet",
      "player_games/year=2025/part.parquet",
      "player_games/year=2026/part.parquet",
    ],
  },
  {
    group: "Ratings",
    paths: [
      "features/ratings_snapshot.json",
      "features/ratings_snapshot.parquet",
      "features/ratings_history.parquet",
      "features/ratings_meta.json",
      "features/player_ratings_snapshot.json",
      "features/player_ratings_snapshot.parquet",
      "features/player_ratings_history.parquet",
      "features/player_ratings_meta.json",
      "features/team_records.json",
      "features/player_records.json",
      "features/major_teams.json",
      "models/elo_wr_calibration.json",
      "models/draft_wr_calibration.json",
    ],
  },
  {
    group: "Article inputs",
    paths: [
      "studies/grubs/grubs_article_contest_ev.json",
      "studies/grubs/grubs_decision_numbers.json",
      "studies/grubs/void_grubs_scrap_value_and_contest_rationality.pdf",
    ],
  },
];

function findFile(man: PackManifest, rel: string): PackFile | undefined {
  return man.files.find((f) => f.path === rel || f.relative === rel);
}

export default async function ReproducePage() {
  const man = await readPackManifest();

  const listed: { group: string; file: PackFile; path: string }[] = [];
  for (const g of ESSENTIALS) {
    for (const p of g.paths) {
      const f = findFile(man, p);
      if (f) listed.push({ group: g.group, file: f, path: p });
    }
  }
  const listedBytes = listed.reduce((s, x) => s + x.file.bytes, 0);

  return (
    <div className="space-y-8">
      <header className="page-header">
        <p className="blog-kicker">Pack · Reproduce</p>
        <h1 className="font-display mt-2 text-3xl">Reproduce</h1>
        <p className="lede">
          Cite <span className="font-mono text-sm text-[var(--ink)]">{man.pack_id}</span>. This list
          contains the finished files cited on Scryglass. Rebuild from the GitHub repo when you need
          the warehouse pipeline.
        </p>
        <p className="method-note">
          Source order: Oracle&apos;s Elixir is the reconciled baseline; completed GRID games may
          bridge the gap until OE publishes them. The refresh metadata records which rows came from
          each source.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Last updated</strong> {packUpdatedLabel(man)}
          </span>
          <span>
            <strong>Listed</strong> {formatMb(listedBytes)}
          </span>
          <span>
            <strong>Schema</strong> {man.schema_version}
          </span>
          <span>
            <strong>Years</strong> {man.filters.years.join("–")}
          </span>
        </div>
      </header>

      <section className="space-y-3 border-t border-[var(--line)] pt-5">
        <h2 className="font-display text-lg">Checklist</h2>
        <ol className="list-decimal space-y-2 pl-5 text-sm text-[var(--ink-muted)]">
          <li>Pin this pack id.</li>
          <li>Use the same year / league / patch filters as the post.</li>
          <li>
            Elo→WR and Draft Score: use the pinned files under Ratings below.
          </li>
          <li>
            Void grubs: article JSON + PDF under Article inputs — leave-mix (~24%) is a sister
            estimand.
          </li>
        </ol>
      </section>

      {ESSENTIALS.map((g) => (
        <section key={g.group} className="space-y-3">
          <h2 className="font-display text-lg">{g.group}</h2>
          <ul className="text-sm">
            {g.paths.map((p) => {
              const f = findFile(man, p);
              if (!f) {
                return (
                  <li key={p} className="py-2.5 border-b border-[var(--line)] muted text-xs">
                    {p} <em>(missing from pack)</em>
                  </li>
                );
              }
              return (
                <li
                  key={p}
                  className="flex flex-wrap items-baseline justify-between gap-2 py-2.5 border-b border-[var(--line)]"
                >
                  <a className="font-mono break-all text-xs underline sm:text-sm" href={packUrl(man, f.path)}>
                    {p}
                  </a>
                  <span className="font-mono text-xs text-[var(--ink-muted)]">
                    {(f.bytes / 1024).toFixed(0)} KB
                    {f.rows != null ? ` · ${f.rows} rows` : ""}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
      ))}

      <section className="space-y-2 text-sm text-[var(--ink-muted)] border-t border-[var(--line)] pt-5">
        <h2 className="font-display text-lg text-[var(--ink)]">Power users</h2>
        <p>
          Full manifest:{" "}
          <a className="row-link" href="/packs/manifest.json">
            /packs/manifest.json
          </a>
          . Pipeline &amp; rebuild:{" "}
          <a className="row-link" href="https://github.com/koimari/scryglass">
            github.com/koimari/scryglass
          </a>
          .
        </p>
      </section>

      <section className="space-y-2 text-sm text-[var(--ink-muted)] border-t border-[var(--line)] pt-5">
        <h2 className="font-display text-lg text-[var(--ink)]">Attribution</h2>
        <p>{man.attribution}</p>
      </section>
    </div>
  );
}
