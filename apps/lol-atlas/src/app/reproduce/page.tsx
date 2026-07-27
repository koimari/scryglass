import { formatMb, packDataThroughLabel, packUpdatedLabel, packUrl, type PackFile, type PackManifest } from "@/lib/pack";
import { compositionRuntimeMetadata } from "@/lib/draftComposition";
import { readValidatedGrubsArticlePublication } from "@/lib/grubsArticlePublication.server";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

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
      "features/player_performance_snapshot.json",
      "features/player_performance_snapshot.parquet",
      "features/player_performance_meta.json",
      "features/player_performance_validation.json",
      "features/team_records.json",
      "features/player_records.json",
      "features/major_teams.json",
      "models/elo_wr_calibration.json",
      "models/elo_year_holdup.json",
      "models/draft_wr_calibration.json",
      "models/model_validation_2026-07-27.json",
      "meta/source_summary.json",
    ],
  },
  {
    group: "Article input",
    paths: [
      "studies/grubs/grubs_article_contest_ev.json",
    ],
  },
];

function findFile(man: PackManifest, rel: string): PackFile | undefined {
  return man.files.find((f) => f.path === rel || f.relative === rel);
}

export default async function ReproducePage() {
  const man = await readPackManifest();
  const compositionPath = "models/draft_composition.json";
  const runtimeMetadata = compositionRuntimeMetadata();
  let compositionVerified = false;
  if (runtimeMetadata && findFile(man, compositionPath)) {
    try {
      await readPackJson(man, compositionPath, {
        expectedSha256: runtimeMetadata.artifact_sha256,
      });
      compositionVerified = true;
    } catch {
      compositionVerified = false;
    }
  }
  const grubsPublication = await readValidatedGrubsArticlePublication(man);
  const essentials = [
    ...ESSENTIALS.filter(
      (group) => group.group !== "Article input" || grubsPublication,
    ),
    ...(compositionVerified
      ? [{ group: "Composition model", paths: [compositionPath] }]
      : []),
  ];

  const listed: { group: string; file: PackFile; path: string }[] = [];
  for (const g of essentials) {
    for (const p of g.paths) {
      const f = findFile(man, p);
      if (f) listed.push({ group: g.group, file: f, path: p });
    }
  }
  const listedBytes = listed.reduce((s, x) => s + x.file.bytes, 0);
  const listedGroups = essentials
    .map((group) => ({
      group: group.group,
      files: listed.filter((item) => item.group === group.group),
    }))
    .filter((group) => group.files.length > 0);

  return (
    <div className="space-y-8">
      <header className="page-header">
        <p className="blog-kicker">Pack · Reproduce</p>
        <h1 className="font-display mt-2 text-3xl">Reproduce</h1>
        <p className="lede">
          Cite <span className="font-mono text-sm text-[var(--ink)]">{man.pack_id}</span>. This list
          contains only the available, manifest-declared finished files cited on Scryglass. Rebuild
          from the GitHub repo when you need the warehouse pipeline.
        </p>
        <p className="method-note">
          The pack separates canonical map inclusion from optional detail enrichment. Oracle&apos;s
          Elixir precedence, verified GRID gap fills, and GRID event-detail contributions are
          declared independently in the source summary.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Pack published</strong> {packUpdatedLabel(man)}
          </span>
          <span>
            <strong>Data through</strong> {packDataThroughLabel(man)}
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
            Ratings: keep the team model, Player Dual Elo, and the role-specific 15-minute
            resource-performance artifacts separate. The latter includes its exact estimand,
            non-estimands, fit-through date, model hash, and compact frozen-test evidence; it is
            not a general player-skill rating.
          </li>
          <li>
            {compositionVerified
              ? "The composition artifact is included after its manifest hash reconciled with the runtime hash. It reproduces a withheld candidate, not a passing probability model."
              : "The composition artifact is unavailable as a verified download in this pack; do not treat this file list as sufficient to reproduce the Draft Score gate."}
          </li>
          <li>
            {grubsPublication
              ? "Void grubs: use only the validated current-mechanics article JSON."
              : "Void grubs: article downloads are withheld because the current pack does not pass the article integrity contract."}
          </li>
        </ol>
      </section>

      {listedGroups.map((group) => (
        <section key={group.group} className="space-y-3">
          <h2 className="font-display text-lg">{group.group}</h2>
          <ul className="text-sm">
            {group.files.map(({ file, path: relativePath }) => (
                <li
                  key={relativePath}
                  className="flex flex-wrap items-baseline justify-between gap-2 py-2.5 border-b border-[var(--line)]"
                >
                  <a className="font-mono break-all text-xs underline sm:text-sm" href={packUrl(man, file.path)}>
                    {relativePath}
                  </a>
                  <span className="font-mono text-xs text-[var(--ink-muted)]">
                    {(file.bytes / 1024).toFixed(0)} KB
                    {file.rows != null ? ` · ${file.rows} rows` : ""}
                  </span>
                </li>
              ))}
          </ul>
        </section>
      ))}

      <section className="space-y-2 text-sm text-[var(--ink-muted)] border-t border-[var(--line)] pt-5">
        <h2 className="font-display text-lg text-[var(--ink)]">Power users</h2>
        <p>
          Immutable manifest:{" "}
          <a className="row-link" href={packUrl(man, "manifest.json")}>
            {man.pack_id}/manifest.json
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
