import { promises as fs } from "fs";
import path from "path";
import { formatMb, type PackManifest } from "@/lib/pack";
import { packUrl } from "@/lib/pack";

export default async function ReproducePage() {
  const man = JSON.parse(
    await fs.readFile(path.join(process.cwd(), "public", "packs", "manifest.json"), "utf8"),
  ) as PackManifest;

  return (
    <div className="space-y-8">
      <header className="page-header">
        <p className="blog-kicker">Pack · Reproduce</p>
        <h1 className="font-display mt-2 text-3xl">Reproduce</h1>
        <p className="lede">
          Cite <span className="font-mono text-sm text-[var(--ink)]">{man.pack_id}</span> when
          matching a published finding. Download files below or load the same parquet URLs in
          DuckDB / Polars.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Size</strong> {formatMb(man.total_bytes)}
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
          <li>Pin this pack version (do not mix with a newer export).</li>
          <li>Apply the same year / league / patch filters stated in the post.</li>
          <li>
            Use pinned calibration under <span className="font-mono">models/</span> when the post
            reports Elo→WR.
          </li>
          <li>
            Void grubs: use{" "}
            <a className="row-link" href="/articles/void-grubs-contest-or-leave">
              the void-grubs article
            </a>{" "}
            and <span className="font-mono">studies/grubs/</span> — do not mix leave-mix break-even
            with the article contest bar.
          </li>
          <li>
            Timelines / Live Stats are not in this pack unless a study add-on is linked separately.
          </li>
        </ol>
      </section>

      <section className="space-y-3">
        <h2 className="font-display text-lg">Files</h2>
        <ul className="text-sm">
          {man.files.map((f) => (
            <li
              key={f.path}
              className="flex flex-wrap items-baseline justify-between gap-2 py-2.5 border-b border-[var(--line)]"
            >
              <a className="font-mono text-xs underline sm:text-sm" href={packUrl(man, f.path)}>
                {f.path}
              </a>
              <span className="font-mono text-xs text-[var(--ink-muted)]">
                {(f.bytes / 1024).toFixed(0)} KB
                {f.rows != null ? ` · ${f.rows} rows` : ""}
              </span>
            </li>
          ))}
        </ul>
        <p className="text-xs text-[var(--ink-muted)]">
          Manifest:{" "}
          <a className="row-link" href="/packs/manifest.json">
            /packs/manifest.json
          </a>
        </p>
      </section>

      <section className="space-y-2 text-sm text-[var(--ink-muted)] border-t border-[var(--line)] pt-5">
        <h2 className="font-display text-lg text-[var(--ink)]">Attribution</h2>
        <p>{man.attribution}</p>
      </section>
    </div>
  );
}
