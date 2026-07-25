import Link from "next/link";
import { promises as fs } from "fs";
import path from "path";
import { formatMb, type PackManifest } from "@/lib/pack";

async function readManifest(): Promise<PackManifest | null> {
  try {
    const p = path.join(process.cwd(), "public", "packs", "manifest.json");
    const raw = await fs.readFile(p, "utf8");
    return JSON.parse(raw) as PackManifest;
  } catch {
    return null;
  }
}

export default async function HomePage() {
  const man = await readManifest();
  return (
    <div>
      <article className="blog-hero anim-fade-up">
        <span className="aperture tl" aria-hidden />
        <span className="aperture tr" aria-hidden />
        <p className="blog-kicker">Latest finding · Void grubs</p>
        <h1 className="blog-title">
          Leave still wins at <span className="olive">50/50</span>
        </h1>
        <p className="blog-dek">
          Two-wave leave vs contest: the contest bar sits at about 58.9% fight-win. At even fights
          the model still prefers leave by about −2.1pp — the article number, not the OE leave-mix
          sister figure.
        </p>
        <Link href="/grubs" className="blog-cta">
          Read the study <span aria-hidden>»</span>
        </Link>
        <div className="micro-log mt-8">
          <span>
            <strong>Estimand</strong> two-wave leave
          </span>
          <span>
            <strong>Contest bar</strong> 58.9%
          </span>
          <span>
            <strong>At 50/50</strong> leave preferred
          </span>
        </div>
      </article>

      <nav className="research-index anim-fade-up-delay-1" aria-label="Research index">
        <Link href="/grubs">
          <span className="idx-label">01</span>
          <span>
            <span className="idx-title">Void grubs</span>
            <span className="idx-blurb block">
              Interactive edge chart for the opportunity-cost surface.
            </span>
          </span>
        </Link>
        <Link href="/elo">
          <span className="idx-label">02</span>
          <span>
            <span className="idx-title">Dual Elo</span>
            <span className="idx-blurb block">Team and player ladders with Elo→WR context.</span>
          </span>
        </Link>
        <Link href="/browse">
          <span className="idx-label">03</span>
          <span>
            <span className="idx-title">Browse maps</span>
            <span className="idx-blurb block">
              One row per OE map — gold, dragons, grubs, towers, GD@15.
            </span>
          </span>
        </Link>
        <Link href="/browse/head-to-head">
          <span className="idx-label">04</span>
          <span>
            <span className="idx-title">Head-to-head</span>
            <span className="idx-blurb block">Meetings plus ticker-style post-game boards.</span>
          </span>
        </Link>
        <Link href="/reproduce">
          <span className="idx-label">05</span>
          <span>
            <span className="idx-title">Reproduce</span>
            <span className="idx-blurb block">Versioned parquet pack and study files.</span>
          </span>
        </Link>
      </nav>

      {man && (
        <section className="pack-ledger anim-fade-up-delay-2" aria-label="Current pack">
          <div className="micro-log">
            <span>
              <strong>Pack</strong> {man.pack_id}
            </span>
            <span>
              <strong>Size</strong> {formatMb(man.total_bytes)}
            </span>
            <span>
              <strong>Years</strong> {man.filters.years.join("–")}
            </span>
            <span>
              <strong>Schema</strong> {man.schema_version}
            </span>
          </div>
        </section>
      )}
    </div>
  );
}
