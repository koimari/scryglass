import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Data Sources",
  description: "The public sources and publication rules behind Scryglass.",
};

export default function SourcesPage() {
  return (
    <article className="page-prose public-note">
      <header className="prose-head">
        <p className="blog-kicker">Receipts first</p>
        <h1>Data Sources</h1>
        <p>Every public result comes from an accepted release with source dates and integrity checks.</p>
      </header>
      <div className="method-content">
        <section>
          <h2>Completed games</h2>
          <p><a className="row-link" href="https://oracleselixir.com/tools/downloads" target="_blank" rel="noreferrer">Oracle&apos;s Elixir</a> supplies public professional match data. Scryglass publishes derived ratings and summaries. Raw source rows stay outside the public pack.</p>
        </section>
        <section>
          <h2>Schedules and identities</h2>
          <p><a className="row-link" href="https://lol.fandom.com/wiki/Leaguepedia:Copyrights" target="_blank" rel="noreferrer">Leaguepedia</a> supplies future fixtures and public identity metadata. Licensable wiki content is available under CC BY-SA 3.0.</p>
        </section>
        <section>
          <h2>Game assets</h2>
          <p><a className="row-link" href="https://www.communitydragon.org/" target="_blank" rel="noreferrer">CommunityDragon</a> exposes game-client assets used for champion images. Riot Games and the relevant teams retain their rights in names, marks, and artwork.</p>
        </section>
        <section>
          <h2>Publication boundary</h2>
          <p>A failed identity, freshness, integrity, or authority check keeps the previous accepted result online. Unsupported fields appear as unavailable.</p>
          <p><Link className="row-link" href="/methodology">View Methodology</Link></p>
        </section>
      </div>
    </article>
  );
}
