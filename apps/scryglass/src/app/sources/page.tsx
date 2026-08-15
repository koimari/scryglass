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
          <h2>Completed Games</h2>
          <p><a className="row-link" href="https://lol.timsevenhuysen.com/matchdata/" target="_blank" rel="noreferrer">Oracle&apos;s Elixir match data</a> supplies public professional game facts. Its <a className="row-link" href="https://lol.timsevenhuysen.com/about/frequently-asked-questions/#comment-148" target="_blank" rel="noreferrer">official noncommercial-use record</a> says the data can only be used noncommercially. Scryglass is a noncommercial independent research publication. It publishes derived ratings and summaries while raw source rows stay outside the public pack.</p>
        </section>
        <section>
          <h2>Schedules And Identities</h2>
          <p><a className="row-link" href="https://wiki.leagueoflegends.com/en-us/Leaguepedia:Copyrights" target="_blank" rel="noreferrer">Leaguepedia</a> supplies future fixtures and public identity metadata. Its copyright page defines the terms for community content.</p>
        </section>
        <section>
          <h2>Game Assets</h2>
          <p><a className="row-link" href="https://www.communitydragon.org/" target="_blank" rel="noreferrer">CommunityDragon</a> exposes game-client assets used for champion images. Riot Games and the relevant teams retain their rights in names, marks, and artwork.</p>
        </section>
        <section>
          <h2>Publication Boundary</h2>
          <p>A failed identity, freshness, integrity, or authority check keeps the previous accepted result online. Unsupported fields appear as unavailable.</p>
          <p><Link className="row-link" href="/methodology">View Methodology</Link></p>
        </section>
      </div>
    </article>
  );
}
