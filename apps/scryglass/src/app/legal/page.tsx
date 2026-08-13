import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Legal And Credits",
  description: "Scryglass software terms, source credits, and Riot Games notice.",
};

export default function LegalPage() {
  return (
    <article className="page-prose public-note">
      <header className="prose-head">
        <p className="blog-kicker">Rights and credits</p>
        <h1>Legal</h1>
        <p>Scryglass publishes original analysis built from credited public sources.</p>
      </header>
      <div className="method-content">
        <section>
          <h2>Software</h2>
          <p>The Scryglass source code is available under its repository license. Source datasets, team marks, champion art, and third-party text keep their own terms.</p>
          <p><a className="row-link" href="https://github.com/koimari/scryglass/blob/main/LICENSE" target="_blank" rel="noreferrer">Read the software license</a></p>
        </section>
        <section>
          <h2>Riot Games notice</h2>
          <p>Scryglass isn&apos;t endorsed by Riot Games and doesn&apos;t reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.</p>
          <p><a className="row-link" href="https://developer.riotgames.com/policies/general" target="_blank" rel="noreferrer">View Riot&apos;s General Policies</a></p>
        </section>
        <section>
          <h2>Research use</h2>
          <p>Ratings and match summaries describe accepted historical data. They do not provide betting advice. Predictive output stays unavailable until its release-bound authority checks pass.</p>
        </section>
      </div>
    </article>
  );
}
