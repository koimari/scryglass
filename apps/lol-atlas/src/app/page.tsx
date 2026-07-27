import Link from "next/link";
import { ARTICLES, latestArticle } from "@/lib/articles";

export default function HomePage() {
  const lead = latestArticle();
  const rest = ARTICLES.slice(1);

  return (
    <div>
      <article className="blog-hero anim-fade-up">
        <span className="aperture tl" aria-hidden />
        <span className="aperture tr" aria-hidden />
        <p className="blog-kicker">
          Latest · {lead.topic} · {lead.date}
        </p>
        <h1 className="blog-title">{lead.title}</h1>
        <p className="blog-dek">{lead.dek}</p>
        <Link href={lead.href} className="blog-cta">
          Read the article <span aria-hidden>»</span>
        </Link>
        <div className="micro-log mt-8">
          <span>
            <strong>Scope</strong> {lead.topic}
          </span>
          <span>
            <strong>Published</strong> {lead.date}
          </span>
          <span>
            <strong>Read</strong> ~{lead.readingMinutes} min
          </span>
        </div>
      </article>

      {rest.length > 0 && (
        <section className="mt-[var(--space-5)] space-y-4 anim-fade-up-delay-1" aria-label="More articles">
          <h2 className="font-display text-xl">More writing</h2>
          <ul className="space-y-4 max-w-[68ch]">
            {rest.map((a) => (
              <li key={a.slug}>
                <Link href={a.href} className="font-display text-lg hover:text-[var(--accent-ink)]">
                  {a.title}
                </Link>
                <p className="text-sm text-[var(--ink-muted)] mt-1">{a.dek}</p>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section
        className="mt-[var(--space-5)] border-t border-[var(--line)] pt-6 anim-fade-up-delay-2"
        aria-label="Explore the data"
      >
        <h2 className="font-display text-xl">Explore the data</h2>
        <p className="mt-2 text-sm text-[var(--ink-muted)] max-w-[62ch]">
          Supporting instruments for the essays — model-specific ratings, canonical series boards,
          and head-to-head records.
        </p>
        <nav className="research-index mt-4" aria-label="Analysis tools">
          <Link href="/elo">
            <span className="idx-label">01</span>
            <span>
              <span className="idx-title">Ratings</span>
              <span className="idx-blurb block">
                Team strength, shared lineup signal, and identified performance kept separate.
              </span>
            </span>
          </Link>
          <Link href="/browse">
            <span className="idx-label">02</span>
            <span>
              <span className="idx-title">Matches</span>
              <span className="idx-blurb block">
                Complete and incomplete series with provenance and frozen diagnostics.
              </span>
            </span>
          </Link>
          <Link href="/browse/head-to-head">
            <span className="idx-label">03</span>
            <span>
              <span className="idx-title">Head-to-head</span>
              <span className="idx-blurb block">Meetings and post-game boards.</span>
            </span>
          </Link>
          <Link href="/articles">
            <span className="idx-label">04</span>
            <span>
              <span className="idx-title">All articles</span>
              <span className="idx-blurb block">The full research index.</span>
            </span>
          </Link>
        </nav>
      </section>
    </div>
  );
}
