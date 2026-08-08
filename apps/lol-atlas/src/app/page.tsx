import Link from "next/link";
import { ArrowUpRightIcon } from "@phosphor-icons/react/dist/ssr";
import { ARTICLES, latestArticle } from "@/lib/articles";

export default function HomePage() {
  const lead = latestArticle();
  const rest = ARTICLES.slice(1);

  return (
    <div className="home-page">
      <article className="home-hero anim-fade-up">
        <div className="home-hero-copy">
          <p className="blog-kicker">
            Latest article · {lead.topic} · {lead.date}
          </p>
          <h1 className="blog-title">
            {lead.title.includes("50/50") ? (
              <>
                Leave still wins at <span className="olive">50/50</span>
              </>
            ) : (
              lead.title
            )}
          </h1>
          <p className="blog-dek">{lead.dek}</p>
          <div className="home-actions">
            <Link href={lead.href} className="blog-cta">
              Read article <ArrowUpRightIcon size={15} aria-hidden />
            </Link>
            <Link href="/sandbox" className="home-secondary-cta">
              Draft analysis
            </Link>
          </div>
        </div>

        <div className="home-lens" aria-label="Article evidence summary">
          <div className="home-lens-readout">
            <p>Decision threshold</p>
            <strong>58.9%</strong>
            <span>fight-win chance needed before contesting beats two waves of farm</span>
          </div>
          <dl>
            <div>
              <dt>At 50/50</dt>
              <dd>Leave</dd>
            </div>
            <div>
              <dt>Reading time</dt>
              <dd>{lead.readingMinutes} min</dd>
            </div>
            <div>
              <dt>Evidence</dt>
              <dd>Reproducible</dd>
            </div>
          </dl>
        </div>
      </article>

      {rest.length > 0 && (
        <section className="home-more-writing anim-fade-up-delay-1" aria-label="More articles">
          <div>
            <h2 className="font-display">Articles</h2>
          </div>
          <ul>
            {rest.map((a) => (
              <li key={a.slug}>
                <Link href={a.href} className="font-display">
                  {a.title}
                </Link>
                <p>{a.dek}</p>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section
        className="home-explore anim-fade-up-delay-2"
        aria-label="Explore the data"
      >
        <div className="home-section-intro">
          <h2 className="font-display">Analysis tools</h2>
          <p>
            Ratings, completed matches, head-to-head records, and reproduction files use the same
            current data pack. Draft analysis remains visible as a withheld surface while its
            independent review is incomplete.
          </p>
        </div>
        <nav className="research-index" aria-label="Analysis tools">
          <Link href="/elo">
            <span>
              <span className="idx-title">Ratings</span>
              <span className="idx-blurb block">Compare team and player strength.</span>
            </span>
            <ArrowUpRightIcon className="idx-arrow" size={21} aria-hidden />
          </Link>
          <Link href="/browse">
            <span>
              <span className="idx-title">Matches</span>
              <span className="idx-blurb block">Inspect completed series and games.</span>
            </span>
            <ArrowUpRightIcon className="idx-arrow" size={21} aria-hidden />
          </Link>
          <Link href="/browse/head-to-head">
            <span>
              <span className="idx-title">Head-to-head</span>
              <span className="idx-blurb block">Compare meetings between two teams.</span>
            </span>
            <ArrowUpRightIcon className="idx-arrow" size={21} aria-hidden />
          </Link>
          <Link href="/sandbox">
            <span>
              <span className="idx-title">Draft analysis</span>
              <span className="idx-blurb block">Unavailable pending independent review.</span>
            </span>
            <ArrowUpRightIcon className="idx-arrow" size={21} aria-hidden />
          </Link>
        </nav>
      </section>
    </div>
  );
}
