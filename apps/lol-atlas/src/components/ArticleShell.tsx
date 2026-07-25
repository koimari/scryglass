import Link from "next/link";
import type { ReactNode } from "react";
import type { ArticleMeta } from "@/lib/articles";

type Props = {
  meta: ArticleMeta;
  packId?: string;
  kicker?: string;
  children: ReactNode;
  /** Optional micro-log under the title (contest bar, pack, etc.). */
  stats?: ReactNode;
};

function formatDate(iso: string) {
  const d = new Date(`${iso}T12:00:00Z`);
  return d.toLocaleDateString("en-GB", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function ArticleShell({ meta, packId, kicker, children, stats }: Props) {
  return (
    <article className="article">
      <header className="article-header">
        <p className="blog-kicker">
          {kicker ?? `Article · ${meta.topic}`}
        </p>
        <h1 className="article-title">{meta.title}</h1>
        <p className="article-dek">{meta.dek}</p>
        <div className="article-byline micro-log">
          <span>
            <strong>By</strong> {meta.author}
          </span>
          <span>
            <strong>Published</strong> {formatDate(meta.date)}
          </span>
          <span>
            <strong>Read</strong> ~{meta.readingMinutes} min
          </span>
          {packId ? (
            <span>
              <strong>Pack</strong> {packId}
            </span>
          ) : null}
        </div>
        {stats ? <div className="article-stats micro-log mt-4">{stats}</div> : null}
      </header>

      <div className="article-body">{children}</div>

      {meta.footnotes && meta.footnotes.length > 0 ? (
        <section className="article-footnotes" aria-labelledby="article-notes">
          <h2 id="article-notes" className="article-section-title">
            Notes
          </h2>
          <ol className="article-footnote-list">
            {meta.footnotes.map((fn, i) => (
              <li key={fn.id} id={fn.id}>
                <span className="article-fn-mark" aria-hidden>
                  {i + 1}.
                </span>{" "}
                {fn.text}
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      {meta.related && meta.related.length > 0 ? (
        <section className="article-related" aria-labelledby="article-related">
          <h2 id="article-related" className="article-section-title">
            Related research
          </h2>
          <ul className="article-related-list">
            {meta.related.map((r) => (
              <li key={r.href}>
                <Link href={r.href} className="article-related-link">
                  {r.label}
                </Link>
                {r.blurb ? <p className="article-related-blurb">{r.blurb}</p> : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <footer className="article-data-footer">
        <h2 className="article-section-title">Data &amp; methods</h2>
        <p>
          Reproduce numbers from the pinned pack
          {packId ? (
            <>
              {" "}
              <span className="font-mono">{packId}</span>
            </>
          ) : null}
          . See{" "}
          <Link href="/reproduce" className="row-link">
            Data &amp; reproduction
          </Link>{" "}
          and{" "}
          <Link href="/methodology" className="row-link">
            Methodology
          </Link>
          .
        </p>
        <p className="mt-2">
          <Link href="/articles" className="row-link">
            ← All articles
          </Link>
        </p>
      </footer>
    </article>
  );
}

/** Figure wrapper: caption under evidence charts/tables. */
export function EvidenceFigure({
  caption,
  children,
  id,
}: {
  caption: string;
  children: ReactNode;
  id?: string;
}) {
  return (
    <figure className="evidence-figure" id={id}>
      <div className="evidence-figure-body">{children}</div>
      <figcaption className="evidence-caption">{caption}</figcaption>
    </figure>
  );
}
