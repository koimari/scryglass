import Link from "next/link";
import type { Metadata } from "next";
import { ARTICLES } from "@/lib/articles";

export const metadata: Metadata = {
  title: "Articles — Scryglass",
  description: "Independent League of Legends research essays from Scryglass.",
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

export default function ArticlesIndexPage() {
  const sorted = [...ARTICLES].sort((a, b) => b.date.localeCompare(a.date));

  return (
    <div className="space-y-[var(--space-5)]">
      <header className="page-header">
        <p className="blog-kicker">Articles</p>
        <h1 className="font-display mt-3 text-[2.25rem] leading-tight text-[var(--ink)] sm:text-[2.75rem]">
          Research notes
        </h1>
        <p className="lede">
          Essays that test ideas against reproducible match data. Interactive charts sit inside the
          argument as evidence.
        </p>
      </header>

      <ul className="article-index">
        {sorted.map((a) => (
          <li key={a.slug} className="article-index-card">
            <p className="blog-kicker">
              {a.topic} · {formatDate(a.date)} · {a.readingMinutes} min · {a.author}
            </p>
            <h2 className="font-display text-2xl mt-2">
              <Link href={a.href} className="hover:text-[var(--accent-ink)]">
                {a.title}
              </Link>
            </h2>
            <p className="mt-2 text-sm text-[var(--ink-muted)] max-w-[62ch]">{a.dek}</p>
            <Link href={a.href} className="blog-cta mt-4 inline-flex">
              Read <span aria-hidden>»</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
