import Link from "next/link";
import { ARTICLES } from "@/lib/articles";

export const metadata = {
  title: "Articles — Scryglass",
  description: "Independent League of Legends research essays from Scryglass.",
};

export default function ArticlesIndexPage() {
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

      <ul className="space-y-6 max-w-[68ch]">
        {ARTICLES.map((a) => (
          <li key={a.slug} className="border-t border-[var(--line)] pt-5">
            <p className="blog-kicker">
              {a.topic} · {a.date} · {a.readingMinutes} min
            </p>
            <h2 className="font-display text-2xl mt-2">
              <Link href={a.href} className="hover:text-[var(--accent-ink)]">
                {a.title}
              </Link>
            </h2>
            <p className="mt-2 text-sm text-[var(--ink-muted)]">{a.dek}</p>
            <Link href={a.href} className="blog-cta mt-4 inline-flex">
              Read <span aria-hidden>»</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
