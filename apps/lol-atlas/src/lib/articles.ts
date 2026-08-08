/** Editorial article registry for Scryglass. */

export type ArticleLink = { href: string; label: string; blurb?: string };

export type ArticleMeta = {
  slug: string;
  title: string;
  dek: string;
  date: string;
  topic: string;
  readingMinutes: number;
  href: string;
  author: string;
  footnotes?: { id: string; text: string }[];
  related?: ArticleLink[];
};

export const ARTICLES: ArticleMeta[] = [
  {
    slug: "void-grubs-contest-or-leave",
    title: "Leave still wins at 50/50",
    dek: "Void grubs opportunity-cost: when the contest bar sits near 58.9% fight-win, a coin-flip river fight still favors leaving for two waves of farm.",
    date: "2026-07-25",
    topic: "Void grubs",
    readingMinutes: 12,
    href: "/articles/void-grubs-contest-or-leave",
    author: "koi",
    footnotes: [
      {
        id: "fn-contest-bar",
        text: "Contest bar = fight-win chance where expected map-win from contesting equals expected map-win from leaving (two-wave farm reference).",
      },
      {
        id: "fn-pp",
        text: "pp = percentage points of map win rate under the side-neutral gold@10 logit conversion used in the article estimand.",
      },
      {
        id: "fn-24",
        text: "The ~24% figure is an Oracle’s Elixir trailing-team leave-mix break-even — a different question from the article contest bar.",
      },
      {
        id: "fn-assoc",
        text: "Gold@10 → map-win uses a side-neutral associational logit conversion under the article estimand.",
      },
    ],
    related: [
      {
        href: "/elo",
        label: "Ratings",
        blurb: "Dual Elo ladders for teams and players.",
      },
      {
        href: "/browse",
        label: "Match explorer",
        blurb: "Map ledgers with objectives and gold.",
      },
      {
        href: "/methodology",
        label: "Methodology",
        blurb: "Estimands, ratings, withheld Draft Score status, and pack years.",
      },
      {
        href: "/reproduce",
        label: "Data & reproduction",
        blurb: "Pinned pack files for this study.",
      },
    ],
  },
];

export function latestArticle(): ArticleMeta {
  return [...ARTICLES].sort((a, b) => b.date.localeCompare(a.date))[0];
}

export function getArticle(slug: string): ArticleMeta | undefined {
  return ARTICLES.find((a) => a.slug === slug);
}
