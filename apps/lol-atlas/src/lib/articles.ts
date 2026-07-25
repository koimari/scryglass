/** Editorial article registry for Scryglass. */

export type ArticleMeta = {
  slug: string;
  title: string;
  dek: string;
  date: string;
  topic: string;
  readingMinutes: number;
  href: string;
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
  },
];

export function latestArticle(): ArticleMeta {
  return ARTICLES[0];
}
