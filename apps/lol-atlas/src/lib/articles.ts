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

/** Frozen editorial expectations for the canonical article JSON. */
export const VOID_GRUBS_PUBLICATION = {
  articlePath: "studies/grubs/grubs_article_contest_ev.json",
  schemaVersion: "scryglass.grubs.article.v1",
  publicationId: "void-grubs-contest-or-leave.patch-26.11",
  mechanicsPatch: "26.11+",
  contestBarPct: 58.24,
  atFiftyEdgePp: -1.94,
} as const;

const voidGrubsTitle = "Void grubs: contest or leave?";
const voidGrubsDek =
  "A current-mechanics opportunity-cost sensitivity comparing a river contest with two waves of farm.";

export const ARTICLES: ArticleMeta[] = [
  {
    slug: "void-grubs-contest-or-leave",
    title: voidGrubsTitle,
    dek: voidGrubsDek,
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
        id: "fn-objective",
        text: "Patch 26.11+ objective equivalent: 90g cash plus a 34.13g upper-bound first-plate progress equivalent from eight seconds of maintained three-stack Touch burn. It is not guaranteed immediate gold.",
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
        blurb: "Model-specific team strength, lineup signal, and player performance.",
      },
      {
        href: "/browse",
        label: "Match explorer",
        blurb: "Map ledgers with objectives and gold.",
      },
      {
        href: "/methodology",
        label: "Methodology",
        blurb: "Estimands, validation gates, ratings, draft policy, and pack scope.",
      },
      {
        href: "/reproduce",
        label: "Data & reproduction",
        blurb: "The canonical current-mechanics article artifact.",
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
