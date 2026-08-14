/** Static support-chat content: methodology sections and navigation help. */

export type MethodologyTopic =
  | "ratings"
  | "grades"
  | "tiers"
  | "draft"
  | "matches"
  | "schedule"
  | "elo"
  | "all";

export const METHODOLOGY_SECTIONS: Record<MethodologyTopic, { title: string; body: string }> = {
  ratings: {
    title: "Ratings",
    body: "Scryglass rates teams and players with a Dual Elo system fit on the complete Oracle's Elixir game history. Each side carries a mean (mu) and an uncertainty (sigma); the displayed win probability is the chance the higher-rated side wins against an even opponent, calibrated to real outcomes. Ratings update sequentially, so a rating is always a strictly-prior statement about games played before it.",
  },
  elo: {
    title: "Ratings (Elo)",
    body: "Ratings are Dual Elo values for teams and players: mu is the strength estimate, sigma the uncertainty. Lower sigma means a more settled rating. The Elo win probability converts a rating gap into a calibrated chance of winning.",
  },
  grades: {
    title: "Player grades",
    body: "Each completed game grades every player A–F. A grade compares the player with their usual form, teammates, role opponent, and league-role history — not a raw stat line. A standout · B strong · C typical · D below standard · F poor.",
  },
  tiers: {
    title: "Tier lists",
    body: "Patch-wide champion tier lists per role, built from the Dual Elo and matchup evidence. Tiers run S/A/B/C; the lists are descriptive (how champions have performed), not recommendations or betting signals.",
  },
  draft: {
    title: "Draft Score",
    body: "Draft Score is a private research track. Public draft results stay unavailable until an independent review issues a release-bound promotion receipt.",
  },
  matches: {
    title: "Matches",
    body: "Match results come from Oracle's Elixir (OE) professional game data. Each match shows the rosters, champions, KDA, and player grades when OE supplies a complete stat line. Games are accepted only with two complete, uniquely identified five-player sides.",
  },
  schedule: {
    title: "Schedule",
    body: "Upcoming fixtures come from Leaguepedia. Schedule availability can lag or be partial; completed results always take precedence over a listed fixture.",
  },
  all: {
    title: "How Scryglass works",
    body: "Scryglass publishes team ratings, player ratings, patch-wide tier lists, and match results for professional League of Legends. Data refreshes every six hours from Oracle's Elixir. Ratings use Dual Elo, grades run from A to F per game, and tier lists cover each role across a patch. Draft Score stays unavailable until its separate promotion gate passes.",
  },
};

export const NAVIGATION_HELP: Array<{ page: string; path: string; description: string }> = [
  { page: "Elo / ratings", path: "/elo", description: "Team and player rating ladders with league and role filters." },
  { page: "Matches", path: "/matches", description: "Completed games and upcoming fixtures; open a match for rosters, champions, KDA, and grades." },
  { page: "Tier lists", path: "/tiers", description: "Patch-wide champion tier lists per role, with role and regional views." },
  { page: "Methodology", path: "/methodology", description: "How ratings, grades, and tier lists are computed, plus the Draft Score release gate." },
  { page: "Chat", path: "/chat", description: "Ask Scryglass about players, teams, matches, ratings, tiers, schedules, or methodology." },
];

export function matchTopic(text: string): MethodologyTopic | null {
  const lower = text.toLowerCase();
  if (/(draft|win share|composition)/.test(lower)) return "draft";
  if (/(grade|a grade|b grade)/.test(lower)) return "grades";
  if (/(tier|patch|champion list)/.test(lower)) return "tiers";
  if (/(schedule|upcoming|fixture|when does|next game)/.test(lower)) return "schedule";
  if (/(match|game|results)/.test(lower)) return "matches";
  if (/(rating|elo)/.test(lower)) return "ratings";
  return null;
}
