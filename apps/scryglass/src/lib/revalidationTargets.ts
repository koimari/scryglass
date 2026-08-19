export type RevalidationTarget = Readonly<{
  path: string;
  type: "page" | "route";
}>;

/**
 * Public pages and route handlers that depend on the active release.
 * Dynamic entries must use the Next.js target type for their handler.
 */
export const REVALIDATED_TARGETS: readonly RevalidationTarget[] = [
  { path: "/", type: "page" },
  { path: "/elo", type: "page" },
  { path: "/matches", type: "page" },
  { path: "/tiers", type: "page" },
  { path: "/chat", type: "page" },
  { path: "/methodology", type: "page" },
  { path: "/packs/manifest.json", type: "route" },
  { path: "/rankings/tierlists.json", type: "route" },
  { path: "/rankings/tierlists-latest.json", type: "route" },
  { path: "/api/public-data/tierlists", type: "route" },
  { path: "/api/chat/compare_players", type: "route" },
  { path: "/api/chat/leaderboards", type: "route" },
  { path: "/api/chat/matches", type: "route" },
  { path: "/api/chat/methodology", type: "route" },
  { path: "/api/chat/navigation", type: "route" },
  { path: "/api/chat/player", type: "route" },
  { path: "/api/chat/player_stats", type: "route" },
  { path: "/api/chat/query_champions", type: "route" },
  { path: "/api/chat/query_drafts", type: "route" },
  { path: "/api/chat/query_players", type: "route" },
  { path: "/api/chat/schedule", type: "route" },
  { path: "/api/chat/team", type: "route" },
  { path: "/api/chat/team_stats", type: "route" },
  { path: "/api/chat/tier", type: "route" },
  { path: "/elo/player/[player]", type: "page" },
  { path: "/elo/team/[team]", type: "page" },
  { path: "/matches/[game]", type: "page" },
  { path: "/api/assets/[...path]", type: "route" },
];
