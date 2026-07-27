"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type {
  CurrentMembershipContext,
  PlayerMetadata,
  PlayerPerformanceMeta,
  PlayerPerformanceRating,
  PlayerPerformanceValidation,
  PlayerRating,
  PlayerRatingsMeta,
  PlayerRecord,
  PlayerWeeklyRanks,
  TeamRating,
  TeamRatingContract,
  TeamRatingsMeta,
  TeamRecord,
} from "@/lib/pack";
import {
  formatWr,
  INTL_LEAGUES,
  INTERREGIONAL_LEAGUES,
  MAJOR_REGIONAL_LEAGUES,
  playerAdjustedRating,
  playerIdentifiabilityInfo,
  playerMatchesQuery,
  playerSigmaFloor,
  playerSlug,
  recordMatchesLeagues,
  REGION_LEAGUES,
  SECONDARY_REGIONAL_LEAGUES,
  scopedTeamWr,
  TIER_FILTERS,
  teamBoundRating,
  teamEvidenceInfo,
  teamMatchesQuery,
  teamRatingContract,
  teamSlug,
  trustInfo,
  verifiedPlayerAffiliation,
  verifiedTeamAffiliation,
} from "@/lib/pack";

type Props = {
  teams: TeamRating[];
  players: PlayerRating[];
  teamRecords: Record<string, TeamRecord>;
  playerRecords: Record<string, PlayerRecord>;
  playerWeeklyRanks: PlayerWeeklyRanks;
  playerMetadata: Record<string, PlayerMetadata>;
  availableLeagues: string[];
  dataAsOf: string;
  recentActivityWindowDays: number;
  currentTournaments: Record<string, string>;
  membershipContext: CurrentMembershipContext;
  teamRatingsMeta: TeamRatingsMeta | null;
  playerRatingsMeta: PlayerRatingsMeta | null;
  playerOrderingVerified: boolean;
  playerPerformanceRows: PlayerPerformanceRating[] | null;
  playerPerformanceMeta: PlayerPerformanceMeta | null;
  playerPerformanceValidation: PlayerPerformanceValidation | null;
  playerPerformanceUnavailableReason: string | null;
};

type TeamCol = "team" | "league" | "soft" | "mu" | "trust" | "wr";
type PlayerCol = "player" | "last_team" | "league" | "soft" | "mu" | "trust" | "games";
type Dir = "asc" | "desc";
type RatingTab = "teams" | "players" | "performance";
type PerformanceRole = PlayerPerformanceRating["role"];

const PERFORMANCE_ROLES: readonly PerformanceRole[] = [
  "top",
  "jng",
  "mid",
  "bot",
  "sup",
];

function roleLabel(role: PerformanceRole): string {
  if (role === "jng") return "Jungle";
  if (role === "bot") return "Bot";
  if (role === "sup") return "Support";
  return role[0].toUpperCase() + role.slice(1);
}

function formatTier(tier: string | null | undefined): string {
  if (tier === "tier1") return "Tier 1";
  if (tier === "tier2") return "Tier 2";
  if (tier === "tier3") return "Tier 3";
  return "—";
}

function formatAffiliation(tier: string | null | undefined, league: string | null | undefined): string {
  if (!league) return "—";
  const tierLabel = formatTier(tier);
  return tierLabel === "—" ? league : `${tierLabel} · ${league}`;
}

function formatScope(scope: string): string {
  return TIER_FILTERS.find((tier) => tier.value === scope)?.label ?? scope;
}

function rankDeltaLabel(delta: number | null | undefined): string {
  if (delta == null || delta === 0) return "—";
  return delta > 0 ? `↑${delta}` : `↓${Math.abs(delta)}`;
}

function rankDeltaClass(delta: number | null | undefined): string {
  if (delta == null || delta === 0) return "text-[var(--ink-faint)]";
  return delta > 0 ? "text-[var(--accent-ink)]" : "text-[var(--danger)]";
}

function playerDisplayRank(
  players: PlayerRating[],
  index: number,
  column: PlayerCol,
  sigmaFloor: number | null,
  orderingVerified: boolean,
): number | null {
  if (!orderingVerified) return null;
  if (column !== "soft" && column !== "mu") return index + 1;
  const score = (player: PlayerRating) =>
    Number(
      (
        column === "soft"
          ? playerAdjustedRating(player, sigmaFloor)
          : player.mu_total
      )?.toFixed(1),
    );
  const value = score(players[index]);
  let first = index;
  while (first > 0 && score(players[first - 1]) === value) first -= 1;
  return first + 1;
}

function teamDisplayRank(
  teams: TeamRating[],
  index: number,
  column: TeamCol,
  contract: TeamRatingContract | null,
  comparable: boolean,
): number | null {
  if (!comparable) return null;
  if (column !== "soft" && column !== "mu") return index + 1;
  const score = (team: TeamRating) =>
    Number(
      (
        column === "soft"
          ? teamBoundRating(team, contract)
          : team.mu_total
      )?.toFixed(1),
    );
  if (!Number.isFinite(score(teams[index]))) return null;
  const value = score(teams[index]);
  let first = index;
  while (first > 0 && score(teams[first - 1]) === value) first -= 1;
  return first + 1;
}

function SortTh({
  label,
  col,
  active,
  dir,
  align = "left",
  title,
  onSort,
}: {
  label: string;
  col: string;
  active: boolean;
  dir: Dir;
  align?: "left" | "num";
  title?: string;
  onSort: (col: string) => void;
}) {
  return (
    <th
      className={align === "num" ? "num" : undefined}
      scope="col"
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        className={`sort-th ${active ? "is-active" : ""}`}
        onClick={() => onSort(col)}
        title={title}
      >
        {label}
        <span className="sort-ind" aria-hidden>
          {active ? (dir === "asc" ? " ↑" : " ↓") : ""}
        </span>
      </button>
    </th>
  );
}

function parseLeagues(raw: string | null): string[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function parseMinGamesParam(
  raw: string | null,
  fallback = 20,
): number {
  const parsed = raw == null || raw.trim() === "" ? fallback : Number(raw);
  if (!Number.isFinite(parsed)) return Math.max(5, Math.floor(fallback));
  return Math.min(10_000, Math.max(5, Math.floor(parsed)));
}

export function EloLadders({
  teams,
  players,
  teamRecords,
  playerRecords,
  playerWeeklyRanks,
  playerMetadata,
  availableLeagues,
  dataAsOf,
  recentActivityWindowDays,
  currentTournaments,
  membershipContext,
  teamRatingsMeta,
  playerRatingsMeta,
  playerOrderingVerified,
  playerPerformanceRows,
  playerPerformanceMeta,
  playerPerformanceValidation,
  playerPerformanceUnavailableReason,
}: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const requestedTab = searchParams.get("tab");
  const [tab, setTab] = useState<RatingTab>(
    requestedTab === "players" || requestedTab === "performance"
      ? requestedTab
      : "teams",
  );
  const requestedRole = searchParams.get("role");
  const [performanceRole, setPerformanceRole] = useState<PerformanceRole>(
    PERFORMANCE_ROLES.includes(requestedRole as PerformanceRole)
      ? (requestedRole as PerformanceRole)
      : "top",
  );
  const [q, setQ] = useState(searchParams.get("q") || "");
  const [leagues, setLeagues] = useState<string[]>(() => {
    const parsed = parseLeagues(searchParams.get("leagues"));
    return parsed.length
      ? parsed
      : membershipContext.valid
        ? ["TIER1"]
        : [];
  });
  const [minGames, setMinGames] = useState(() =>
    parseMinGamesParam(searchParams.get("min")),
  );
  const [expanded, setExpanded] = useState(false);
  const [teamCol, setTeamCol] = useState<TeamCol>("soft");
  const [teamDir, setTeamDir] = useState<Dir>("desc");
  const [playerCol, setPlayerCol] = useState<PlayerCol>(
    playerOrderingVerified ? "soft" : "player",
  );
  const [playerDir, setPlayerDir] = useState<Dir>(
    playerOrderingVerified ? "desc" : "asc",
  );
  const teamContract = useMemo(
    () => teamRatingContract(teamRatingsMeta),
    [teamRatingsMeta],
  );
  const playerFloor = useMemo(
    () => playerSigmaFloor(playerRatingsMeta),
    [playerRatingsMeta],
  );
  const chips = useMemo(() => {
    const present = new Set(availableLeagues);
    const core = [...REGION_LEAGUES];
    const shown = [
      ...core.filter((L) => availableLeagues.includes(L) || present.has(L)),
      ...REGION_LEAGUES.filter((L) => !(core as readonly string[]).includes(L) && availableLeagues.includes(L)),
      ...INTERREGIONAL_LEAGUES.filter((L) => availableLeagues.includes(L)),
      "INTL",
      ...INTL_LEAGUES.filter((L) => availableLeagues.includes(L)),
    ];
    return [...new Set(shown)];
  }, [availableLeagues]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (tab !== "teams") params.set("tab", tab);
    if (q.trim()) params.set("q", q.trim());
    if (leagues.length) params.set("leagues", leagues.join(","));
    if ((tab === "players" || tab === "performance") && minGames !== 20) {
      params.set("min", String(minGames));
    }
    if (tab === "performance" && performanceRole !== "top") {
      params.set("role", performanceRole);
    }
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }, [tab, q, leagues, minGames, performanceRole, pathname, router]);

  const toggleLeague = (lg: string) => {
    setLeagues((prev) => (prev.includes(lg) ? prev.filter((x) => x !== lg) : [...prev, lg]));
    setExpanded(false);
  };

  const onTeamSort = useCallback(
    (col: string) => {
      const c = col as TeamCol;
      if (c === teamCol) setTeamDir((d) => (d === "desc" ? "asc" : "desc"));
      else {
        setTeamCol(c);
        setTeamDir(c === "team" || c === "league" ? "asc" : "desc");
      }
    },
    [teamCol],
  );

  const onPlayerSort = useCallback(
    (col: string) => {
      const c = col as PlayerCol;
      if (c === playerCol) setPlayerDir((d) => (d === "desc" ? "asc" : "desc"));
      else {
        setPlayerCol(c);
        setPlayerDir(
          c === "player" || c === "last_team" || c === "league" ? "asc" : "desc",
        );
      }
    },
    [playerCol],
  );

  const sortedTeams = useMemo(() => {
    let list = teams.filter((t) => {
      const rec = teamRecords[t.team];
      const affiliation = verifiedTeamAffiliation(rec, membershipContext);
      if (
        !teamMatchesQuery(t.team, q) &&
        !teamMatchesQuery(affiliation?.team ?? "", q)
      ) {
        return false;
      }
      return recordMatchesLeagues(rec, leagues, {
        dataAsOf,
        recentActivityWindowDays,
        currentTournaments,
        membershipContext,
      });
    });
    const sign = teamDir === "asc" ? 1 : -1;
    list = [...list].sort((a, b) => {
      let cmp = 0;
      if (
        (teamCol === "soft" || teamCol === "mu") &&
        a.comparison_component_id !== b.comparison_component_id
      ) {
        return String(a.comparison_component_id ?? "").localeCompare(
          String(b.comparison_component_id ?? ""),
        );
      }
      const ra = teamRecords[a.team];
      const rb = teamRecords[b.team];
      switch (teamCol) {
        case "team": {
          const aa = verifiedTeamAffiliation(ra, membershipContext);
          const ab = verifiedTeamAffiliation(rb, membershipContext);
          cmp = (aa?.team ?? a.team).localeCompare(ab?.team ?? b.team);
          break;
        }
        case "league": {
          const aa = verifiedTeamAffiliation(ra, membershipContext);
          const ab = verifiedTeamAffiliation(rb, membershipContext);
          cmp = (aa?.league ?? "").localeCompare(ab?.league ?? "");
          break;
        }
        case "soft":
          cmp =
            (teamBoundRating(a, teamContract) ?? Number.NEGATIVE_INFINITY) -
            (teamBoundRating(b, teamContract) ?? Number.NEGATIVE_INFINITY);
          break;
        case "mu":
          cmp = a.mu_total - b.mu_total;
          break;
        case "trust":
          cmp = a.sigma - b.sigma;
          break;
        case "wr": {
          const wa = scopedTeamWr(ra, leagues, { currentTournaments }) ?? -1;
          const wb = scopedTeamWr(rb, leagues, { currentTournaments }) ?? -1;
          cmp = wa - wb;
          break;
        }
      }
      return cmp !== 0 ? sign * cmp : a.team.localeCompare(b.team);
    });
    return list;
  }, [teams, q, leagues, teamRecords, teamCol, teamDir, dataAsOf, recentActivityWindowDays, currentTournaments, membershipContext, teamContract]);
  const teamRanksComparable = useMemo(
    () =>
      new Set(
        sortedTeams.map(
          (team) => team.comparison_component_id ?? `missing:${team.team}`,
        ),
      ).size <= 1,
    [sortedTeams],
  );

  const sortedPlayers = useMemo(() => {
    let list = players
      .filter((p) => (p.n_maps ?? 0) >= minGames)
      .filter((p) => {
        const affiliation = verifiedPlayerAffiliation(
          playerRecords[p.player],
          membershipContext,
        );
        return playerMatchesQuery(
          p.player,
          affiliation?.team ?? null,
          q,
        );
      })
      .filter((p) => {
        const rec = playerRecords[p.player];
        const affiliation = verifiedPlayerAffiliation(rec, membershipContext);
        const fromTeam = affiliation ? teamRecords[affiliation.team] : undefined;
        return rec
          ? recordMatchesLeagues(rec, leagues, {
              dataAsOf,
              recentActivityWindowDays,
              currentTournaments,
              membershipRegistryValid: affiliation != null,
            })
          : recordMatchesLeagues(fromTeam, leagues, {
              dataAsOf,
              recentActivityWindowDays,
              currentTournaments,
              membershipContext,
            }) || (!fromTeam && !leagues.length);
      });
    // if leagues selected and no player record match via team
    if (leagues.length) {
      list = list.filter((p) => {
        const rec = playerRecords[p.player];
        const affiliation = verifiedPlayerAffiliation(rec, membershipContext);
        const fromTeam = affiliation ? teamRecords[affiliation.team] : undefined;
        return rec
          ? recordMatchesLeagues(rec, leagues, {
              dataAsOf,
              recentActivityWindowDays,
              currentTournaments,
              membershipRegistryValid: affiliation != null,
            })
          : recordMatchesLeagues(fromTeam, leagues, {
              dataAsOf,
              recentActivityWindowDays,
              currentTournaments,
              membershipContext,
            });
      });
    }
    const sign = playerDir === "asc" ? 1 : -1;
    list = [...list].sort((a, b) => {
      let cmp = 0;
      const ra = playerRecords[a.player];
      const rb = playerRecords[b.player];
      const aa = verifiedPlayerAffiliation(ra, membershipContext);
      const ab = verifiedPlayerAffiliation(rb, membershipContext);
      switch (playerCol) {
        case "player":
          cmp = a.player.localeCompare(b.player);
          break;
        case "last_team":
          cmp = (aa?.team || "").localeCompare(ab?.team || "");
          break;
        case "league":
          cmp = (aa?.league || "").localeCompare(ab?.league || "");
          break;
        case "soft":
          cmp = playerOrderingVerified
            ? (playerAdjustedRating(a, playerFloor) ?? Number.NEGATIVE_INFINITY) -
              (playerAdjustedRating(b, playerFloor) ?? Number.NEGATIVE_INFINITY)
            : a.player.localeCompare(b.player);
          break;
        case "mu":
          cmp = playerOrderingVerified
            ? a.mu_total - b.mu_total
            : a.player.localeCompare(b.player);
          break;
        case "trust":
          cmp = playerOrderingVerified
            ? a.sigma - b.sigma
            : a.player.localeCompare(b.player);
          break;
        case "games":
          cmp = (a.n_maps ?? 0) - (b.n_maps ?? 0);
          break;
      }
      return cmp !== 0 ? sign * cmp : a.player.localeCompare(b.player);
    });
    return list;
  }, [
    players,
    q,
    minGames,
    leagues,
    playerRecords,
    teamRecords,
    playerCol,
    playerDir,
    dataAsOf,
    recentActivityWindowDays,
    currentTournaments,
    membershipContext,
    playerFloor,
    playerOrderingVerified,
  ]);

  const sortedPerformance = useMemo(() => {
    if (!playerPerformanceRows) return [];
    const query = q.trim().toLocaleLowerCase();
    return playerPerformanceRows
      .filter((row) => row.role === performanceRole)
      .filter((row) => row.effective_sample_maps >= minGames)
      .filter(
        (row) =>
          !query ||
          row.player_name.toLocaleLowerCase().includes(query) ||
          row.last_team_key.toLocaleLowerCase().includes(query),
      )
      .sort(
        (a, b) =>
          b.lower_bound - a.lower_bound ||
          b.performance_mean - a.performance_mean ||
          a.player_id.localeCompare(b.player_id),
      );
  }, [playerPerformanceRows, performanceRole, minGames, q]);

  const visibleTeams = expanded ? sortedTeams : sortedTeams.slice(0, 20);
  const visiblePlayers = expanded ? sortedPlayers : sortedPlayers.slice(0, 20);
  const visiblePerformance = expanded
    ? sortedPerformance
    : sortedPerformance.slice(0, 20);
  const showPlayerRankDeltas =
    playerOrderingVerified && playerWeeklyRanks.as_of != null;
  const intlSet = new Set<string>(["INTL", ...INTL_LEAGUES]);
  const scopeSummary = leagues.length ? leagues.map(formatScope).join(" + ") : "All competitive tiers and event scopes";
  const rankScopeFor = (tier: string | null | undefined): "all" | "tier1" | "tier2" | "tier3" => {
    const selectedTiers = leagues.filter((scope) => scope.startsWith("TIER")).map((scope) => scope.toLowerCase()) as Array<"tier1" | "tier2" | "tier3">;
    if (selectedTiers.length === 1) return selectedTiers[0];
    if (tier === "tier1" || tier === "tier2" || tier === "tier3") return tier;
    return "all";
  };
  const playerRankDelta = (player: string, tier: string | null | undefined): number | null | undefined => {
    if (!showPlayerRankDeltas) return undefined;
    const scope = rankScopeFor(tier);
    return playerWeeklyRanks.by_player[player]?.[scope]?.delta;
  };
  const playerSortLabel =
    playerCol === "player"
      ? "player name"
      : playerCol === "last_team"
        ? "team name"
        : playerCol === "league"
          ? "league"
          : playerCol === "games"
            ? "games"
            : playerOrderingVerified
              ? playerCol === "soft"
                ? "adjusted outcome signal"
                : playerCol === "mu"
                  ? "raw outcome signal"
                  : "evidence"
              : "player name";
  const liveSummary =
    tab === "teams"
      ? `${sortedTeams.length} teams shown in ${scopeSummary}, sorted by ${teamCol === "soft" ? "adjusted rating" : teamCol}`
      : tab === "players"
        ? `${sortedPlayers.length} players shown in ${scopeSummary}, sorted by ${playerSortLabel}`
        : `${sortedPerformance.length} ${roleLabel(performanceRole)} player-role rows shown, sorted by the uncertainty-adjusted 15-minute resource score`;

  return (
    <div className="space-y-6">
      <p className="text-sm text-[var(--ink-muted)] max-w-[62ch]">
        Team rows are time-varying series Bradley–Terry estimates. Player Dual Elo is a shared
        team-outcome signal with identifiability warnings. The separate 15-minute
        resource-performance view is role-relative and descriptive, not a general skill,
        win-contribution, or win-probability rating.
      </p>
      {tab === "performance" ? (
        <p className="method-note max-w-[72ch]">
          Within-role score: the equal-weight mean of training-only robust standardized
          gold, experience, and creep-score differentials at 15 minutes, adjusted for
          observed champion, team, league, and patch context. Fit through{" "}
          <strong>{playerPerformanceMeta?.fit_through.slice(0, 10) ?? "unavailable"}</strong>;
          team and league columns are the last fit-period observation, not a current-roster claim.
        </p>
      ) : (
        <p className="method-note max-w-[62ch]">
          Scope: <strong>{scopeSummary}</strong>. Domestic views require the pack’s reviewed Riot current-tournament
          registry; match appearances cannot create current membership. Historical and international evidence remains
          part of the rating. A win rate is shown only when the selected chips identify one supported
          map denominator; mixed domestic/international scopes are withheld.
        </p>
      )}
      {tab !== "performance" && !membershipContext.valid ? (
        <p className="error-banner">
          Current domestic ladders are unavailable because the membership registry is missing or overdue for review.
        </p>
      ) : null}
      {tab === "teams" && !teamContract ? (
        <p className="error-banner">
          Team uncertainty-adjusted ratings are unavailable because the dynamic series
          model metadata is missing or inconsistent.
        </p>
      ) : null}
      {tab === "teams" && !teamRanksComparable ? (
        <p className="error-banner">
          Rank numbers are withheld because the selected rows span disconnected
          historical comparison components. Ratings are ordered only within each
          connected component.
        </p>
      ) : null}
      {tab === "players" && !playerOrderingVerified ? (
        <p className="error-banner">
          Individual player ordering is withheld because team outcomes do not identify
          individual skill in this model. Rows fall back to name order; equal values mean
          shared team-result exposure, not equal player ability.
        </p>
      ) : null}
      {tab === "players" && showPlayerRankDeltas ? (
        <p className="method-note max-w-[62ch]">
          Rank Δ compares with the previous Sunday at 00:00 UTC; flags use Leaguepedia country metadata when available.
        </p>
      ) : null}
      {tab === "performance" && !playerPerformanceRows ? (
        <p className="error-banner">
          15-minute resource performance is unavailable because the current immutable pack
          does not contain a matching snapshot, metadata, and passing validation artifact
          {playerPerformanceUnavailableReason
            ? ` (${playerPerformanceUnavailableReason})`
            : ""}. Player Dual Elo is not used as a fallback.
        </p>
      ) : null}
      <p className="sr-only" aria-live="polite">
        {liveSummary}
      </p>

      <div className="filter-bar">
        <div className="flex gap-1">
          <button
            type="button"
            className={tab === "teams" ? "btn-primary" : "status-pill ghost"}
            onClick={() => {
              setTab("teams");
              setExpanded(false);
            }}
          >
            Teams
          </button>
          <button
            type="button"
            className={tab === "players" ? "btn-primary" : "status-pill ghost"}
            onClick={() => {
              setTab("players");
              setExpanded(false);
            }}
          >
            Players
          </button>
          <button
            type="button"
            className={tab === "performance" ? "btn-primary" : "status-pill ghost"}
            onClick={() => {
              setTab("performance");
              setExpanded(false);
            }}
          >
            15-minute resource performance
          </button>
        </div>
        <label className="field grow">
          <span>Search</span>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={tab === "teams" ? "Team or alias (G2, KC…)" : "Player or team"}
          />
        </label>
        {(tab === "players" || tab === "performance") && (
          <label className="field">
            <span>{tab === "performance" ? "Min fit maps" : "Min games"}</span>
            <input
              type="number"
              min={5}
              value={minGames}
              onChange={(e) => setMinGames(Math.max(5, Number(e.target.value) || 5))}
            />
          </label>
        )}
      </div>

      {tab === "performance" ? (
        <div className="league-filter" role="group" aria-label="Role filter">
          <div className="scope-filter-head">
            <span className="chip-group-label">Role</span>
            <span className="scope-filter-help">
              Scores and ranks are comparable only within the same canonical role.
            </span>
          </div>
          <div className="chip-group">
            {PERFORMANCE_ROLES.map((role) => (
              <button
                key={role}
                type="button"
                className={`chip ${performanceRole === role ? "is-on" : ""}`}
                onClick={() => {
                  setPerformanceRole(role);
                  setExpanded(false);
                }}
              >
                {roleLabel(role)}
              </button>
            ))}
          </div>
        </div>
      ) : (
      <div className="league-filter" role="group" aria-label="Rating scope filters">
        <div className="scope-filter-head">
          <span className="chip-group-label">Scope filters</span>
          <span className="scope-filter-help">Tier sets current competitive level; leagues and events narrow within it. Groups combine with AND.</span>
          <button
            type="button"
            className={`chip ${leagues.length === 0 ? "is-on" : ""}`}
            onClick={() => setLeagues([])}
          >
            All scopes
          </button>
        </div>
        <div className="chip-group chip-group-tier">
          <span className="chip-group-label">Competitive tier</span>
          {TIER_FILTERS.filter(
            (tier) => tier.value === "TIER1" && membershipContext.valid,
          ).map((tier) => (
            <button
              key={tier.value}
              type="button"
              className={`chip ${leagues.includes(tier.value) ? "is-on" : ""}`}
              onClick={() => toggleLeague(tier.value)}
              title={tier.description}
            >
              {tier.label}
            </button>
          ))}
        </div>
        <div className="chip-group">
          <span className="chip-group-label">Major regional</span>
          {chips
            .filter((lg) => (MAJOR_REGIONAL_LEAGUES as readonly string[]).includes(lg))
            .map((lg) => (
              <button
                key={lg}
                type="button"
                className={`chip ${leagues.includes(lg) ? "is-on" : ""}`}
                onClick={() => toggleLeague(lg)}
              >
                {lg}
              </button>
            ))}
        </div>
        <div className="chip-group">
          <span className="chip-group-label">Other regional</span>
          {chips
            .filter((lg) => (SECONDARY_REGIONAL_LEAGUES as readonly string[]).includes(lg))
            .map((lg) => (
              <button
                key={lg}
                type="button"
                className={`chip ${leagues.includes(lg) ? "is-on" : ""}`}
                onClick={() => toggleLeague(lg)}
              >
                {lg}
              </button>
          ))}
        </div>
        <div className="chip-group">
          <span className="chip-group-label">Cross-region</span>
          {chips
            .filter((lg) => INTERREGIONAL_LEAGUES.includes(lg as (typeof INTERREGIONAL_LEAGUES)[number]))
            .map((lg) => (
              <button
                key={lg}
                type="button"
                className={`chip ${leagues.includes(lg) ? "is-on" : ""}`}
                onClick={() => toggleLeague(lg)}
              >
                {lg}
              </button>
            ))}
        </div>
        <div className="chip-group">
          <span className="chip-group-label">International events</span>
          {chips
            .filter((lg) => intlSet.has(lg) || (INTL_LEAGUES as readonly string[]).includes(lg))
            .map((lg) => (
              <button
                key={lg}
                type="button"
                className={`chip ${leagues.includes(lg) ? "is-on" : ""}`}
                onClick={() => toggleLeague(lg)}
              >
                {lg}
              </button>
            ))}
        </div>
      </div>
      )}

      {tab === "teams" ? (
        <>
          <div className="table-scroll elo-desktop" tabIndex={0} aria-label="Team ratings table; scroll horizontally if needed">
            <table className="data-table">
              <caption className="sr-only">
                Team ratings in the current pack. Default order is adjusted rating.
              </caption>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <SortTh label="Team" col="team" active={teamCol === "team"} dir={teamDir} onSort={onTeamSort} />
                  <SortTh
                    label="League"
                    col="league"
                    active={teamCol === "league"}
                    dir={teamDir}
                    title="Current competition affiliation and tier; international appearances stay on the profile."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label={teamContract?.boundLabel ?? "Conservative bound"}
                    col="soft"
                    active={teamCol === "soft"}
                    dir={teamDir}
                    align="num"
                    title={
                      teamContract
                        ? `Mean minus ${teamContract.conservativeZ.toFixed(3)} × display uncertainty. This is a conservative ordering score, not a calibrated posterior-coverage claim.`
                        : "Unavailable without valid dynamic model and quantile metadata."
                    }
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Raw rating"
                    col="mu"
                    active={teamCol === "mu"}
                    dir={teamDir}
                    align="num"
                    title="Current latent series strength on the model's Elo-like display scale."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Evidence"
                    col="trust"
                    active={teamCol === "trust"}
                    dir={teamDir}
                    align="num"
                    title="Diagonal Gaussian filter spread, including uncertainty growth during inactivity; no empirical coverage claim."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Win rate"
                    col="wr"
                    active={teamCol === "wr"}
                    dir={teamDir}
                    align="num"
                    title="Empirical win rate in the pack for the active league filter."
                    onSort={onTeamSort}
                  />
                </tr>
              </thead>
              <tbody>
                {visibleTeams.map((t, i) => {
                  const rec = teamRecords[t.team];
                  const evidence = teamEvidenceInfo(t.sigma, teamContract, rec?.games);
                  const wr = scopedTeamWr(rec, leagues, { currentTournaments });
                  const displayRank = teamDisplayRank(
                    sortedTeams,
                    i,
                    teamCol,
                    teamContract,
                    teamRanksComparable,
                  );
                  const affiliation = verifiedTeamAffiliation(rec, membershipContext);
                  const displayName = affiliation?.team ?? t.team;
                  const bound = teamBoundRating(t, teamContract);
                  return (
                    <tr
                      key={t.team}
                      className="row-click"
                      onClick={() => router.push(`/elo/team/${teamSlug(t.team)}`)}
                    >
                      <td className="font-mono text-[var(--ink-muted)]">{displayRank ?? "—"}</td>
                      <td className="font-medium">
                        <Link
                          href={`/elo/team/${teamSlug(t.team)}`}
                          className="row-link"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {displayName}
                        </Link>
                      </td>
                      <td>
                        {formatAffiliation(affiliation?.tier, affiliation?.league)}
                        {affiliation ? <span className="block text-xs text-[var(--ink-faint)]">{affiliation.tournament}</span> : null}
                      </td>
                      <td className="num">{bound != null ? bound.toFixed(1) : "—"}</td>
                      <td className="num">{t.mu_total.toFixed(1)}</td>
                      <td className="num" title={evidence.layman}>
                        {evidence.label}
                      </td>
                      <td className="num">{formatWr(wr)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <ul className="elo-cards elo-mobile">
            {visibleTeams.map((t, i) => {
              const rec = teamRecords[t.team];
              const evidence = teamEvidenceInfo(t.sigma, teamContract, rec?.games);
              const displayRank = teamDisplayRank(
                sortedTeams,
                i,
                teamCol,
                teamContract,
                teamRanksComparable,
              );
              const affiliation = verifiedTeamAffiliation(rec, membershipContext);
              const displayName = affiliation?.team ?? t.team;
              const bound = teamBoundRating(t, teamContract);
              return (
                <li key={t.team}>
                  <Link href={`/elo/team/${teamSlug(t.team)}`} className="elo-card">
                    <span className="elo-card-rank">{displayRank == null ? "—" : `#${displayRank}`}</span>
                    <span className="elo-card-title">{displayName}</span>
                    <span className="elo-card-meta">
                      {formatAffiliation(affiliation?.tier, affiliation?.league)} · Uncertainty {evidence.label} · Win rate{" "}
                      {formatWr(scopedTeamWr(rec, leagues, { currentTournaments }))}
                    </span>
                    <span className="elo-card-rating">
                      {bound != null ? bound.toFixed(1) : "Unavailable"}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>

          {sortedTeams.length === 0 && (
            <p className="empty-hint">
              No teams match these chips
              {q.trim() ? ` and “${q.trim()}”` : ""}. Clear a filter or try another alias.
            </p>
          )}
          {sortedTeams.length > 20 && (
            <p className="empty-hint flex flex-wrap items-center gap-3">
              Showing {visibleTeams.length} of {sortedTeams.length}.
              <button type="button" className="btn-primary" onClick={() => setExpanded((x) => !x)}>
                {expanded ? "Collapse" : "Expand"}
              </button>
            </p>
          )}
        </>
      ) : tab === "players" ? (
        <>
          <div className="table-scroll elo-desktop" tabIndex={0} aria-label="Player ratings table; scroll horizontally if needed">
            <table className="data-table">
              <caption className="sr-only">
                Shared player outcome signals in the current pack.{" "}
                {playerOrderingVerified
                  ? "The individual ordering contract is verified."
                  : "Rating-based ordering is withheld and rows default to player name."}
              </caption>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  {showPlayerRankDeltas ? (
                    <th scope="col" title="Rank movement since the previous Sunday 00:00 UTC">Δ</th>
                  ) : null}
                  <SortTh label="Player" col="player" active={playerCol === "player"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="Team" col="last_team" active={playerCol === "last_team"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="League" col="league" active={playerCol === "league"} dir={playerDir} onSort={onPlayerSort} />
                  {playerOrderingVerified ? (
                    <SortTh
                      label="Adjusted outcome signal"
                      col="soft"
                      active={playerCol === "soft"}
                      dir={playerDir}
                      align="num"
                      title="Player Dual Elo raw rating minus sigma above the model-declared minimum. Unavailable when metadata is missing."
                      onSort={onPlayerSort}
                    />
                  ) : (
                    <th className="num" scope="col">Adjusted outcome signal</th>
                  )}
                  {playerOrderingVerified ? (
                    <SortTh label="Raw outcome signal" col="mu" active={playerCol === "mu"} dir={playerDir} align="num" onSort={onPlayerSort} />
                  ) : (
                    <th className="num" scope="col">Raw outcome signal</th>
                  )}
                  {playerOrderingVerified ? (
                    <SortTh
                      label="Evidence"
                      col="trust"
                      active={playerCol === "trust"}
                      dir={playerDir}
                      align="num"
                      title="Outcome-identifiability status plus Player Dual Elo sigma evidence when both contracts are present."
                      onSort={onPlayerSort}
                    />
                  ) : (
                    <th className="num" scope="col" title="Outcome-identifiability status plus Player Dual Elo sigma evidence when both contracts are present.">
                      Evidence
                    </th>
                  )}
                  <SortTh label="Games" col="games" active={playerCol === "games"} dir={playerDir} align="num" onSort={onPlayerSort} />
                </tr>
              </thead>
              <tbody>
                {visiblePlayers.map((p, i) => {
                  const rec = playerRecords[p.player];
                  const affiliation = verifiedPlayerAffiliation(rec, membershipContext);
                  const fromTeam = affiliation ? teamRecords[affiliation.team] : undefined;
                  const rankDelta = playerRankDelta(
                    p.player,
                    affiliation?.tier ?? fromTeam?.current_tier,
                  );
                  const metadata = playerMetadata[p.player];
                  const displayRank = playerDisplayRank(
                    sortedPlayers,
                    i,
                    playerCol,
                    playerFloor,
                    playerOrderingVerified,
                  );
                  const identifiability = playerIdentifiabilityInfo(p);
                  const playerTrust =
                    playerFloor != null
                      ? trustInfo(p.sigma, playerFloor, p.n_maps)
                      : null;
                  const evidenceLabel =
                    identifiability.status === "identified" && playerTrust
                      ? `${identifiability.label} · ${playerTrust.label}`
                      : identifiability.label;
                  const evidenceTitle = `${identifiability.layman}${
                    playerTrust ? ` ${playerTrust.layman}` : ""
                  }`;
                  const adjusted = playerAdjustedRating(p, playerFloor);
                  return (
                    <tr
                      key={p.player}
                      className="row-click"
                      onClick={() => router.push(`/elo/player/${playerSlug(p.player)}`)}
                    >
                      <td className="font-mono text-[var(--ink-muted)]">{displayRank ?? "—"}</td>
                      {showPlayerRankDeltas ? (
                        <td className={`font-mono ${rankDeltaClass(rankDelta)}`} title="Change since the previous Sunday 00:00 UTC">
                          {rankDeltaLabel(rankDelta)}
                        </td>
                      ) : null}
                      <td className="font-medium">
                        <Link
                          href={`/elo/player/${playerSlug(p.player)}`}
                          className="row-link"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {metadata?.flag ? (
                            <span className="player-country" title={metadata.country || undefined} aria-label={metadata.country || undefined}>
                              {metadata.flag}
                            </span>
                          ) : null}
                          {p.player}
                        </Link>
                      </td>
                      <td>
                        {affiliation ? (
                          <Link
                            href={`/elo/team/${teamSlug(affiliation.team)}`}
                            className="row-link"
                            onClick={(e) => e.stopPropagation()}
                          >
                            {affiliation.team}
                          </Link>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td>
                        {formatAffiliation(affiliation?.tier, affiliation?.league)}
                        {affiliation ? <span className="block text-xs text-[var(--ink-faint)]">{affiliation.tournament}</span> : null}
                      </td>
                      <td className="num">
                        {adjusted != null ? adjusted.toFixed(1) : "—"}
                      </td>
                      <td className="num">{p.mu_total.toFixed(1)}</td>
                      <td className="num" title={evidenceTitle}>
                        {evidenceLabel}
                      </td>
                      <td className="num">{p.n_maps}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <ul className="elo-cards elo-mobile">
            {visiblePlayers.map((p, i) => {
              const rec = playerRecords[p.player];
              const affiliation = verifiedPlayerAffiliation(rec, membershipContext);
              const fromTeam = affiliation ? teamRecords[affiliation.team] : undefined;
              const rankDelta = playerRankDelta(
                p.player,
                affiliation?.tier ?? fromTeam?.current_tier,
              );
              const metadata = playerMetadata[p.player];
              const displayRank = playerDisplayRank(
                sortedPlayers,
                i,
                playerCol,
                playerFloor,
                playerOrderingVerified,
              );
              const identifiability = playerIdentifiabilityInfo(p);
              const playerTrust =
                playerFloor != null ? trustInfo(p.sigma, playerFloor, p.n_maps) : null;
              const evidenceLabel =
                identifiability.status === "identified" && playerTrust
                  ? `${identifiability.label} · ${playerTrust.label}`
                  : identifiability.label;
              const adjusted = playerAdjustedRating(p, playerFloor);
              return (
                <li key={p.player}>
                  <Link href={`/elo/player/${playerSlug(p.player)}`} className="elo-card">
                    <span className="elo-card-rank">
                      {displayRank == null ? "—" : `#${displayRank}`}
                      {showPlayerRankDeltas ? (
                        <> <span className={rankDeltaClass(rankDelta)}>{rankDeltaLabel(rankDelta)}</span></>
                      ) : null}
                    </span>
                    <span className="elo-card-title">
                      {metadata?.flag ? <span className="player-country" title={metadata.country || undefined}>{metadata.flag}</span> : null}
                      {p.player}
                    </span>
                    <span className="elo-card-meta">
                      {affiliation?.team ?? "Current team unverified"} · {formatAffiliation(affiliation?.tier, affiliation?.league)} · Evidence {evidenceLabel} · {p.n_maps} games
                    </span>
                    <span className="elo-card-rating">
                      {adjusted != null ? adjusted.toFixed(1) : "Unavailable"}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>

          {sortedPlayers.length === 0 && (
            <p className="empty-hint">
              No players match these chips
              {q.trim() ? ` and “${q.trim()}”` : ""}. Try a different scope or clear a filter.
            </p>
          )}
          {sortedPlayers.length > 20 && (
            <p className="empty-hint flex flex-wrap items-center gap-3">
              Showing {visiblePlayers.length} of {sortedPlayers.length}.
              <button type="button" className="btn-primary" onClick={() => setExpanded((x) => !x)}>
                {expanded ? "Collapse" : "Expand"}
              </button>
            </p>
          )}
        </>
      ) : (
        <>
          {playerPerformanceRows &&
          playerPerformanceMeta &&
          playerPerformanceValidation ? (
            <>
              <p className="method-note max-w-[72ch]">
                Frozen test: {playerPerformanceValidation.test_metrics.rows.toLocaleString()} player-map
                rows, RMSE {playerPerformanceValidation.test_metrics.rmse.toFixed(4)} versus{" "}
                {playerPerformanceValidation.test_metrics.zero_baseline_rmse.toFixed(4)} for the
                zero baseline. Player identity improves RMSE over the context-only model by{" "}
                {(100 * playerPerformanceValidation.player_incremental_test_rmse_lift).toFixed(2)}%;
                calendar-day bootstrap{" "}
                {(100 * playerPerformanceValidation.player_incremental_test_contrast.confidence_level).toFixed(0)}%
                interval{" "}
                {(100 * playerPerformanceValidation.player_incremental_test_contrast.ci_low).toFixed(2)}%–
                {(100 * playerPerformanceValidation.player_incremental_test_contrast.ci_high).toFixed(2)}%.
                This validates a narrow predictive signal, not causal skill.
              </p>
              <div
                className="table-scroll elo-desktop"
                tabIndex={0}
                aria-label={`${roleLabel(performanceRole)} 15-minute resource-performance table; scroll horizontally if needed`}
              >
                <table className="data-table">
                  <caption className="sr-only">
                    Role-relative 15-minute resource performance. Exact tied lower-bound
                    values share competition rank.
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Role rank</th>
                      <th scope="col">Player</th>
                      <th scope="col">Team at fit-through</th>
                      <th scope="col">League at fit-through</th>
                      <th
                        scope="col"
                        className="num"
                        title={playerPerformanceMeta.uncertainty.lower_bound}
                      >
                        Lower-bound score
                      </th>
                      <th
                        scope="col"
                        className="num"
                        title="Role-specific fitted coefficient on the robust standardized 15-minute resource target."
                      >
                        Fitted score
                      </th>
                      <th
                        scope="col"
                        className="num"
                        title={playerPerformanceMeta.uncertainty.interpretation}
                      >
                        Local uncertainty
                      </th>
                      <th scope="col" className="num">Fit maps</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visiblePerformance.map((row) => {
                      const metadata = playerMetadata[row.player_name];
                      return (
                        <tr key={`${row.player_id}:${row.role}`}>
                          <td className="font-mono text-[var(--ink-muted)]">{row.rank}</td>
                          <td className="font-medium">
                            {metadata?.flag ? (
                              <span
                                className="player-country"
                                title={metadata.country || undefined}
                                aria-label={metadata.country || undefined}
                              >
                                {metadata.flag}
                              </span>
                            ) : null}
                            {row.player_name}
                          </td>
                          <td>{row.last_team_key || "—"}</td>
                          <td>{row.last_observed_league || "—"}</td>
                          <td className="num">{row.lower_bound.toFixed(3)}</td>
                          <td className="num">{row.performance_mean.toFixed(3)}</td>
                          <td className="num">{row.performance_sd.toFixed(3)}</td>
                          <td className="num">{row.effective_sample_maps}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <ul className="elo-cards elo-mobile">
                {visiblePerformance.map((row) => (
                  <li key={`${row.player_id}:${row.role}`}>
                    <div className="elo-card">
                      <span className="elo-card-rank">#{row.rank}</span>
                      <span className="elo-card-title">{row.player_name}</span>
                      <span className="elo-card-meta">
                        {roleLabel(row.role)} · {row.last_team_key || "Fit-period team unavailable"} ·{" "}
                        {row.effective_sample_maps} fit maps · local uncertainty{" "}
                        {row.performance_sd.toFixed(3)}
                      </span>
                      <span className="elo-card-rating">{row.lower_bound.toFixed(3)}</span>
                    </div>
                  </li>
                ))}
              </ul>

              {sortedPerformance.length === 0 ? (
                <p className="empty-hint">
                  No {roleLabel(performanceRole)} player-role rows meet this search and
                  minimum-map threshold.
                </p>
              ) : null}
              {sortedPerformance.length > 20 ? (
                <p className="empty-hint flex flex-wrap items-center gap-3">
                  Showing {visiblePerformance.length} of {sortedPerformance.length}.
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => setExpanded((value) => !value)}
                  >
                    {expanded ? "Collapse" : "Expand"}
                  </button>
                </p>
              ) : null}
              <p className="method-note max-w-[72ch]">
                Not estimated: {playerPerformanceMeta.non_estimands.join(", ")}. Model{" "}
                <span className="font-mono">{playerPerformanceMeta.model_id}</span>; hash{" "}
                <span className="font-mono">{playerPerformanceMeta.model_hash.slice(0, 12)}</span>.
              </p>
            </>
          ) : null}
        </>
      )}
    </div>
  );
}
