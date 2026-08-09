"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type {
  PlayerMetadata,
  PlayerRating,
  PlayerRecord,
  PlayerWeeklyRanks,
  TeamRating,
  TeamRecord,
  TeamWeeklyRanks,
} from "@/lib/pack";
import {
  adjustedRating,
  formatWr,
  INTL_LEAGUES,
  INTERREGIONAL_LEAGUES,
  PLAYER_SIGMA_MIN,
  playerMatchesQuery,
  playerSlug,
  recordMatchesLeagues,
  REGION_LEAGUES,
  scopedTeamWr,
  softMu,
  TIER_FILTERS,
  TEAM_SIGMA_MIN,
  teamMatchesQuery,
  teamSlug,
} from "@/lib/pack";
import { evidenceFields, evidenceInfo, formatEvidenceCell } from "@/lib/evidence";
import styles from "./EloLadders.module.css";

type Props = {
  teams: TeamRating[];
  players: PlayerRating[];
  teamRecords: Record<string, TeamRecord>;
  teamWeeklyRanks: TeamWeeklyRanks;
  playerRecords: Record<string, PlayerRecord>;
  playerWeeklyRanks: PlayerWeeklyRanks;
  playerMetadata: Record<string, PlayerMetadata>;
  availableLeagues: string[];
};

type TeamCol = "team" | "league" | "soft" | "mu" | "meta" | "trust" | "wr";
type PlayerCol = "player" | "last_team" | "league" | "role" | "soft" | "mu" | "trust" | "games";
type Dir = "asc" | "desc";

const CHIP_ORDER = [...REGION_LEAGUES, ...INTERREGIONAL_LEAGUES, "INTL", ...INTL_LEAGUES];
const PLAYER_ROLES = [
  ["top", "Top"],
  ["jungle", "Jungle"],
  ["mid", "Mid"],
  ["bot", "Bot"],
  ["support", "Support"],
] as const;

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

function formatPlayerRole(role: string | null | undefined): string {
  return PLAYER_ROLES.find(([value]) => value === role)?.[1] ?? "—";
}

function rankDeltaLabel(delta: number | null | undefined): string {
  if (delta == null || delta === 0) return "—";
  return delta > 0 ? `↑${delta}` : `↓${Math.abs(delta)}`;
}

function rankDeltaClass(delta: number | null | undefined): string {
  if (delta == null || delta === 0) return styles.rankFlat;
  return delta > 0 ? styles.rankUp : styles.rankDown;
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
      className={align === "num" ? styles.numeric : undefined}
      scope="col"
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        className={`${styles.sortButton} ${active ? styles.sortActive : ""}`}
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

export function EloLadders({
  teams,
  players,
  teamRecords,
  teamWeeklyRanks,
  playerRecords,
  playerWeeklyRanks,
  playerMetadata,
  availableLeagues,
}: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [tab, setTab] = useState<"teams" | "players">(
    searchParams.get("tab") === "players" ? "players" : "teams",
  );
  const [q, setQ] = useState(searchParams.get("q") || "");
  const [leagues, setLeagues] = useState<string[]>(() => {
    const parsed = parseLeagues(searchParams.get("leagues"));
    return parsed.length ? parsed : ["TIER1"];
  });
  const [minGames, setMinGames] = useState(Math.max(5, Number(searchParams.get("min") || 20)));
  const [playerRole, setPlayerRole] = useState(searchParams.get("role") || "");
  const [expanded, setExpanded] = useState(false);
  const [teamCol, setTeamCol] = useState<TeamCol>("soft");
  const [teamDir, setTeamDir] = useState<Dir>("desc");
  const [playerCol, setPlayerCol] = useState<PlayerCol>("soft");
  const [playerDir, setPlayerDir] = useState<Dir>("desc");

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
    return shown.length > 1 ? [...new Set(shown)] : [...CHIP_ORDER];
  }, [availableLeagues]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (tab !== "teams") params.set("tab", tab);
    if (q.trim()) params.set("q", q.trim());
    if (leagues.length) params.set("leagues", leagues.join(","));
    if (tab === "players" && minGames !== 20) params.set("min", String(minGames));
    if (tab === "players" && playerRole) params.set("role", playerRole);
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }, [tab, q, leagues, minGames, playerRole, pathname, router]);

  const toggleLeague = (lg: string) => {
    setLeagues((prev) => (prev.includes(lg) ? prev.filter((x) => x !== lg) : [...prev, lg]));
    setExpanded(false);
  };

  const setTier = (tier: string | null) => {
    setLeagues((prev) => [
      ...prev.filter((scope) => !scope.startsWith("TIER")),
      ...(tier ? [tier] : []),
    ]);
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
          c === "player" || c === "last_team" || c === "league" || c === "role" ? "asc" : "desc",
        );
      }
    },
    [playerCol],
  );

  const sortedTeams = useMemo(() => {
    let list = teams.filter((t) => {
      if (!teamMatchesQuery(t.team, q)) return false;
      return recordMatchesLeagues(teamRecords[t.team], leagues);
    });
    const sign = teamDir === "asc" ? 1 : -1;
    list = [...list].sort((a, b) => {
      let cmp = 0;
      const ra = teamRecords[a.team];
      const rb = teamRecords[b.team];
      switch (teamCol) {
        case "team":
          cmp = a.team.localeCompare(b.team);
          break;
        case "league":
          cmp = (ra?.primary || "").localeCompare(rb?.primary || "");
          break;
        case "soft":
          cmp =
            adjustedRating(a, TEAM_SIGMA_MIN) - adjustedRating(b, TEAM_SIGMA_MIN);
          break;
        case "mu":
          cmp = a.mu_total - b.mu_total;
          break;
        case "meta":
          cmp = a.mu_meta - b.mu_meta;
          break;
        case "trust":
          cmp =
            Math.max(0, a.sigma - TEAM_SIGMA_MIN) - Math.max(0, b.sigma - TEAM_SIGMA_MIN);
          break;
        case "wr": {
          const wa = scopedTeamWr(ra, leagues) ?? -1;
          const wb = scopedTeamWr(rb, leagues) ?? -1;
          cmp = wa - wb;
          break;
        }
      }
      return sign * cmp;
    });
    return list;
  }, [teams, q, leagues, teamRecords, teamCol, teamDir]);

  const sortedPlayers = useMemo(() => {
    let list = players
      .filter((p) => (p.n_maps ?? 0) >= minGames)
      .filter((p) => {
        if (!playerRole) return true;
        const role = playerRecords[p.player]?.primary_role ?? playerRecords[p.player]?.roles?.[0];
        return role === playerRole;
      })
      .filter((p) => {
        const currentTeam = playerRecords[p.player]?.current_team ?? p.last_team;
        return playerMatchesQuery(p.player, currentTeam, q);
      })
      .filter((p) => {
        const rec = playerRecords[p.player];
        const currentTeam = rec?.current_team ?? p.last_team;
        const fromTeam = currentTeam ? teamRecords[currentTeam] : undefined;
        return rec
          ? recordMatchesLeagues(rec, leagues)
          : recordMatchesLeagues(fromTeam, leagues) || (!fromTeam && !leagues.length);
      });
    // if leagues selected and no player record match via team
    if (leagues.length) {
      list = list.filter((p) => {
        const rec = playerRecords[p.player];
        const currentTeam = rec?.current_team ?? p.last_team;
        const fromTeam = currentTeam ? teamRecords[currentTeam] : undefined;
        return rec ? recordMatchesLeagues(rec, leagues) : recordMatchesLeagues(fromTeam, leagues);
      });
    }
    const sign = playerDir === "asc" ? 1 : -1;
    list = [...list].sort((a, b) => {
      let cmp = 0;
      const ra = playerRecords[a.player];
      const rb = playerRecords[b.player];
      switch (playerCol) {
        case "player":
          cmp = a.player.localeCompare(b.player);
          break;
        case "last_team":
          cmp = (a.last_team || "").localeCompare(b.last_team || "");
          break;
        case "league":
          cmp = (ra?.primary || "").localeCompare(rb?.primary || "");
          break;
        case "role":
          cmp = formatPlayerRole(ra?.primary_role ?? ra?.roles?.[0]).localeCompare(
            formatPlayerRole(rb?.primary_role ?? rb?.roles?.[0]),
          );
          break;
        case "soft":
          cmp =
            softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN) -
            softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN);
          break;
        case "mu":
          cmp = a.mu_total - b.mu_total;
          break;
        case "trust":
          cmp =
            Math.max(0, a.sigma - PLAYER_SIGMA_MIN) -
            Math.max(0, b.sigma - PLAYER_SIGMA_MIN);
          break;
        case "games":
          cmp = (a.n_maps ?? 0) - (b.n_maps ?? 0);
          break;
      }
      return sign * cmp;
    });
    return list;
  }, [players, q, minGames, playerRole, leagues, playerRecords, teamRecords, playerCol, playerDir]);

  const visibleTeams = expanded ? sortedTeams : sortedTeams.slice(0, 20);
  const visiblePlayers = expanded ? sortedPlayers : sortedPlayers.slice(0, 20);
  const intlSet = new Set<string>(["INTL", ...INTL_LEAGUES]);
  const tierScopes = leagues.filter((scope) => scope.startsWith("TIER"));
  const refinementScopes = leagues.filter((scope) => !scope.startsWith("TIER"));
  const scopeSummary = leagues.length ? leagues.map(formatScope).join(", ") : "All tiers";
  const regionalChips = chips.filter((lg) =>
    (REGION_LEAGUES as readonly string[]).includes(lg),
  );
  const crossRegionChips = chips.filter((lg) =>
    INTERREGIONAL_LEAGUES.includes(lg as (typeof INTERREGIONAL_LEAGUES)[number]),
  );
  const internationalChips = chips.filter(
    (lg) => intlSet.has(lg) || (INTL_LEAGUES as readonly string[]).includes(lg),
  );
  const teamWinRateAvailable = sortedTeams.some(
    (team) => scopedTeamWr(teamRecords[team.team], leagues) != null,
  );
  const rankScopeFor = (tier: string | null | undefined): "all" | "tier1" | "tier2" | "tier3" => {
    const selectedTiers = leagues.filter((scope) => scope.startsWith("TIER")).map((scope) => scope.toLowerCase()) as Array<"tier1" | "tier2" | "tier3">;
    if (selectedTiers.length === 1) return selectedTiers[0];
    if (tier === "tier1" || tier === "tier2" || tier === "tier3") return tier;
    return "all";
  };
  const playerRankDelta = (player: string, tier: string | null | undefined): number | null | undefined => {
    const scope = rankScopeFor(tier);
    return playerWeeklyRanks.by_player[player]?.[scope]?.delta;
  };
  const teamRankDelta = (team: string): number | null | undefined =>
    teamWeeklyRanks.by_team[team]?.delta;
  const liveSummary =
    tab === "teams"
      ? `${sortedTeams.length} teams shown in ${scopeSummary}, sorted by ${teamCol === "soft" ? "adjusted rating" : teamCol}`
      : `${sortedPlayers.length} players shown in ${scopeSummary}, sorted by ${playerCol === "soft" ? "adjusted rating" : playerCol}`;

  return (
    <div className={styles.root}>
      <p className="sr-only" aria-live="polite">
        {liveSummary}
      </p>

      <section className={styles.controls} aria-label="Rating controls">
        <div className={styles.toolbar}>
          <div className={styles.tabs} aria-label="Rating type">
          <button
            type="button"
            className={`${styles.tab} ${tab === "teams" ? styles.tabActive : ""}`}
            aria-pressed={tab === "teams"}
            onClick={() => {
              setTab("teams");
              setExpanded(false);
            }}
          >
            Teams
          </button>
          <button
            type="button"
            className={`${styles.tab} ${tab === "players" ? styles.tabActive : ""}`}
            aria-pressed={tab === "players"}
            onClick={() => {
              setTab("players");
              setExpanded(false);
            }}
          >
            Players
          </button>
          </div>
          <label className={styles.search}>
          <span className="sr-only">Search</span>
          <input
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={tab === "teams" ? "Team or alias (G2, KC…)" : "Player or team"}
          />
        </label>
        {tab === "players" && (
          <div className={styles.playerFilters}>
            <label className={styles.minGames}>
              <span>Min games</span>
              <input
                type="number"
                min={5}
                value={minGames}
                onChange={(e) => setMinGames(Math.max(5, Number(e.target.value) || 5))}
              />
            </label>
            <label className={styles.minGames}>
              <span>Role</span>
              <select value={playerRole} onChange={(e) => setPlayerRole(e.target.value)}>
                <option value="">All roles</option>
                {PLAYER_ROLES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}
        </div>

        <div className={styles.scopeBar} role="group" aria-label="Competitive tier">
          <span className={styles.scopeLabel}>Level</span>
          <button
            type="button"
            className={`${styles.scopeButton} ${tierScopes.length === 0 ? styles.scopeButtonActive : ""}`}
            aria-pressed={tierScopes.length === 0}
            onClick={() => setTier(null)}
          >
            All
          </button>
          {TIER_FILTERS.map((tier) => (
            <button
              key={tier.value}
              type="button"
              className={`${styles.scopeButton} ${leagues.includes(tier.value) ? styles.scopeButtonActive : ""}`}
              aria-pressed={leagues.includes(tier.value)}
              onClick={() => setTier(tier.value)}
              title={tier.description}
            >
              {tier.label}
            </button>
          ))}
        </div>
        <details className={styles.refine}>
            <summary>
              Leagues and events
              {refinementScopes.length > 0 ? ` (${refinementScopes.length})` : ""}
            </summary>
            <div className={styles.refinementPanel}>
              <p>Selections within a row are alternatives. Rows combine.</p>
              <div className={styles.refinementRow}>
                <span>Regional leagues</span>
                <div>
                  {regionalChips.map((lg) => (
                    <button
                      key={lg}
                      type="button"
                      className={`${styles.scopeButton} ${leagues.includes(lg) ? styles.scopeButtonActive : ""}`}
                      aria-pressed={leagues.includes(lg)}
                      onClick={() => toggleLeague(lg)}
                    >
                      {lg}
                    </button>
                  ))}
                </div>
              </div>
              {crossRegionChips.length > 0 && (
                <div className={styles.refinementRow}>
                  <span>Cross-region</span>
                  <div>
                    {crossRegionChips.map((lg) => (
                      <button
                        key={lg}
                        type="button"
                        className={`${styles.scopeButton} ${leagues.includes(lg) ? styles.scopeButtonActive : ""}`}
                        aria-pressed={leagues.includes(lg)}
                        onClick={() => toggleLeague(lg)}
                      >
                        {lg}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div className={styles.refinementRow}>
                <span>International events</span>
                <div>
                  {internationalChips.map((lg) => (
                    <button
                      key={lg}
                      type="button"
                      className={`${styles.scopeButton} ${leagues.includes(lg) ? styles.scopeButtonActive : ""}`}
                      aria-pressed={leagues.includes(lg)}
                      onClick={() => toggleLeague(lg)}
                    >
                      {lg}
                    </button>
                  ))}
                </div>
              </div>
              <button
                type="button"
                className={styles.resetButton}
                onClick={() => {
                  setLeagues(["TIER1"]);
                  setExpanded(false);
                }}
              >
                Reset to Tier 1
              </button>
            </div>
        </details>

        <div className={styles.resultSummary}>
          <p>
            <strong>
              {tab === "teams" ? sortedTeams.length : sortedPlayers.length}{" "}
              {tab === "teams"
                ? sortedTeams.length === 1
                  ? "team"
                  : "teams"
                : sortedPlayers.length === 1
                  ? "player"
                  : "players"}
            </strong>
            <span>{scopeSummary}</span>
          </p>
          <p>
            {tab === "teams" && !teamWinRateAvailable
              ? "Filters change who appears. Scoped win rate is unavailable for this selection. Published ratings do not change."
              : tab === "teams"
                ? "Filters change who appears and scoped win rate. Published ratings do not change."
                : "Filters change who appears. Published ratings do not change."}
          </p>
        </div>
      </section>

      {tab === "teams" ? (
        <>
          <div className={styles.tableViewport} tabIndex={0} aria-label="Team ratings table; scroll horizontally if needed">
            <table className={styles.table}>
              <caption className="sr-only">
                Team ratings in the current pack. Default order is adjusted rating.
              </caption>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col" title="Rank movement since the previous Sunday 00:00 UTC">Δ</th>
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
                    label="Adjusted rating"
                    col="soft"
                    active={teamCol === "soft"}
                    dir={teamDir}
                    align="num"
                    title="One-sided 90% conservative rating bound; wider when the team lacks an international bridge. Default sort. See Method."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Raw rating"
                    col="mu"
                    active={teamCol === "mu"}
                    dir={teamDir}
                    align="num"
                    title="Full rating: regional plus international components."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="International component"
                    col="meta"
                    active={teamCol === "meta"}
                    dir={teamDir}
                    align="num"
                    title="International component from MSI, EWC, Worlds, or First Stand."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Evidence"
                    col="trust"
                    active={teamCol === "trust"}
                    dir={teamDir}
                    align="num"
                    title="Evidence state from the validated contract: interval width, precision, stability, freshness, and coverage. Settled requires all gates; anything missing fails closed."
                    onSort={onTeamSort}
                  />
                  <SortTh
                    label="Scoped win rate"
                    col="wr"
                    active={teamCol === "wr"}
                    dir={teamDir}
                    align="num"
                    title="Empirical win rate in the pack for the active scope."
                    onSort={onTeamSort}
                  />
                </tr>
              </thead>
              <tbody>
                {visibleTeams.map((t, i) => {
                  const rec = teamRecords[t.team];
                  const trust = evidenceInfo(evidenceFields(t as unknown as Record<string, unknown>), t.sigma, rec?.games);
                  const wr = scopedTeamWr(rec, leagues);
                  return (
                    <tr
                      key={t.team}
                    >
                      <td className={styles.rank}>{i + 1}</td>
                      <td className={`${styles.rankDelta} ${rankDeltaClass(teamRankDelta(t.team))}`} title="Change since the previous Sunday 00:00 UTC">
                        {rankDeltaLabel(teamRankDelta(t.team))}
                      </td>
                      <td className={styles.entity}>
                        <Link
                          href={`/elo/team/${teamSlug(t.team)}`}
                          className={styles.entityLink}
                          onClick={(e) => e.stopPropagation()}
                        >
                          {t.team}
                        </Link>
                      </td>
                      <td>{formatAffiliation(rec?.current_tier, rec?.current_league ?? rec?.primary)}</td>
                      <td className={styles.numeric}>{adjustedRating(t, TEAM_SIGMA_MIN).toFixed(1)}</td>
                      <td className={styles.numeric}>{t.mu_total.toFixed(1)}</td>
                      <td className={styles.numeric}>{t.mu_meta.toFixed(1)}</td>
                      <td className={`${styles.numeric} ${styles.evidence}`} title={trust.layman}>
                        <span>{formatEvidenceCell(trust)}</span>
                        {rec?.games ? <small>{rec.games} games</small> : null}
                      </td>
                      <td className={styles.numeric} title={wr == null ? "No games in this scope." : undefined}>
                        {formatWr(wr)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <ul className={styles.cards}>
            {visibleTeams.map((t, i) => {
              const rec = teamRecords[t.team];
              const trust = evidenceInfo(evidenceFields(t as unknown as Record<string, unknown>), t.sigma, rec?.games);
              const wr = scopedTeamWr(rec, leagues);
              return (
                <li key={t.team}>
                  <Link href={`/elo/team/${teamSlug(t.team)}`} className={styles.card}>
                    <span className={styles.cardRank}>{i + 1} <span className={rankDeltaClass(teamRankDelta(t.team))}>{rankDeltaLabel(teamRankDelta(t.team))}</span></span>
                    <span className={styles.cardTitle}>{t.team}</span>
                    <span className={styles.cardMeta}>
                      {formatAffiliation(rec?.current_tier, rec?.current_league ?? rec?.primary)}
                      {" · "}
                      {formatEvidenceCell(trust)} evidence
                      {rec?.games ? `, ${rec.games} games` : ""}
                      {wr != null ? ` · ${formatWr(wr)} scoped win rate` : ""}
                    </span>
                    <span className={styles.cardRating}>
                      {adjustedRating(t, TEAM_SIGMA_MIN).toFixed(1)}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>

          {sortedTeams.length === 0 && (
            <p className={styles.empty}>
              No teams match this scope
              {q.trim() ? ` and “${q.trim()}”` : ""}. Change the scope or search.
            </p>
          )}
          {sortedTeams.length > 20 && (
            <p className={styles.more}>
              Showing {visibleTeams.length} of {sortedTeams.length}.
              <button type="button" onClick={() => setExpanded((x) => !x)}>
                {expanded ? "Show 20" : "Show all"}
              </button>
            </p>
          )}
        </>
      ) : (
        <>
          <div className={styles.tableViewport} tabIndex={0} aria-label="Player ratings table; scroll horizontally if needed">
            <table className={styles.table}>
              <caption className="sr-only">
                Player ratings in the current pack. Default order is adjusted rating.
              </caption>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col" title="Rank movement since the previous Sunday 00:00 UTC">Δ</th>
                  <SortTh label="Player" col="player" active={playerCol === "player"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="Team" col="last_team" active={playerCol === "last_team"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="League" col="league" active={playerCol === "league"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh label="Role" col="role" active={playerCol === "role"} dir={playerDir} onSort={onPlayerSort} />
                  <SortTh
                    label="Adjusted rating"
                    col="soft"
                    active={playerCol === "soft"}
                    dir={playerDir}
                    align="num"
                    title="Raw rating with a soft penalty when evidence is still thin."
                    onSort={onPlayerSort}
                  />
                  <SortTh label="Raw rating" col="mu" active={playerCol === "mu"} dir={playerDir} align="num" onSort={onPlayerSort} />
                  <SortTh
                    label="Evidence"
                    col="trust"
                    active={playerCol === "trust"}
                    dir={playerDir}
                    align="num"
                    title="Evidence state from the validated contract; a low sigma alone never reads as Settled."
                    onSort={onPlayerSort}
                  />
                  <SortTh label="Games" col="games" active={playerCol === "games"} dir={playerDir} align="num" onSort={onPlayerSort} />
                </tr>
              </thead>
              <tbody>
                {visiblePlayers.map((p, i) => {
                  const rec = playerRecords[p.player];
                  const trust = evidenceInfo(evidenceFields(p as unknown as Record<string, unknown>), p.sigma, p.n_maps);
                  const currentTeam = rec?.current_team ?? p.last_team;
                  const fromTeam = currentTeam ? teamRecords[currentTeam] : undefined;
                  const league = rec?.current_league ?? rec?.primary ?? fromTeam?.current_league ?? fromTeam?.primary;
                  const rankDelta = playerRankDelta(p.player, rec?.current_tier ?? fromTeam?.current_tier);
                  const metadata = playerMetadata[p.player];
                  return (
                    <tr
                      key={p.player}
                    >
                      <td className={styles.rank}>{i + 1}</td>
                      <td className={`${styles.rankDelta} ${rankDeltaClass(rankDelta)}`} title="Change since the previous Sunday 00:00 UTC">
                        {rankDeltaLabel(rankDelta)}
                      </td>
                      <td className={styles.entity}>
                        <Link
                          href={`/elo/player/${playerSlug(p.player)}`}
                          className={styles.entityLink}
                          onClick={(e) => e.stopPropagation()}
                        >
                          {metadata?.flag ? (
                            <span className={styles.flag} title={metadata.country || undefined} aria-label={metadata.country || undefined}>
                              {metadata.flag}
                            </span>
                          ) : null}
                          {p.player}
                        </Link>
                      </td>
                      <td>
                        {currentTeam ? (
                          <Link
                            href={`/elo/team/${teamSlug(currentTeam)}`}
                            className={styles.entityLink}
                            onClick={(e) => e.stopPropagation()}
                          >
                            {currentTeam}
                          </Link>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td>{formatAffiliation(rec?.current_tier ?? fromTeam?.current_tier, league)}</td>
                      <td>{formatPlayerRole(rec?.primary_role ?? rec?.roles?.[0])}</td>
                      <td className={styles.numeric}>
                        {softMu(p.mu_total, p.sigma, PLAYER_SIGMA_MIN).toFixed(1)}
                      </td>
                      <td className={styles.numeric}>{p.mu_total.toFixed(1)}</td>
                      <td className={styles.numeric} title={trust.layman}>
                        {formatEvidenceCell(trust)}
                      </td>
                      <td className={styles.numeric}>{p.n_maps}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <ul className={styles.cards}>
            {visiblePlayers.map((p, i) => {
              const rec = playerRecords[p.player];
              const currentTeam = rec?.current_team ?? p.last_team;
              const fromTeam = currentTeam ? teamRecords[currentTeam] : undefined;
              const trust = evidenceInfo(evidenceFields(p as unknown as Record<string, unknown>), p.sigma, p.n_maps);
              const rankDelta = playerRankDelta(p.player, rec?.current_tier ?? fromTeam?.current_tier);
              const metadata = playerMetadata[p.player];
              return (
                <li key={p.player}>
                  <Link href={`/elo/player/${playerSlug(p.player)}`} className={styles.card}>
                    <span className={styles.cardRank}>{i + 1} <span className={rankDeltaClass(rankDelta)}>{rankDeltaLabel(rankDelta)}</span></span>
                    <span className={styles.cardTitle}>
                      {metadata?.flag ? <span className={styles.flag} title={metadata.country || undefined}>{metadata.flag}</span> : null}
                      {p.player}
                    </span>
                    <span className={styles.cardMeta}>
                      {currentTeam ?? "—"} · {formatAffiliation(rec?.current_tier ?? fromTeam?.current_tier, rec?.current_league ?? rec?.primary ?? fromTeam?.primary)} · {formatPlayerRole(rec?.primary_role ?? rec?.roles?.[0])} · {formatEvidenceCell(trust)} evidence · {p.n_maps} games
                    </span>
                    <span className={styles.cardRating}>
                      {softMu(p.mu_total, p.sigma, PLAYER_SIGMA_MIN).toFixed(1)}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>

          {sortedPlayers.length === 0 && (
            <p className={styles.empty}>
              No players match this scope
              {q.trim() ? ` and “${q.trim()}”` : ""}. Change the scope or search.
            </p>
          )}
          {sortedPlayers.length > 20 && (
            <p className={styles.more}>
              Showing {visiblePlayers.length} of {sortedPlayers.length}.
              <button type="button" onClick={() => setExpanded((x) => !x)}>
                {expanded ? "Show 20" : "Show all"}
              </button>
            </p>
          )}
        </>
      )}
    </div>
  );
}
