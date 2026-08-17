"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { executeTool, routeQuestion, type RouteResult, type ToolCall } from "@/lib/supportChat";
import { playerSlug, teamSlug } from "@/lib/pack";
import { DataBars, type DataBarRow } from "./DataBars";
import styles from "./SupportChat.module.css";

type Message = {
  role: "user" | "assistant";
  text?: string;
  error?: string;
  result?: unknown;
  call?: ToolCall;
};

type LeaderboardRow = {
  name: string;
  rating: number | null;
  role: string | null;
  team: string | null;
  league: string | null;
  tier: string | null;
  games: number;
  wins: number | null;
  win_rate: number | null;
  grade_a_games: number;
  grade_games: number;
  recent_form: number | null;
  draft_edge?: number | null;
  positive_edge_rate?: number | null;
  best_available_rate?: number | null;
};

type PlayerQueryRow = LeaderboardRow & {
  champion: string | null;
  champion_games: number | null;
  champion_wins: number | null;
  champion_win_rate: number | null;
  champion_score: number | null;
};

type PlayerQueryResult = {
  kind: "player_query";
  answer: {
    headline: string;
    basis: string;
    caveat: string | null;
  };
  rows: PlayerQueryRow[];
};

type TeamDraftRow = {
  team: string;
  average_edge: number;
  games: number;
  best_edge: number;
  worst_edge: number;
  positive_edge_rate: number | null;
};

type TeamDraftQueryResult = {
  kind: "team_draft_query" | "team_draft_comparison";
  answer: {
    headline: string;
    basis: string;
    caveat: string;
  };
  rows: TeamDraftRow[];
};

type ChampionQueryRow = {
  champion: string;
  role: string | null;
  tier_bucket: string | null;
  rank: number | null;
  patch: string | null;
  games: number | null;
  wins: number | null;
  win_rate: number | null;
  players: number | null;
};

type ChampionQueryResult = {
  kind: "champion_query";
  metric: "tier" | "win_rate" | "games";
  answer: {
    headline: string;
    basis: string;
    caveat: string;
  };
  rows: ChampionQueryRow[];
};

type TierRow = {
  champion: string;
  role: string;
  tier_bucket: string;
  rank: number;
  played_maps: number;
};

type MatchRow = {
  date: string;
  league: string;
  blue_team: string;
  red_team: string;
  blue_win: number;
  game_id: string;
};

type ScheduleRow = {
  start_utc: string;
  team1: string;
  team2: string;
  best_of?: number;
  series_id?: string;
};

function present(value: string | number | null | undefined): string | number {
  return value == null || value === "" ? "—" : value;
}

function rating(value: number | null | undefined): string | number {
  return value == null || !Number.isFinite(value) ? "—" : Math.round(value);
}

function percentage(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : `${Math.round(value * 100)}%`;
}

function draftEdge(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function roleName(value: string | null | undefined): string {
  const roles: Record<string, string> = {
    bot: "Bot",
    jng: "Jungle",
    jungle: "Jungle",
    mid: "Mid",
    sup: "Support",
    support: "Support",
    top: "Top",
  };
  return value ? (roles[value.toLowerCase()] ?? value) : "—";
}

function tierName(value: string | null | undefined): string {
  const match = /^tier\s*(\d+)$/i.exec(value ?? "");
  return match ? `Tier ${match[1]}` : String(present(value));
}

function renderValue(value: unknown, depth = 0): string {
  if (value == null) return "—";
  if (typeof value === "string") return value || "—";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map((entry) => renderValue(entry, depth + 1)).filter(Boolean).join(" · ");
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry != null && entry !== "")
      .map(([key, entry]) => {
        if (typeof entry === "object") return `${key}: ${renderValue(entry, depth + 1)}`;
        return `${key}: ${String(entry)}`;
      });
    return entries.join(" · ");
  }
  return String(value);
}

function table(children: React.ReactNode): React.ReactNode {
  return <div className={styles.tableScroll}>{children}</div>;
}

function leaderboardTable(rows: LeaderboardRow[], category: string): React.ReactNode {
  if (category === "teams_draft" || category === "players_draft") {
    const teamsDraft = category === "teams_draft";
    const chartRows: DataBarRow[] = rows.flatMap((row) => {
      const value = teamsDraft ? row.draft_edge : row.best_available_rate;
      return value == null ? [] : [{
      id: row.name,
      label: row.name,
      href: teamsDraft ? `/elo/team/${teamSlug(row.name)}` : `/elo/player/${playerSlug(row.name)}`,
      value,
      valueLabel: teamsDraft ? draftEdge(value) : percentage(value),
      detail: `${row.games} ${teamsDraft ? "games" : "picks"}${row.role ? ` · ${roleName(row.role)}` : ""}`,
      tone: teamsDraft ? (value >= 0 ? "positive" : "negative") : "positive",
    }];
    });
    return (
      <>
        <DataBars
          className={styles.chatChart}
          title={teamsDraft ? "Team draft edge" : "Player best-available rate"}
          description={teamsDraft ? "Descriptive edge in model units" : "Published pick evidence"}
          rows={chartRows}
          domain={teamsDraft ? { min: Math.min(0, ...chartRows.map((row) => row.value)), max: Math.max(0, ...chartRows.map((row) => row.value)) } : { min: 0, max: 1 }}
          baseline={teamsDraft ? 0 : undefined}
          baselineLabel={teamsDraft ? "0 even" : undefined}
          axisLeft={teamsDraft ? "lower" : "0%"}
          axisRight={teamsDraft ? "higher" : "100%"}
        />
        {table(
          <table className={styles.resultTable}>
            <thead><tr>{teamsDraft ? <th>Team</th> : <th>Player</th>}{teamsDraft ? null : <th>Role</th>}<th className={styles.numeric}>{teamsDraft ? "Games" : "Picks"}</th><th className={styles.numeric}>{teamsDraft ? "Draft edge" : "Best-available rate"}</th></tr></thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name}>
                  <td><a className="row-link" href={teamsDraft ? `/elo/team/${teamSlug(row.name)}` : `/elo/player/${playerSlug(row.name)}`}>{present(row.name)}</a></td>
                  {teamsDraft ? null : <td>{roleName(row.role)}</td>}
                  <td className={styles.numeric}>{present(row.games)}</td>
                  <td className={styles.numeric}>{teamsDraft ? draftEdge(row.draft_edge) : percentage(row.best_available_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>,
        )}
      </>
    );
  }

  if (category === "teams") {
    const chartRows: DataBarRow[] = rows
      .filter((row): row is LeaderboardRow & { rating: number } => row.rating != null)
      .map((row) => ({
        id: row.name,
        label: row.name,
        href: `/elo/team/${teamSlug(row.name)}`,
        value: row.rating,
        valueLabel: rating(row.rating).toString(),
        detail: `${row.games} games · ${tierName(row.tier)}`,
        tone: "neutral" as const,
      }));
    return (
      <>
        {chartRows.length ? <DataBars className={styles.chatChart} title="Team adjusted rating" description="Uncertainty-adjusted published rating" rows={chartRows} axisLeft="lower" axisRight="higher" /> : null}
        {table(
          <table className={styles.resultTable}>
            <thead><tr><th>Team</th><th>Level</th><th className={styles.numeric}>Rating</th><th>League</th><th className={styles.numeric}>Wins</th><th className={styles.numeric}>Games</th><th className={styles.numeric}>WR</th></tr></thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name}>
                  <td><a className="row-link" href={`/elo/team/${teamSlug(row.name)}`}>{present(row.name)}</a></td>
                  <td>{tierName(row.tier)}</td>
                  <td className={styles.numeric}>{rating(row.rating)}</td>
                  <td>{present(row.league)}</td>
                  <td className={styles.numeric}>{present(row.wins)}</td>
                  <td className={styles.numeric}>{present(row.games)}</td>
                  <td className={styles.numeric}>{percentage(row.win_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>,
        )}
      </>
    );
  }

  const chartRows: DataBarRow[] = rows
    .filter((row): row is LeaderboardRow & { rating: number } => row.rating != null)
    .map((row) => ({
      id: row.name,
      label: row.name,
      href: `/elo/player/${playerSlug(row.name)}`,
      value: row.rating,
      valueLabel: rating(row.rating).toString(),
      detail: `${roleName(row.role)} · ${row.games} games`,
      tone: "neutral" as const,
    }));
  return (
    <>
      {chartRows.length ? <DataBars className={styles.chatChart} title="Player adjusted rating" description="Uncertainty-adjusted published rating" rows={chartRows} axisLeft="lower" axisRight="higher" /> : null}
      {table(
        <table className={styles.resultTable}>
          <thead><tr><th>Player</th><th>Role</th><th>Level</th><th className={styles.numeric}>Rating</th><th className={styles.numeric}>A grades</th><th className={styles.numeric}>Games</th><th className={styles.numeric}>WR</th></tr></thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <td><a className="row-link" href={`/elo/player/${playerSlug(row.name)}`}>{present(row.name)}</a></td>
                <td>{roleName(row.role)}</td>
                <td>{tierName(row.tier)}</td>
                <td className={styles.numeric}>{rating(row.rating)}</td>
                <td className={styles.numeric}>{present(row.grade_a_games)}</td>
                <td className={styles.numeric}>{present(row.games)}</td>
                <td className={styles.numeric}>{percentage(row.win_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>,
      )}
    </>
  );
}

function resultTable(result: unknown, call: ToolCall): React.ReactNode {
  const data = result as Record<string, unknown> | null;

  if (
    data?.authority === "unavailable"
    && (call.tool === "query_drafts" || (
      call.tool === "leaderboards"
      && (call.args.category === "teams_draft" || call.args.category === "players_draft")
    ))
  ) {
    return (
      <div className={styles.queryResult} data-testid="draft-unavailable-result">
        <p className={styles.queryHeadline}>Draft Score is unavailable for this release.</p>
        <p className={styles.queryBasis}>An independent, release-bound promotion receipt is required before Scryglass can publish draft results.</p>
        <p className={styles.queryCaveat}><a className="row-link" href="/methodology#composition-signal">View Methodology</a></p>
      </div>
    );
  }

  if (call.tool === "query_champions" && data?.kind === "champion_query" && Array.isArray(data.rows)) {
    const query = data as unknown as ChampionQueryResult;
    return (
      <div className={styles.queryResult} data-testid="champion-query-result">
        <p className={styles.queryHeadline} data-testid="champion-query-headline">{query.answer.headline}</p>
        <p className={styles.queryBasis}>{query.answer.basis}</p>
        <p className={styles.queryCaveat}>{query.answer.caveat}</p>
        {query.rows.length ? query.metric === "tier" ? table(
          <table className={styles.resultTable}>
            <thead><tr><th>Champion</th><th>Role</th><th>Tier</th><th className={styles.numeric}>Rank</th><th className={styles.numeric}>Maps</th></tr></thead>
            <tbody>
              {query.rows.map((row) => (
                <tr key={`${row.champion}-${row.role ?? ""}-${row.rank ?? ""}`}>
                  <td>{row.champion}</td>
                  <td>{roleName(row.role)}</td>
                  <td>{present(row.tier_bucket)}</td>
                  <td className={styles.numeric}>{present(row.rank)}</td>
                  <td className={styles.numeric}>{present(row.games)}</td>
                </tr>
              ))}
            </tbody>
          </table>,
        ) : table(
          <table className={styles.resultTable}>
            <thead><tr><th>Champion</th><th className={styles.numeric}>Games</th><th className={styles.numeric}>Wins</th><th className={styles.numeric}>WR</th><th className={styles.numeric}>Players</th></tr></thead>
            <tbody>
              {query.rows.map((row) => (
                <tr key={row.champion}>
                  <td>{row.champion}</td>
                  <td className={styles.numeric}>{present(row.games)}</td>
                  <td className={styles.numeric}>{present(row.wins)}</td>
                  <td className={styles.numeric}>{percentage(row.win_rate)}</td>
                  <td className={styles.numeric}>{present(row.players)}</td>
                </tr>
              ))}
            </tbody>
          </table>,
        ) : null}
      </div>
    );
  }

  if (call.tool === "query_drafts" && (data?.kind === "team_draft_query" || data?.kind === "team_draft_comparison") && Array.isArray(data.rows)) {
    const query = data as unknown as TeamDraftQueryResult;
    const chartRows: DataBarRow[] = query.rows.map((row) => ({
      id: row.team,
      label: row.team,
      href: `/elo/team/${teamSlug(row.team)}`,
      value: row.average_edge,
      valueLabel: draftEdge(row.average_edge),
      detail: `${row.games} complete drafts`,
      tone: row.average_edge >= 0 ? "positive" : "negative",
    }));
    return (
      <div className={styles.queryResult} data-testid={query.kind === "team_draft_comparison" ? "team-draft-comparison-result" : "team-draft-query-result"}>
        <p className={styles.queryHeadline} data-testid="team-draft-query-headline">{query.answer.headline}</p>
        <p className={styles.queryBasis}>{query.answer.basis}</p>
        <p className={styles.queryCaveat}>{query.answer.caveat}</p>
        {query.rows.length ? <DataBars className={styles.chatChart} title="Draft Score" description="Ten-pick composition score in model units" rows={chartRows} domain={{ min: Math.min(0, ...chartRows.map((row) => row.value)), max: Math.max(0, ...chartRows.map((row) => row.value)) }} baseline={0} baselineLabel="0 even" axisLeft="lower" axisRight="higher" /> : null}
        {query.rows.length ? table(
          <table className={styles.resultTable}>
            <thead><tr><th>Team</th><th className={styles.numeric}>Draft edge</th><th className={styles.numeric}>Complete drafts</th></tr></thead>
            <tbody>
              {query.rows.map((row) => (
                <tr key={row.team}>
                  <td><a className="row-link" href={`/elo/team/${teamSlug(row.team)}`}>{row.team}</a></td>
                  <td className={styles.numeric}>{draftEdge(row.average_edge)}</td>
                  <td className={styles.numeric}>{row.games}</td>
                </tr>
              ))}
            </tbody>
          </table>,
        ) : null}
      </div>
    );
  }

  if (call.tool === "query_players" && data?.kind === "player_query" && Array.isArray(data.rows)) {
    const query = data as unknown as PlayerQueryResult;
    const hasChampion = query.rows.some((row) => Boolean(row.champion));
    const singlePlayerChampion = hasChampion
      && query.rows.length > 0
      && query.rows.every((row) => row.name === query.rows[0].name);
    return (
      <div className={styles.queryResult} data-testid="player-query-result">
        <p className={styles.queryHeadline} data-testid="player-query-headline">{query.answer.headline}</p>
        <p className={styles.queryBasis} data-testid="player-query-basis">{query.answer.basis}</p>
        {query.answer.caveat ? <p className={styles.queryCaveat} data-testid="player-query-caveat">{query.answer.caveat}</p> : null}
        {query.rows.length ? table(
          <table className={styles.resultTable}>
            <thead>
              {singlePlayerChampion ? (
                <tr><th>Champion</th><th className={styles.numeric}>Games</th><th className={styles.numeric}>Wins</th><th className={styles.numeric}>WR</th></tr>
              ) : hasChampion ? (
                <tr><th>Player</th><th>Champion</th><th>Role</th><th>Level</th><th className={styles.numeric}>Champion games</th><th className={styles.numeric}>Champion WR</th><th className={styles.numeric}>Rating</th></tr>
              ) : (
                <tr><th>Player</th><th>Role</th><th>Team</th><th>League</th><th>Level</th><th className={styles.numeric}>Rating</th><th className={styles.numeric}>Games</th><th className={styles.numeric}>WR</th></tr>
              )}
            </thead>
            <tbody>
              {query.rows.map((row) => (
                <tr key={`${row.name}-${row.champion ?? "player"}`}>
                  {singlePlayerChampion ? (
                    <>
                      <td>{present(row.champion)}</td>
                      <td className={styles.numeric}>{present(row.champion_games)}</td>
                      <td className={styles.numeric}>{present(row.champion_wins)}</td>
                      <td className={styles.numeric}>{percentage(row.champion_win_rate)}</td>
                    </>
                  ) : hasChampion ? (
                    <>
                      <td><a className="row-link" href={`/elo/player/${playerSlug(row.name)}`}>{row.name}</a></td>
                      <td>{present(row.champion)}</td>
                      <td>{roleName(row.role)}</td>
                      <td>{tierName(row.tier)}</td>
                      <td className={styles.numeric}>{present(row.champion_games)}</td>
                      <td className={styles.numeric}>{percentage(row.champion_win_rate)}</td>
                      <td className={styles.numeric}>{rating(row.rating)}</td>
                    </>
                  ) : (
                    <>
                      <td><a className="row-link" href={`/elo/player/${playerSlug(row.name)}`}>{row.name}</a></td>
                      <td>{roleName(row.role)}</td>
                      <td>{row.team ? <a className="row-link" href={`/elo/team/${teamSlug(row.team)}`}>{row.team}</a> : "—"}</td>
                      <td>{present(row.league)}</td>
                      <td>{tierName(row.tier)}</td>
                      <td className={styles.numeric}>{rating(row.rating)}</td>
                      <td className={styles.numeric}>{present(row.games)}</td>
                      <td className={styles.numeric}>{percentage(row.win_rate)}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>,
        ) : null}
      </div>
    );
  }

  if (call.tool === "compare_players" && Array.isArray(data?.players)) {
    const players = data.players as LeaderboardRow[];
    const better = typeof data.better === "string" ? data.better : null;
    const difference = typeof data.difference === "number" ? Math.round(data.difference) : null;
    const chartRows: DataBarRow[] = players
      .filter((player): player is LeaderboardRow & { rating: number } => player.rating != null)
      .map((player) => ({
        id: player.name,
        label: player.name,
        href: `/elo/player/${playerSlug(player.name)}`,
        value: player.rating,
        valueLabel: rating(player.rating).toString(),
        detail: `${roleName(player.role)} · ${player.games} games`,
        tone: "neutral" as const,
      }));
    return (
      <div className={styles.comparison}>
        <p className={styles.comparisonAnswer}>
        {better && difference != null
            ? <><strong>{better}</strong> has the higher rating by {difference} {difference === 1 ? "point" : "points"}.</>
            : "The published ratings do not support a winner."}
        </p>
        {chartRows.length ? <DataBars className={styles.chatChart} title="Rating comparison" description="Adjusted published rating" rows={chartRows} axisLeft="lower" axisRight="higher" /> : null}
        {table(
          <table className={styles.resultTable}>
            <thead><tr><th>Player</th><th>Role</th><th>Level</th><th className={styles.numeric}>Rating</th><th className={styles.numeric}>Games</th><th className={styles.numeric}>WR</th></tr></thead>
            <tbody>
              {players.map((player) => (
                <tr key={player.name}>
                  <td><a className="row-link" href={`/elo/player/${playerSlug(player.name)}`}>{player.name}</a></td>
                  <td>{roleName(player.role)}</td>
                  <td>{tierName(player.tier)}</td>
                  <td className={styles.numeric}>{rating(player.rating)}</td>
                  <td className={styles.numeric}>{present(player.games)}</td>
                  <td className={styles.numeric}>{percentage(player.win_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>,
        )}
      </div>
    );
  }

  if (call.tool === "player" && data?.name) {
    const profile = data as unknown as LeaderboardRow;
    return (
      <div className={styles.resultLines}>
        <p className={styles.resultTitle}><a className="row-link" href={`/elo/player/${playerSlug(profile.name)}`}>{profile.name}</a></p>
        <dl className={styles.statGrid}>
          <div><dt>Rating</dt><dd>{rating(profile.rating)}</dd></div>
          <div><dt>Role</dt><dd>{roleName(profile.role)}</dd></div>
          <div><dt>Team</dt><dd>{profile.team ? <a className="row-link" href={`/elo/team/${teamSlug(profile.team)}`}>{profile.team}</a> : "—"}</dd></div>
          <div><dt>League</dt><dd>{present(profile.league)}</dd></div>
          <div><dt>A-grade games</dt><dd>{present(profile.grade_a_games)}</dd></div>
          <div><dt>Win rate</dt><dd>{percentage(profile.win_rate)}</dd></div>
        </dl>
      </div>
    );
  }

  if (call.tool === "team" && data?.team) {
    const profile = data as unknown as {
      team: string;
      rating: number | null;
      league: string | null;
      games: number;
      wins: number | null;
      win_rate: number | null;
      recent?: Array<{ date: string; opponent: string; side: string; won: boolean; game_id: string }>;
    };
    return (
      <div className={styles.resultLines}>
        <p className={styles.resultTitle}><a className="row-link" href={`/elo/team/${teamSlug(profile.team)}`}>{profile.team}</a></p>
        <dl className={styles.statGrid}>
          <div><dt>Rating</dt><dd>{rating(profile.rating)}</dd></div>
          <div><dt>League</dt><dd>{present(profile.league)}</dd></div>
          <div><dt>Wins</dt><dd>{present(profile.wins)}</dd></div>
          <div><dt>Games</dt><dd>{present(profile.games)}</dd></div>
          <div><dt>Win rate</dt><dd>{percentage(profile.win_rate)}</dd></div>
        </dl>
        {profile.recent?.length ? (
          <div className={styles.recentBlock}>
            <strong>Recent matches</strong>
            {table(
              <table className={styles.resultTable}>
                <thead><tr><th>Date</th><th>Opponent</th><th>Side</th><th>Result</th></tr></thead>
                <tbody>
                  {profile.recent.map((match) => (
                    <tr key={match.game_id}>
                      <td><a className="row-link" href={`/matches/${encodeURIComponent(match.game_id)}`}>{present(match.date.slice(0, 10))}</a></td>
                      <td><a className="row-link" href={`/elo/team/${teamSlug(match.opponent)}`}>{match.opponent}</a></td>
                      <td>{match.side}</td>
                      <td>{match.won ? "Win" : "Loss"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>,
            )}
          </div>
        ) : null}
      </div>
    );
  }

  if (call.tool === "leaderboards" && Array.isArray(data?.rows)) {
    return leaderboardTable(data.rows as LeaderboardRow[], String(data.category ?? call.args.category ?? "rating"));
  }

  if (call.tool === "tier" && Array.isArray(data?.rows)) {
    return table(
      <table className={styles.resultTable}>
        <thead><tr><th>Champion</th><th>Role</th><th>Tier</th><th className={styles.numeric}>Rank</th><th className={styles.numeric}>Maps</th></tr></thead>
        <tbody>
          {(data.rows as TierRow[]).map((row, index) => (
            <tr key={`${row.champion}-${row.role}-${index}`}>
              <td>{present(row.champion)}</td>
              <td>{roleName(row.role)}</td>
              <td>{present(row.tier_bucket)}</td>
              <td className={styles.numeric}>{present(row.rank)}</td>
              <td className={styles.numeric}>{present(row.played_maps)}</td>
            </tr>
          ))}
        </tbody>
      </table>,
    );
  }

  if (Array.isArray(data?.matches)) {
    return table(
      <table className={styles.resultTable}>
        <thead><tr><th>Date</th><th>League</th><th>Blue</th><th>Red</th><th>Winner</th></tr></thead>
        <tbody>
          {(data.matches as MatchRow[]).map((match) => {
            const winner = match.blue_win === 1 ? match.blue_team : match.red_team;
            return (
              <tr key={match.game_id}>
                <td><a className="row-link" href={`/matches/${encodeURIComponent(match.game_id)}`}>{present(match.date?.slice(0, 10))}</a></td>
                <td>{present(match.league)}</td>
                <td><a className="row-link" href={`/elo/team/${teamSlug(match.blue_team)}`}>{present(match.blue_team)}</a></td>
                <td><a className="row-link" href={`/elo/team/${teamSlug(match.red_team)}`}>{present(match.red_team)}</a></td>
                <td>{present(winner)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>,
    );
  }

  if (Array.isArray(data?.upcoming)) {
    return table(
      <table className={styles.resultTable}>
        <thead><tr><th>Start (UTC)</th><th>Team 1</th><th>Team 2</th><th className={styles.numeric}>BO</th></tr></thead>
        <tbody>
          {(data.upcoming as ScheduleRow[]).slice(0, 10).map((row, index) => (
            <tr key={row.series_id ?? index}>
              <td>{present(row.start_utc)}</td>
              <td><a className="row-link" href={`/elo/team/${teamSlug(row.team1)}`}>{present(row.team1)}</a></td>
              <td><a className="row-link" href={`/elo/team/${teamSlug(row.team2)}`}>{present(row.team2)}</a></td>
              <td className={styles.numeric}>{present(row.best_of)}</td>
            </tr>
          ))}
        </tbody>
      </table>,
    );
  }

  return <p className={styles.resultText}>{renderValue(result)}</p>;
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5.25 5.75h13.5v9.5H10l-4.75 3v-12.5Z" />
    </svg>
  );
}

function ExpandIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      {expanded ? (
        <path d="M9 4v5H4M15 20v-5h5M4 9l5-5M20 15l-5 5" />
      ) : (
        <path d="M9 4H4v5M15 20h5v-5M4 9l5-5M20 15l-5 5" />
      )}
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m6.5 6.5 11 11m0-11-11 11" />
    </svg>
  );
}

function ChatThinking() {
  return (
    <div className={styles.thinking} role="status" aria-label="Scryglass is reading the published data">
      <span>Reading published data</span>
      <span className={styles.thinkingDots} aria-hidden="true"><i /><i /><i /></span>
    </div>
  );
}

export default function SupportChat({ floating = false }: { floating?: boolean }) {
  const pathname = usePathname();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(!floating);
  const [expanded, setExpanded] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const thread = threadRef.current;
    if (!thread) return;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    thread.scrollTo({ top: thread.scrollHeight, behavior: reducedMotion ? "auto" : "smooth" });
  }, [messages, busy]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setMessages((previous) => [...previous, { role: "user", text }]);
    setBusy(true);
    try {
      const result: RouteResult = await routeQuestion(text);
      if ("explanation" in result) {
        setMessages((previous) => [...previous, { role: "assistant", text: result.explanation }]);
      } else {
        const data = await executeTool(result.call);
        setMessages((previous) => [...previous, { role: "assistant", result: data, call: result.call }]);
      }
    } catch (error) {
      setMessages((previous) => [...previous, { role: "assistant", error: error instanceof Error ? error.message : "Something went wrong." }]);
    } finally {
      setBusy(false);
    }
  }

  const suggested = [
    "who is best rated between Faker and Chovy",
    "who is the best Galio player",
    "which team has the best draft score",
    "what is the worst champion in general?",
    "show me T1's recent matches",
    "best Tier 1 LCK mid with at least 100 games",
    "what is Faker's most median performance champion?",
  ];

  if (floating && pathname === "/chat") return null;

  if (floating && !open) {
    return (
      <button type="button" className={styles.floatingButton} onClick={() => setOpen(true)} aria-label="Open Ask Scryglass">
        <ChatIcon />
      </button>
    );
  }

  const chatClassName = [
    floating ? styles.chatFloating : styles.chat,
    expanded ? styles.chatExpanded : "",
  ].filter(Boolean).join(" ");

  return (
    <section className={chatClassName} aria-label="Ask Scryglass">
      <header className={styles.chatHeader}>
        <strong>Ask Scryglass</strong>
        <div className={styles.windowActions}>
          <button
            type="button"
            className={styles.windowButton}
            onClick={() => setExpanded((value) => !value)}
            aria-label={expanded ? "Restore chat size" : "Expand chat"}
            aria-pressed={expanded}
          >
            <ExpandIcon expanded={expanded} />
          </button>
          {floating ? (
            <button
              type="button"
              className={styles.windowButton}
              onClick={() => {
                setExpanded(false);
                setOpen(false);
              }}
              aria-label="Close Ask Scryglass"
            >
              <CloseIcon />
            </button>
          ) : null}
        </div>
      </header>
      <div
        ref={threadRef}
        className={styles.thread}
        data-native-scroll
        role="log"
        aria-label="Chat messages"
        aria-live="polite"
        tabIndex={0}
      >
        {messages.length === 0 && (
          <div className={styles.empty}>
            <p>Ask about players, champions, teams, matches, ratings, tier lists, schedules, methodology, or Draft Score availability.</p>
            <div className={styles.suggestions}>
              {suggested.map((suggestion) => (
                <button key={suggestion} type="button" onClick={() => setInput(suggestion)}>{suggestion}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((message, index) => (
          <div key={index} className={`${styles.message} ${message.role === "user" ? styles.user : styles.assistant}`}>
            {message.text && <p>{message.text}</p>}
            {message.error && <p className={styles.error}>{message.error}</p>}
            {message.call && message.result !== undefined && resultTable(message.result, message.call)}
          </div>
        ))}
        {busy && <div className={`${styles.message} ${styles.assistant}`}><ChatThinking /></div>}
      </div>
      <form
        className={styles.form}
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <input
          aria-label="Ask a question"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask about a player, team, or match"
        />
        <button type="submit" disabled={busy || !input.trim()}>Ask</button>
      </form>
      <p className={styles.note}>Answers use the active public release. Numerical results come from published data.</p>
    </section>
  );
}
