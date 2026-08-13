"use client";

import { useEffect, useRef, useState } from "react";
import { executeTool, routeQuestion, type RouteResult, type ToolCall } from "@/lib/supportChat";
import { playerSlug, teamSlug } from "@/lib/pack";
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
  if (category === "teams") {
    return table(
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
    );
  }

  return table(
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
  );
}

function resultTable(result: unknown, call: ToolCall): React.ReactNode {
  const data = result as Record<string, unknown> | null;

  if (call.tool === "query_players" && data?.kind === "player_query" && Array.isArray(data.rows)) {
    const query = data as unknown as PlayerQueryResult;
    const hasChampion = query.rows.some((row) => Boolean(row.champion));
    return (
      <div className={styles.queryResult} data-testid="player-query-result">
        <p className={styles.queryHeadline} data-testid="player-query-headline">{query.answer.headline}</p>
        <p className={styles.queryBasis} data-testid="player-query-basis">{query.answer.basis}</p>
        {query.answer.caveat ? <p className={styles.queryCaveat} data-testid="player-query-caveat">{query.answer.caveat}</p> : null}
        {query.rows.length ? table(
          <table className={styles.resultTable}>
            <thead>
              {hasChampion ? (
                <tr><th>Player</th><th>Champion</th><th>Role</th><th>Level</th><th className={styles.numeric}>Champion games</th><th className={styles.numeric}>Champion WR</th><th className={styles.numeric}>Rating</th></tr>
              ) : (
                <tr><th>Player</th><th>Role</th><th>Team</th><th>League</th><th>Level</th><th className={styles.numeric}>Rating</th><th className={styles.numeric}>Games</th><th className={styles.numeric}>WR</th></tr>
              )}
            </thead>
            <tbody>
              {query.rows.map((row) => (
                <tr key={`${row.name}-${row.champion ?? "player"}`}>
                  <td><a className="row-link" href={`/elo/player/${playerSlug(row.name)}`}>{row.name}</a></td>
                  {hasChampion ? (
                    <>
                      <td>{present(row.champion)}</td>
                      <td>{roleName(row.role)}</td>
                      <td>{tierName(row.tier)}</td>
                      <td className={styles.numeric}>{present(row.champion_games)}</td>
                      <td className={styles.numeric}>{percentage(row.champion_win_rate)}</td>
                      <td className={styles.numeric}>{rating(row.rating)}</td>
                    </>
                  ) : (
                    <>
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
    return (
      <div className={styles.comparison}>
        <p className={styles.comparisonAnswer}>
          {better && difference != null
            ? <><strong>{better}</strong> has the higher rating by {difference} {difference === 1 ? "point" : "points"}.</>
            : "The published ratings do not support a winner."}
        </p>
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

export default function SupportChat({ floating = false }: { floating?: boolean }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(!floating);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
    "show me T1's recent matches",
    "best Tier 1 LCK mid with at least 100 games",
    "when does the next LEC game happen",
    "how does the draft win share work",
  ];

  if (floating && !open) {
    return (
      <button type="button" className={styles.floatingButton} onClick={() => setOpen(true)} aria-label="Open support chat">
        <ChatIcon />
      </button>
    );
  }

  return (
    <section className={floating ? styles.chatFloating : styles.chat} aria-label="Scryglass support chat">
      <header className={styles.chatHeader}>
        <div>
          <span>Research support</span>
          <strong>Ask Scryglass</strong>
        </div>
        {floating ? (
          <button type="button" className={styles.floatingClose} onClick={() => setOpen(false)} aria-label="Close support chat">×</button>
        ) : null}
      </header>
      <div className={styles.thread}>
        {messages.length === 0 && (
          <div className={styles.empty}>
            <p>Ask about players, teams, matches, ratings, tier lists, schedules, or methodology.</p>
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
        {busy && <div className={`${styles.message} ${styles.assistant}`}><p>…</p></div>}
        <div ref={endRef} />
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
