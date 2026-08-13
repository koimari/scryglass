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

function renderValue(value: unknown, depth = 0): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map((entry) => renderValue(entry, depth + 1)).filter(Boolean).join(" · ");
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v != null && v !== "")
      .map(([key, v]) => {
        if (typeof v === "object") return `${key}: ${renderValue(v, depth + 1)}`;
        return `${key}: ${String(v)}`;
      });
    return entries.join(" · ");
  }
  return String(value);
}

function resultTable(result: unknown, call: ToolCall): React.ReactNode {
  const data = result as { rows?: unknown[]; matches?: unknown[]; upcoming?: unknown[]; teams?: unknown[]; player?: string; players?: unknown[]; count?: number } | null;
  if (data?.matches) {
    return (
      <table className={styles.resultTable}>
        <thead><tr><th>Date</th><th>League</th><th>Blue</th><th>Red</th><th>Winner</th></tr></thead>
        <tbody>
          {(data.matches as Array<{ date: string; league: string; blue_team: string; red_team: string; blue_win: number; game_id: string }>).map((match) => (
            <tr key={match.game_id}>
              <td>{match.date.slice(0, 10)}</td>
              <td>{match.league}</td>
              <td>{match.blue_team}</td>
              <td>{match.red_team}</td>
              <td>{match.blue_win === 1 ? match.blue_team : match.red_team}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (data?.rows) {
    return (
      <table className={styles.resultTable}>
        <thead><tr><th>Champion</th><th>Role</th><th>Tier</th><th>Rank</th><th>Maps</th></tr></thead>
        <tbody>
          {(data.rows as Array<{ champion: string; role: string; tier_bucket: string; rank: number; played_maps: number }>).map((row, index) => (
            <tr key={`${row.champion}-${index}`}>
              <td>{row.champion}</td><td>{row.role}</td><td>{row.tier_bucket}</td><td>{row.rank}</td><td>{row.played_maps}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (data?.upcoming) {
    return (
      <table className={styles.resultTable}>
        <thead><tr><th>Start (UTC)</th><th>Team 1</th><th>Team 2</th><th>BO</th></tr></thead>
        <tbody>
          {(data.upcoming as Array<{ start_utc: string; team1: string; team2: string; best_of?: number }>).slice(0, 10).map((row, index) => (
            <tr key={index}><td>{row.start_utc}</td><td>{row.team1}</td><td>{row.team2}</td><td>{row.best_of ?? ""}</td></tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (data?.teams) {
    return (
      <table className={styles.resultTable}>
        <thead><tr><th>Team</th><th>Rating</th><th>League</th><th>W</th><th>WR</th></tr></thead>
        <tbody>
          {(data.teams as Array<{ team: string; rating: number; league: string | null; wins: number; win_rate: number | null }>).map((row) => (
            <tr key={row.team}>
              <td><a className="row-link" href={`/teams/${teamSlug(row.team)}`}>{row.team}</a></td>
              <td>{Math.round(row.rating)}</td>
              <td>{row.league ?? ""}</td>
              <td>{row.wins}</td>
              <td>{row.win_rate != null ? `${Math.round(row.win_rate * 100)}%` : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (call.tool === "player" && data?.player) {
    const profile = data as unknown as { player: string; rating: number | null; role: string | null; team: string | null; league: string | null; grade_a_games: number; games: number; win_rate: number | null; recent_form: number | null };
    return (
      <div className={styles.resultLines}>
        <p><a className="row-link" href={`/players/${playerSlug(profile.player)}`}>{profile.player}</a> — {profile.role ?? "role unknown"}{profile.team ? ` · ${profile.team}` : ""}{profile.league ? ` · ${profile.league}` : ""}</p>
        <p>Rating {profile.rating != null ? Math.round(profile.rating) : "—"} · {profile.grade_a_games} A-grade games · {profile.games} games · WR {profile.win_rate != null ? `${Math.round(profile.win_rate * 100)}%` : "—"}</p>
      </div>
    );
  }
  return <p className={styles.resultText}>{renderValue(result)}</p>;
}

export default function SupportChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
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
    "who is the player with most A grade games",
    "what is the jungler with a rating of 1643",
    "show me T1's recent matches",
    "what is the best mid laner this patch",
    "when does the next LEC game happen",
    "how does the draft win share work",
  ];

  return (
    <section className={styles.chat} aria-label="Scryglass support chat">
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
          placeholder="e.g. who is the player with the most A grade games?"
        />
        <button type="submit" disabled={busy || !input.trim()}>Ask</button>
      </form>
      <p className={styles.note}>Answers are computed from the active public release; no AI-generated numbers.</p>
    </section>
  );
}
