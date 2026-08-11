"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { ProfileGame, ProfileGrade, ProfileParticipant } from "@/lib/pack";
import { TeamMark } from "./TeamMark";
import styles from "./RatingProfiles.module.css";

type GradeFilter = "all" | "A" | "B" | "C" | "D" | "F" | "ungraded";

function shortDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value.slice(0, 10);
  return date.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
}

function playerInGame(game: ProfileGame, player: string): ProfileParticipant | undefined {
  return game.players.find((participant) => participant.player.toLowerCase() === player.toLowerCase());
}

function teamWon(game: ProfileGame, team: string): boolean {
  return game.blue_team.toLowerCase() === team.toLowerCase() ? game.blue_win === 1 : game.blue_win === 0;
}

function gradeGroup(grade: ProfileGrade | undefined): GradeFilter {
  if (!grade || grade.status !== "available") return "ungraded";
  return grade.grade.startsWith("A") ? "A" : grade.grade as GradeFilter;
}

function gradeMeaning(grade: string): string {
  if (grade === "A+" || grade === "A") return "Standout";
  if (grade === "B") return "Strong";
  if (grade === "C") return "Typical";
  if (grade === "D") return "Below standard";
  if (grade === "F") return "Poor";
  return "Unavailable";
}

function gradeSignal(grade: Extract<ProfileGrade, { status: "available" }>): string {
  const signals = [
    { value: grade.components.self, positive: "Above usual", negative: "Below usual" },
    { value: grade.components.team, positive: "Above teammates", negative: "Below teammates" },
    { value: grade.components.opponent, positive: "Ahead of role opponent", negative: "Behind role opponent" },
    { value: grade.components.league_role, positive: "Above role baseline", negative: "Below role baseline" },
  ]
    .filter((signal) => Math.abs(signal.value) >= 0.25)
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
    .slice(0, 2)
    .map((signal) => signal.value >= 0 ? signal.positive : signal.negative);
  return signals.join(" · ") || "Near all baselines";
}

function gradeTitle(grade: Extract<ProfileGrade, { status: "available" }>): string {
  const direction = (value: number) => value >= 0.25 ? "above" : value <= -0.25 ? "below" : "near";
  return `Grade ${grade.grade}, score ${grade.score.toFixed(1)}. Usual form: ${direction(grade.components.self)}. Teammates: ${direction(grade.components.team)}. Opposing role: ${direction(grade.components.opponent)}. League-role baseline: ${direction(grade.components.league_role)}. Full-game output matters more than KDA alone.`;
}

function ChampionPortrait({ name, imageUrl }: { name: string | null; imageUrl?: string | null }) {
  return (
    <span className={styles.portrait} title={name ?? undefined}>
      {imageUrl ? (
        // CommunityDragon supplies the champion portraits in the published pack.
        // eslint-disable-next-line @next/next/no-img-element
        <img src={imageUrl} alt={name ?? "Champion"} loading="lazy" />
      ) : <span aria-hidden>{name?.slice(0, 1) ?? "?"}</span>}
    </span>
  );
}

export function RecentGames({
  games,
  championImages,
  player,
  team,
}: {
  games: ProfileGame[];
  championImages: Record<string, string>;
  player?: string;
  team?: string;
}) {
  const [filter, setFilter] = useState<GradeFilter>("all");
  const gradeCounts = useMemo(() => {
    const counts = new Map<GradeFilter, number>();
    if (!player) return counts;
    for (const game of games) {
      const group = gradeGroup(playerInGame(game, player)?.grade);
      counts.set(group, (counts.get(group) ?? 0) + 1);
    }
    return counts;
  }, [games, player]);
  const visibleGames = player && filter !== "all"
    ? games.filter((game) => gradeGroup(playerInGame(game, player)?.grade) === filter)
    : games;
  const filters: Array<{ value: GradeFilter; label: string }> = [
    { value: "all", label: "All" },
    ...(["A", "B", "C", "D", "F"] as const).map((value) => ({ value, label: value })),
    ...((gradeCounts.get("ungraded") ?? 0) > 0
      ? [{ value: "ungraded" as const, label: "Unavailable" }]
      : []),
  ];

  if (!games.length) return <p className={styles.empty}>Recent game details are waiting for the next accepted data refresh.</p>;

  return (
    <>
      {player ? (
        <div className={styles.matchFilters} role="group" aria-label="Filter recent games by grade">
          <span>Grade</span>
          {filters.map((option) => (
            <button
              type="button"
              key={option.value}
              className={filter === option.value ? styles.matchFilterActive : styles.matchFilterButton}
              aria-pressed={filter === option.value}
              onClick={() => setFilter(option.value)}
            >
              {option.label}
              <small>{option.value === "all" ? games.length : gradeCounts.get(option.value) ?? 0}</small>
            </button>
          ))}
        </div>
      ) : null}
      {visibleGames.length ? (
        <div className={styles.matchList}>
          {visibleGames.map((game) => {
            const participant = player ? playerInGame(game, player) : undefined;
            const focusTeam = participant
              ? participant.side === "Blue" ? game.blue_team : game.red_team
              : team ?? "";
            const won = focusTeam ? teamWon(game, focusTeam) : false;
            const opponent = game.blue_team.toLowerCase() === focusTeam.toLowerCase() ? game.red_team : game.blue_team;
            const availableGrade = participant?.grade?.status === "available" ? participant.grade : null;
            const grade = availableGrade?.grade ?? "—";
            return (
              <Link className={`${styles.matchRow} ${participant ? styles.playerMatch : styles.teamMatch}`} href={`/matches/${encodeURIComponent(game.game_id)}`} key={game.game_id}>
                <span className={`${styles.resultMark} ${won ? styles.win : styles.loss}`}>{won ? "W" : "L"}</span>
                {participant ? <ChampionPortrait name={participant.champion} imageUrl={participant.champion ? championImages[participant.champion] : null} /> : null}
                <div className={styles.matchMain}>
                  <strong>{(participant?.champion ?? opponent) || `${game.blue_team} vs ${game.red_team}`}</strong>
                  <small>{game.league} · {shortDate(game.date)}{participant ? "" : ` · ${game.blue_team === focusTeam ? "Blue" : "Red"} side`}</small>
                </div>
                {participant ? (
                  <div className={styles.matchOpponent} title={`Versus ${opponent}`}>
                    <span>vs</span>
                    <TeamMark team={opponent} size="small" />
                  </div>
                ) : null}
                {participant ? (
                  <div className={styles.matchScore} title={availableGrade ? gradeTitle(availableGrade) : "Grade unavailable."}>
                    <strong>{grade}{availableGrade ? <em>{availableGrade.score.toFixed(0)}</em> : null}</strong>
                    <small>{availableGrade ? gradeSignal(availableGrade) : gradeMeaning(grade)}</small>
                    <span>{participant.kills ?? "—"} / {participant.deaths ?? "—"} / {participant.assists ?? "—"}</span>
                  </div>
                ) : null}
              </Link>
            );
          })}
        </div>
      ) : (
        <p className={styles.empty}>No recent games have this grade.</p>
      )}
    </>
  );
}
