"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  loadMatchBundle,
  loadMatchModelPrior,
  type MatchModelPrior,
  type QueryRow,
} from "@/lib/duck";
import { MatchScoreboard } from "./MatchScoreboard";
import { ModelChecklist } from "./ModelChecklist";

type Props = {
  baseUrl: string;
  years: number[];
  gameId: string;
  yearHint?: number;
};

export function ratingHistorySource(
  teamEloDiff: number | null | undefined,
  playerEloDiff: number | null | undefined,
): string | null {
  if (teamEloDiff != null && playerEloDiff != null) {
    return "pre-match team + player rating history";
  }
  if (teamEloDiff != null) return "pre-match team rating history";
  if (playerEloDiff != null) return "pre-match player rating history";
  return null;
}

export function MatchLoader({ baseUrl, years, gameId, yearHint }: Props) {
  const [map, setMap] = useState<QueryRow | null>(null);
  const [players, setPlayers] = useState<QueryRow[]>([]);
  const [prior, setPrior] = useState<MatchModelPrior | null>(null);
  const [priorLoading, setPriorLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("Loading…");

  const strength = useMemo(
    () => ({
      teamEloDiff: prior?.muDiff ?? null,
      playerEloDiff: prior?.playerMuDiff ?? null,
      source: ratingHistorySource(prior?.muDiff, prior?.playerMuDiff),
    }),
    [prior],
  );
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setStatus("Loading match from pack…");
      setError(null);
      setPrior(null);
      try {
        const order =
          yearHint && years.includes(yearHint)
            ? [yearHint, ...years.filter((y) => y !== yearHint)]
            : years;
        const bundle = await loadMatchBundle(baseUrl, order, gameId);
        if (cancelled) return;
        if (!bundle) {
          setError("This game is outside the selected pack years.");
          setStatus("Missing");
          return;
        }
        setMap(bundle.map);
        setPlayers(bundle.players);
        setStatus(
          bundle.players.length
            ? `Loaded · ${bundle.players.length} players`
            : "Loaded · game data only · player rows unavailable",
        );
        setPriorLoading(true);
        const p = await loadMatchModelPrior(baseUrl, bundle.year, bundle.map);
        if (!cancelled) {
          setPrior(p);
          setPriorLoading(false);
        }
      } catch {
        if (!cancelled) {
          setError("Could not load this match from the public pack. Try again.");
          setStatus("Error");
          setPriorLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, years, gameId, yearHint]);

  return (
    <div className="space-y-4">
      <p className="text-xs text-[var(--ink-muted)]">
        <Link href="/browse" className="row-link">
          ← Browse
        </Link>
        {" · "}
        <Link href="/browse/head-to-head" className="row-link">
          Head-to-head
        </Link>
        {" · "}
        <span className="status-hint">{status}</span>
      </p>
      {error && <p className="error-banner">{error}</p>}
      {map && (
        <>
          <MatchScoreboard
            map={map}
            players={players}
            draftPctBlue={null}
          />
          <ModelChecklist
            map={map}
            players={players}
            prior={prior}
            loading={priorLoading}
            strength={strength}
          />
        </>
      )}
    </div>
  );
}
