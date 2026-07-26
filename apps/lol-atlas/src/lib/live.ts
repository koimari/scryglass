export type LivePlayer = {
  name: string;
  role?: string | null;
  champion?: string | null;
};

export type LiveTeam = {
  name: string;
  score?: number | null;
  game_winner?: boolean | null;
  players: LivePlayer[];
};

export type LiveContribution = {
  key: string;
  label: string;
  delta_pp: number;
  value?: number;
  source?: string | null;
};

export type LiveEvaluation = {
  status: string;
  model: string;
  phase: string;
  minute?: number | null;
  p_blue?: number | null;
  p_red?: number | null;
  blue_team?: string | null;
  red_team?: string | null;
  draft_status?: string;
  strength_status?: string;
  features: Record<string, number | null>;
  feature_sources: Record<string, string | null>;
  missing: string[];
  warnings: string[];
  contributions: LiveContribution[];
};

export type LiveSnapshot = {
  schema_version: string;
  series_id: string;
  game_id?: string | null;
  game_number?: number | null;
  sequence_number?: number | null;
  emitted_utc: string;
  status: "live" | "stale" | "finished" | "unavailable" | string;
  tournament?: string | null;
  patch?: string | null;
  teams: { blue: LiveTeam; red: LiveTeam };
  game_state: {
    clock_seconds?: number | null;
    gold_by_side: Record<string, number | null>;
    kills_by_side: Record<string, number | null>;
    objectives: Record<string, Array<{ name: string; count?: number; completed?: boolean }>>;
  };
  evaluation: LiveEvaluation;
  provenance: {
    source: string;
    feed_sequence?: number | null;
    rating_pack_id?: string | null;
    rating?: {
      source?: string | null;
      blue?: { team?: string; mu_total?: number | null } | null;
      red?: { team?: string; mu_total?: number | null } | null;
      missing?: string[];
    };
    broadcast_synchronized?: boolean;
  };
};

export type LivePointer = {
  schema_version: string;
  series_id: string;
  sequence_number?: number | null;
  emitted_utc: string;
  status: string;
  evaluation_status?: string;
  state_clock_seconds?: number | null;
  snapshot_path: string;
  snapshot_url: string;
  latest_path?: string;
  latest_url?: string;
  tournament?: string | null;
  game_number?: number | null;
  teams?: { blue?: { name?: string }; red?: { name?: string } };
};

export type LiveIndex = {
  schema_version: string;
  updated_utc: string;
  series: LivePointer[];
};

export function secondsLabel(seconds?: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

export function relativeLiveTime(raw?: string | null): string {
  if (!raw) return "time unavailable";
  const timestamp = Date.parse(raw);
  if (!Number.isFinite(timestamp)) return "time unavailable";
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 10) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  return `${minutes}m ago`;
}

export function probabilityLabel(value?: number | null): string {
  return value == null || !Number.isFinite(value) ? "—" : `${Math.round(value * 100)}%`;
}
