"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import styles from "./TierListExplorer.module.css";

const ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"] as const;
const TIER_ORDER = ["Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D"] as const;
const LEVEL_ORDER = ["tier1", "tier2", "tier3", "international", "interregional"] as const;
const LEAGUE_ORDER = ["LCS", "LEC", "LCK", "LPL", "LCP", "CBLOL", "PCS"] as const;
const LEVEL_LABELS: Record<string, string> = {
  tier1: "Tier 1",
  tier2: "Tier 2",
  tier3: "Tier 3",
  international: "International",
  interregional: "Interregional",
};
type TierBucket = (typeof TIER_ORDER)[number];

type TierRow = {
  scope_id: string;
  region: string | null;
  league: string | null;
  event_kind: string | null;
  competition_tier: string | null;
  role: string;
  patch: string;
  as_of: string;
  champion: string;
  champion_id: string;
  champion_image_url: string | null;
  tier_value_pp: number;
  rating: number;
  rating_delta: number | null;
  rank: number;
  previous_rank: number | null;
  rank_delta: number | null;
  movement: "up" | "down" | "flat" | "new";
  tier_bucket: TierBucket;
  played_maps: number;
  counterability_status: string;
  matchup_maps: number;
  matchup_opponents: number;
  blind_score_pp: number | null;
  countered_opponent_count: number | null;
  countered_opponent_share: number | null;
  expected_counter_breadth: number | null;
};

type Scope = {
  scope_id: string;
  scope_kind: string;
  region: string | null;
  league: string | null;
  event_kind: string | null;
  competition_tier: string | null;
  role: string;
  patch: string;
  as_of: string;
  status: "production" | "unavailable";
  row_count: number;
  fail_closed_status: string;
};

type Response = {
  status: string;
  reason?: string;
  generated_at?: string;
  as_of?: string;
  provenance?: {
    source_mode?: "oe_only" | "oe_plus_grid";
    freshness?: "oe_daily_export" | "oe_with_same_day_grid_bridge";
  };
  cells_available?: number;
  cells_total?: number;
  options?: {
    leagues: string[];
    event_kinds: string[];
    competition_tiers: string[];
    roles: string[];
    patches: string[];
    tier_buckets: TierBucket[];
  };
  scopes?: Scope[];
  rows?: TierRow[];
};

const EMPTY: Response = { status: "unavailable" };

function labelScope(scope: Scope | undefined, fallback: string): string {
  if (!scope) return fallback;
  if (scope.league) return `${scope.league} · ${scope.competition_tier ?? "league"}`;
  return `${scope.event_kind?.toUpperCase() ?? scope.scope_id} · ${scope.competition_tier ?? "international"}`;
}

function movementText(row: TierRow): string {
  if (row.movement === "new" || row.rank_delta === null) return "new";
  if (row.rank_delta > 0) return `↑ ${row.rank_delta}`;
  if (row.rank_delta < 0) return `↓ ${Math.abs(row.rank_delta)}`;
  return "—";
}

function movementClass(row: TierRow): string {
  if (row.movement === "up") return styles.movementUp;
  if (row.movement === "down") return styles.movementDown;
  if (row.movement === "new") return styles.movementNew;
  return styles.movementFlat;
}

function freshnessLabel(data: Response): string {
  if (data.provenance?.source_mode === "oe_plus_grid") return "OE + GRID bridge";
  if (data.provenance?.source_mode === "oe_only") return "OE daily source";
  return "source watermark";
}

function Select({
  label,
  value,
  options,
  onChange,
  emptyLabel = "all",
  allowEmpty = true,
  formatOption = (option: string) => option,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
  emptyLabel?: string;
  allowEmpty?: boolean;
  formatOption?: (option: string) => string;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {allowEmpty ? <option value="">{emptyLabel}</option> : null}
        {options.map((option) => (
          <option key={option} value={option}>
            {formatOption(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

function ChampionTile({ row }: { row: TierRow }) {
  const [imageFailed, setImageFailed] = useState(false);
  const matchupNote = row.counterability_status === "available"
    ? `${row.matchup_opponents} supported opponents across ${row.matchup_maps.toFixed(1)} effective maps; expected counter breadth ${row.expected_counter_breadth?.toFixed(1) ?? "—"}`
    : "matchup sample below the Blind/Counter threshold";
  return (
    <article className={styles.championCard} title={`${row.champion}, rank ${row.rank}. ${matchupNote}.`}>
      <div className={styles.imageFrame}>
        {row.champion_image_url && !imageFailed ? (
          <img
            src={row.champion_image_url}
            alt=""
            loading="lazy"
            onError={() => setImageFailed(true)}
          />
        ) : (
          <span className={styles.imageFallback} aria-hidden>
            {row.champion.slice(0, 1)}
          </span>
        )}
      </div>
      <div className={styles.cardCopy}>
        <strong>{row.champion}</strong>
        <span className={styles.cardMeta}>
          <span className={movementClass(row)} aria-label={`rank movement ${movementText(row)}`}>
            {movementText(row)}
          </span>
          <span>#{row.rank}</span>
        </span>
      </div>
    </article>
  );
}

function TierBoard({
  scope,
  role,
  rows,
}: {
  scope: Scope | undefined;
  role: string;
  rows: TierRow[];
}) {
  const byTier = useMemo(() => {
    const grouped = new Map<TierBucket, TierRow[]>();
    for (const tier of TIER_ORDER) grouped.set(tier, []);
    for (const row of rows) grouped.get(row.tier_bucket)?.push(row);
    return grouped;
  }, [rows]);

  return (
    <section className={styles.board} aria-label={`${labelScope(scope, "tier list")} ${role} tier list`}>
      <header className={styles.boardHeader}>
        <div>
          <p className={styles.eyebrow}>{role}</p>
          <h2>{labelScope(scope, "Tier list")}</h2>
        </div>
        <div className={styles.boardMeta}>
          <span>{rows.length} champions</span>
          <span>through patch {scope?.patch ?? rows[0]?.patch ?? "—"}</span>
          <span>{scope?.status ?? "production"}</span>
        </div>
      </header>
      <div className={styles.tierRows}>
        {TIER_ORDER.map((tier) => {
          const tierRows = byTier.get(tier) ?? [];
          return (
            <div className={styles.tierRow} key={tier}>
              <div className={`${styles.tierLabel} ${styles[`tier${tier.replace(" ", "")}`]}`}>
                <strong>{tier.split(" ")[0]}</strong>
                <span>{tier.split(" ")[1] ?? "rating"}</span>
              </div>
              <div className={styles.championGrid}>
                {tierRows.length ? (
                  tierRows.map((row) => <ChampionTile key={`${row.scope_id}|${row.role}|${row.champion_id}`} row={row} />)
                ) : (
                  <span className={styles.emptyTier}>no eligible champions in this cell</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function TierListExplorer() {
  const [level, setLevel] = useState("tier1");
  const [leagueOrEvent, setLeagueOrEvent] = useState("");
  const [role, setRole] = useState("");
  const [patch, setPatch] = useState("");
  const [data, setData] = useState<Response>(EMPTY);
  const [loading, setLoading] = useState(true);

  const query = useMemo(() => {
    const params = new URLSearchParams();
    if (level) params.set("competition_tier", level);
    if (leagueOrEvent) {
      if (level === "international") params.set("international", leagueOrEvent);
      else params.set("league", leagueOrEvent);
    }
    if (role) params.set("role", role);
    if (patch) params.set("patch", patch);
    return params.toString();
  }, [level, leagueOrEvent, role, patch]);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v2/tierlist${query ? `?${query}` : ""}`, { signal });
      const payload = (await response.json()) as Response;
      setData(payload);
    } catch {
      if (signal?.aborted) return;
      setData(EMPTY);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [query]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => void refresh(controller.signal), 0);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [refresh]);

  if (data.status !== "available") {
    return (
      <div className={styles.unavailable}>
        <p>Tier lists are waiting for the approved production artifact.</p>
        <span>{data.reason ?? "The source is unavailable or failed its integrity checks."}</span>
      </div>
    );
  }

  const rows = data.rows ?? [];
  const scopes = data.scopes ?? [];
  const levels = LEVEL_ORDER.filter((candidate) => scopes.some((scope) => scope.competition_tier === candidate));
  const scopeCandidates = scopes.filter((scope) => {
    if (scope.competition_tier !== level) return false;
    if (role && scope.role !== role) return false;
    if (!leagueOrEvent) return true;
    return level === "international" ? scope.event_kind === leagueOrEvent : scope.league === leagueOrEvent;
  });
  const validLeagueOptions = [...new Set(
    scopes
      .filter((scope) => scope.competition_tier === level && (!role || scope.role === role))
      .map((scope) => level === "international" ? scope.event_kind : scope.league)
      .filter((value): value is string => Boolean(value)),
  )].sort((a, b) => {
    const aIndex = LEAGUE_ORDER.indexOf(a as (typeof LEAGUE_ORDER)[number]);
    const bIndex = LEAGUE_ORDER.indexOf(b as (typeof LEAGUE_ORDER)[number]);
    if (aIndex >= 0 || bIndex >= 0) return (aIndex < 0 ? 999 : aIndex) - (bIndex < 0 ? 999 : bIndex);
    return a.localeCompare(b);
  });
  const roleOptions = data.options?.roles ?? [...ROLE_ORDER];
  const patchOptions = [...new Set(scopeCandidates.map((scope) => scope.patch))].sort();
  const boardGroups = new Map<string, TierRow[]>();
  for (const row of rows) {
    const key = `${row.scope_id}|${row.role}`;
    const group = boardGroups.get(key) ?? [];
    group.push(row);
    boardGroups.set(key, group);
  }
  const orderedGroups = [...boardGroups.entries()].sort(([, aRows], [, bRows]) => {
    const aScope = aRows[0];
    const bScope = bRows[0];
    const aLabel = aScope?.league ?? aScope?.event_kind ?? "";
    const bLabel = bScope?.league ?? bScope?.event_kind ?? "";
    const aLeague = LEAGUE_ORDER.indexOf(aLabel as (typeof LEAGUE_ORDER)[number]);
    const bLeague = LEAGUE_ORDER.indexOf(bLabel as (typeof LEAGUE_ORDER)[number]);
    if (aLeague !== bLeague && (aLeague >= 0 || bLeague >= 0)) return (aLeague < 0 ? 999 : aLeague) - (bLeague < 0 ? 999 : bLeague);
    const labelOrder = aLabel.localeCompare(bLabel);
    if (labelOrder) return labelOrder;
    return ROLE_ORDER.indexOf(aScope?.role as (typeof ROLE_ORDER)[number]) - ROLE_ORDER.indexOf(bScope?.role as (typeof ROLE_ORDER)[number]);
  });
  const selectedGroups = role
    ? orderedGroups.filter(([, group]) => group[0]?.role === role)
    : orderedGroups;
  const scopeById = new Map(scopes.map((scope) => [scope.scope_id, scope]));

  return (
    <section className={styles.section}>
      <div className={styles.filters}>
        <Select label="League tier" value={level} options={levels} formatOption={(value) => LEVEL_LABELS[value] ?? value} onChange={(value) => { setLevel(value); setLeagueOrEvent(""); setPatch(""); }} allowEmpty={false} />
        <Select label={level === "international" ? "Event" : "League"} value={leagueOrEvent} options={validLeagueOptions} onChange={(value) => { setLeagueOrEvent(value); setPatch(""); }} emptyLabel={level === "international" ? "all events" : "all leagues"} />
        <Select label="Role" value={role} options={roleOptions} onChange={(value) => { setRole(value); setPatch(""); }} emptyLabel="all roles" />
        <Select label="Patch" value={patch} options={patchOptions} onChange={setPatch} emptyLabel="latest available" />
        <button className={styles.button} onClick={() => void refresh()} disabled={loading}>
          {loading ? "loading…" : "refresh"}
        </button>
      </div>
      <div className={styles.meta}>
        {data.cells_available ?? 0}/{data.cells_total ?? 0} scope cells · {rows.length} champions · {freshnessLabel(data)} · updated {data.as_of ?? data.generated_at}
      </div>
      {selectedGroups.length ? (
        <div className={styles.boards}>
          {selectedGroups.map(([key, group]) => {
            const scopeId = key.split("|")[0];
            return <TierBoard key={key} scope={scopeById.get(scopeId)} role={group[0]?.role ?? ""} rows={group} />;
          })}
        </div>
      ) : (
        <div className={styles.unavailable}>
          <p>No published tier list matches these filters.</p>
          <span>{LEVEL_LABELS[level] ?? level} · {leagueOrEvent || "all leagues"} · {role || "all roles"} · {patch || "latest available patch"}</span>
          <button className={styles.button} onClick={() => { setLeagueOrEvent(""); setRole(""); setPatch(""); }}>clear filters</button>
        </div>
      )}
    </section>
  );
}
