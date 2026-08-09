"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import styles from "./TierListExplorer.module.css";

const ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"] as const;
const TIER_ORDER = ["Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D"] as const;
type TierBucket = (typeof TIER_ORDER)[number];

type TierRow = {
  scope_id: string;
  role: string;
  patch: string;
  champion: string;
  champion_id: string;
  champion_image_url: string | null;
  rank: number;
  rank_delta: number | null;
  movement: "up" | "down" | "flat" | "new";
  tier_bucket: TierBucket;
  played_maps: number;
  counterability_status: string;
  matchup_maps: number;
  matchup_opponents: number;
  expected_counter_breadth: number | null;
};

type Scope = {
  scope_id: string;
  scope_kind: "patch";
  role: string;
  patch: string;
  as_of: string;
  status: "production" | "unavailable";
  row_count: number;
};

type Response = {
  status: string;
  reason?: string;
  generated_at?: string;
  as_of?: string;
  source_freshness?: "oe_daily_export" | "oe_with_same_day_grid_bridge";
  options?: { roles: string[]; patches: string[]; tier_buckets?: TierBucket[] };
  scopes?: Scope[];
  rows?: TierRow[];
};

const EMPTY: Response = { status: "unavailable" };

function patchOrder(value: string): number {
  const [major, minor] = value.split(".").map(Number);
  return (Number.isFinite(major) ? major : 0) * 1000 + (Number.isFinite(minor) ? minor : 0);
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
  if (data.source_freshness === "oe_with_same_day_grid_bridge") return "OE + same-day source";
  if (data.source_freshness === "oe_daily_export") return "OE daily source";
  return "accepted source snapshot";
}

function Select({
  label,
  value,
  options,
  onChange,
  emptyLabel,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
  emptyLabel: string;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">{emptyLabel}</option>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );
}

function ChampionTile({ row }: { row: TierRow }) {
  const [imageFailed, setImageFailed] = useState(false);
  const matchupNote = row.counterability_status === "available"
    ? `${row.matchup_opponents} supported opponents across ${row.matchup_maps.toFixed(1)} effective maps; expected counter breadth ${row.expected_counter_breadth?.toFixed(1) ?? "—"}`
    : "matchup sample below the Blind or Counter threshold";

  return (
    <article className={styles.championCard} title={`${row.champion}, rank ${row.rank}. ${matchupNote}.`}>
      <div className={styles.imageFrame}>
        {row.champion_image_url && !imageFailed ? (
          // The source is already a small square asset. Direct loading keeps
          // this page independent of an image-optimization service.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={row.champion_image_url} alt="" loading="lazy" onError={() => setImageFailed(true)} />
        ) : <span className={styles.imageFallback} aria-hidden>{row.champion.slice(0, 1)}</span>}
      </div>
      <div className={styles.cardCopy}>
        <strong>{row.champion}</strong>
        <span className={styles.cardMeta}>
          <span className={movementClass(row)} aria-label={`rank movement ${movementText(row)}`}>{movementText(row)}</span>
          <span>#{row.rank}</span>
        </span>
      </div>
    </article>
  );
}

function TierBoard({ scope, role, rows }: { scope?: Scope; role: string; rows: TierRow[] }) {
  const byTier = useMemo(() => {
    const grouped = new Map<TierBucket, TierRow[]>();
    for (const tier of TIER_ORDER) grouped.set(tier, []);
    for (const row of rows) grouped.get(row.tier_bucket)?.push(row);
    return grouped;
  }, [rows]);

  return (
    <section className={styles.board} aria-label={`Patch ${scope?.patch ?? rows[0]?.patch} ${role} tier list`}>
      <header className={styles.boardHeader}>
        <div><p className={styles.eyebrow}>{role}</p><h2>Patch {scope?.patch ?? rows[0]?.patch}</h2></div>
        <div className={styles.boardMeta}>
          <span>{rows.length} champions</span>
          <span>all eligible competitions</span>
          <span>{scope?.status ?? "production"}</span>
        </div>
      </header>
      <div className={styles.tierRows}>
        {TIER_ORDER.map((tier) => {
          const tierRows = byTier.get(tier) ?? [];
          return (
            <div className={styles.tierRow} key={tier}>
              <div className={`${styles.tierLabel} ${styles[`tier${tier.replace(" ", "")}`]}`}>
                <strong>{tier.split(" ")[0]}</strong><span>{tier.split(" ")[1] ?? "rating"}</span>
              </div>
              <div className={styles.championGrid}>
                {tierRows.length
                  ? tierRows.map((row) => <ChampionTile key={`${row.role}|${row.champion_id}`} row={row} />)
                  : <span className={styles.emptyTier}>no eligible champions in this tier</span>}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function TierListExplorer() {
  const [role, setRole] = useState("");
  const [patch, setPatch] = useState("");
  const [data, setData] = useState<Response>(EMPTY);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async (signal?: AbortSignal, showLoading = false) => {
    if (showLoading) setLoading(true);
    try {
      const response = await fetch("/rankings/tierlists.json", { signal });
      if (!response.ok) throw new Error("tier-list file is unavailable");
      setData((await response.json()) as Response);
    } catch {
      if (!signal?.aborted) setData(EMPTY);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/rankings/tierlists.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("tier-list file is unavailable");
        return response.json() as Promise<Response>;
      })
      .then((payload) => setData(payload))
      .catch(() => {
        if (!controller.signal.aborted) setData(EMPTY);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  if (data.status !== "available") {
    return (
      <div className={styles.unavailable}>
        <p>Patch-wide tier lists are waiting for an accepted local rebuild.</p>
        <span>{data.reason ?? "The previous league-specific boards are retired."}</span>
      </div>
    );
  }

  const scopes = data.scopes ?? [];
  const patchOptions = [...new Set(data.options?.patches ?? scopes.map((scope) => scope.patch))]
    .sort((a, b) => patchOrder(b) - patchOrder(a));
  const activePatch = patch || patchOptions[0] || "";
  const roleOptions = data.options?.roles ?? [...ROLE_ORDER];
  const rows = (data.rows ?? []).filter((row) => row.patch === activePatch && (!role || row.role === role));
  const selectedRoles = role ? [role] : ROLE_ORDER.filter((candidate) => rows.some((row) => row.role === candidate));
  const scopeByRole = new Map(scopes.filter((scope) => scope.patch === activePatch).map((scope) => [scope.role, scope]));

  return (
    <section className={styles.section}>
      <div className={styles.filters}>
        <Select label="Patch" value={patch} options={patchOptions} onChange={setPatch} emptyLabel={`latest (${activePatch})`} />
        <Select label="Role" value={role} options={roleOptions} onChange={setRole} emptyLabel="all roles" />
        <button className={styles.button} onClick={() => void refresh(undefined, true)} disabled={loading}>{loading ? "loading…" : "refresh"}</button>
      </div>
      <div className={styles.meta}>{rows.length} champion rows · {freshnessLabel(data)} · updated {data.as_of ?? data.generated_at}</div>
      {selectedRoles.length ? (
        <div className={styles.boards}>
          {selectedRoles.map((selectedRole) => (
            <TierBoard key={`${activePatch}|${selectedRole}`} scope={scopeByRole.get(selectedRole)} role={selectedRole} rows={rows.filter((row) => row.role === selectedRole)} />
          ))}
        </div>
      ) : (
        <div className={styles.unavailable}><p>No accepted tier list matches this patch and role.</p></div>
      )}
    </section>
  );
}
