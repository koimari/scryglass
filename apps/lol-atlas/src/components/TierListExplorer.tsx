"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import styles from "./TierListExplorer.module.css";

type TierRow = {
  scope_id: string;
  league: string | null;
  event_kind: string | null;
  competition_tier: string | null;
  role: string;
  patch: string;
  champion: string;
  tier_value_pp: number;
  rank: number;
  tier_bucket: "S" | "A" | "B" | "C" | "D";
  played_maps: number;
  counterability_status: string;
};

type Response = {
  status: string;
  generated_at?: string;
  development_only?: boolean;
  claim_ceiling?: Record<string, unknown>;
  options?: {
    regions: string[];
    leagues: string[];
    event_kinds: string[];
    competition_tiers: string[];
    roles: string[];
    patches: string[];
  };
  rows?: TierRow[];
};

const EMPTY: Response = { status: "unavailable" };

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">all</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export function TierListExplorer() {
  const [region, setRegion] = useState("");
  const [league, setLeague] = useState("");
  const [international, setInternational] = useState("");
  const [tier, setTier] = useState("");
  const [role, setRole] = useState("");
  const [patch, setPatch] = useState("");
  const [data, setData] = useState<Response>(EMPTY);
  const [loading, setLoading] = useState(true);

  const query = useMemo(() => {
    const params = new URLSearchParams();
    if (region) params.set("region", region);
    if (league) params.set("league", league);
    if (international) params.set("international", international);
    if (tier) params.set("competition_tier", tier);
    if (role) params.set("role", role);
    if (patch) params.set("patch", patch);
    return params.toString();
  }, [region, league, international, tier, role, patch]);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v2/tierlist?${query}`, { signal });
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
    const timer = window.setTimeout(() => {
      void refresh(controller.signal);
    }, 0);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [refresh]);

  const options = data.options ?? {
    regions: [],
    leagues: [],
    event_kinds: [],
    competition_tiers: [],
    roles: ["top", "jungle", "mid", "bot", "support"],
    patches: [],
  };

  if (data.status !== "available") {
    return (
      <div className={styles.unavailable}>
        <p>
          Tier lists are unavailable right now. The development artifact is
          rebuilt from the frozen Draft Score candidate and the warehouse; it
          is not part of the public pack yet.
        </p>
      </div>
    );
  }

  const rows = data.rows ?? [];
  const scopes = new Set(rows.map((row) => row.scope_id));

  return (
    <section className={styles.section}>
      <div className={styles.filters}>
        <Select label="Region" value={region} options={options.regions} onChange={setRegion} />
        <Select label="League" value={league} options={options.leagues} onChange={setLeague} />
        <Select label="International" value={international} options={options.event_kinds} onChange={setInternational} />
        <Select label="Competition tier" value={tier} options={options.competition_tiers} onChange={setTier} />
        <Select label="Role" value={role} options={options.roles} onChange={setRole} />
        <Select label="Patch" value={patch} options={options.patches} onChange={setPatch} />
        <button className={styles.button} onClick={() => void refresh()} disabled={loading}>
          {loading ? "loading…" : "refresh"}
        </button>
      </div>
      <div className={styles.meta}>
        {scopes.size} scope{scopes.size === 1 ? "" : "s"} · {rows.length} rows · generated {data.generated_at}
      </div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Scope</th>
              <th>Role</th>
              <th>Champion</th>
              <th>Tier</th>
              <th>IV (pp)</th>
              <th>Played</th>
              <th>Counterability</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.scope_id}|${row.role}|${row.champion}`}>
                <td>{row.scope_id}</td>
                <td>{row.role}</td>
                <td className={styles.champion}>{row.champion}</td>
                <td className={styles[`tier${row.tier_bucket}`] ?? ""}>{row.tier_bucket}</td>
                <td>{row.tier_value_pp.toFixed(2)}</td>
                <td>{row.played_maps}</td>
                <td>{row.counterability_status === "available" ? "yes" : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
