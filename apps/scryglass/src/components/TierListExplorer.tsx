"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import styles from "./TierListExplorer.module.css";

const TIER_LIST_URL =
  "https://97gks2fobqkgppwx.public.blob.vercel-storage.com/rankings/tierlists.json";
const ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"] as const;
const ROLE_LABELS: Record<string, string> = {
  top: "Top",
  jungle: "Jungle",
  mid: "Mid",
  bot: "Bot",
  support: "Support",
};
const BOARD_MODES = [
  { value: "first_pick", label: "First pick", note: "overall strength" },
  { value: "blind", label: "Blind", note: "stability across matchups" },
  { value: "counter", label: "Counter reach", note: "positive responses" },
  { value: "responses", label: "Responses", note: "answer a champion" },
  { value: "regions", label: "Regions", note: "regional context" },
] as const;
type BoardMode = (typeof BOARD_MODES)[number]["value"];

type TierBucket = "Z Blind" | "Z Counter" | "S Blind" | "S Counter" | "A" | "B" | "C" | "D";

type MatchupProfile = {
  champion: string;
  champion_id: string;
  model_edge_pp: number;
  posterior_interval_pp: { low: number; high: number };
  posterior_positive_probability?: number;
  effective_maps: number;
  series_count: number;
  evidence_status: "supported" | "limited";
  evidence_source?: string;
};

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
  tier_value_pp?: number;
  counterability_status: "available" | "unavailable" | string;
  matchup_maps: number;
  matchup_opponents: number;
  blind_score_pp: number | null;
  counter_score: number | null;
  countered_opponent_count: number | null;
  countered_opponent_share?: number | null;
  expected_counter_breadth: number | null;
  matchup_profile?: MatchupProfile[];
};

type RegionalRow = {
  champion: string;
  champion_id: string;
  regional_rank: number;
  global_rank: number;
  strength_score_pp: number;
  played_maps: number;
  sample_status: "thin" | "observed" | string;
};

type RegionalView = {
  id: string;
  label: string;
  maps: number;
  basis: string;
  rows: RegionalRow[];
};

type Scope = {
  scope_id: string;
  scope_kind: "patch";
  role: string;
  patch: string;
  as_of: string;
  status: "production" | "unavailable";
  row_count: number;
  regional_views?: RegionalView[];
};

type TierResponse = {
  status: string;
  reason?: string;
  generated_at?: string;
  as_of?: string;
  source_freshness?: "oe_daily_export" | "oe_with_same_day_grid_bridge";
  options?: { roles: string[]; patches: string[]; tier_buckets?: TierBucket[] };
  scopes?: Scope[];
  rows?: TierRow[];
};

const EMPTY: TierResponse = { status: "unavailable" };

function patchOrder(value: string): number {
  const [major, minor] = value.split(".").map(Number);
  return (Number.isFinite(major) ? major : 0) * 1000 + (Number.isFinite(minor) ? minor : 0);
}

function roleLabel(role: string): string {
  return ROLE_LABELS[role] ?? role;
}

function signedScore(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Unavailable";
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)} pp`;
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

function freshnessLabel(data: TierResponse): string {
  if (data.source_freshness === "oe_with_same_day_grid_bridge") return "OE + same-day source";
  if (data.source_freshness === "oe_daily_export") return "OE daily export";
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
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
  emptyLabel: string;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">{emptyLabel}</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function ChampionThumb({ name, imageUrl }: { name: string; imageUrl?: string | null }) {
  const [imageFailed, setImageFailed] = useState(false);
  return (
    <span className={styles.championThumb} aria-hidden="true">
      {imageUrl && !imageFailed ? (
        // These are small square assets. Direct loading keeps the board independent of image optimization.
        // eslint-disable-next-line @next/next/no-img-element
        <img src={imageUrl} alt="" loading="lazy" onError={() => setImageFailed(true)} />
      ) : (
        <span>{name.slice(0, 1)}</span>
      )}
    </span>
  );
}

function evidenceLabel(row: TierRow): string {
  if (row.counterability_status === "available") return "Supported matchup sample";
  if (row.matchup_profile?.some((matchup) => matchup.evidence_status === "supported")) {
    return "Some supported matchups";
  }
  return "Thin matchup sample";
}

function SummaryCard({
  label,
  description,
  row,
  value,
}: {
  label: string;
  description: string;
  row?: TierRow;
  value: string;
}) {
  return (
    <article className={styles.summaryCard}>
      <p className={styles.cardLabel}>{label}</p>
      {row ? (
        <div className={styles.summaryChampion}>
          <ChampionThumb name={row.champion} imageUrl={row.champion_image_url} />
          <div>
            <strong>{row.champion}</strong>
            <span>{roleLabel(row.role)} · #{row.rank}</span>
          </div>
        </div>
      ) : (
        <strong className={styles.summaryUnavailable}>Unavailable</strong>
      )}
      <strong className={styles.summaryValue}>{value}</strong>
      <p className={styles.summaryDescription}>{description}</p>
    </article>
  );
}

function sortRows(rows: TierRow[], mode: BoardMode): TierRow[] {
  return [...rows].sort((left, right) => {
    if (mode === "blind") {
      const leftValue = left.blind_score_pp;
      const rightValue = right.blind_score_pp;
      if (leftValue === null && rightValue !== null) return 1;
      if (rightValue === null && leftValue !== null) return -1;
      if (leftValue !== null && rightValue !== null && rightValue !== leftValue) return rightValue - leftValue;
    }
    if (mode === "counter") {
      const leftCount = left.countered_opponent_count;
      const rightCount = right.countered_opponent_count;
      if (leftCount === null && rightCount !== null) return 1;
      if (rightCount === null && leftCount !== null) return -1;
      if (leftCount !== null && rightCount !== null && rightCount !== leftCount) return rightCount - leftCount;
      const leftScore = left.counter_score;
      const rightScore = right.counter_score;
      if (leftScore === null && rightScore !== null) return 1;
      if (rightScore === null && leftScore !== null) return -1;
      if (leftScore !== null && rightScore !== null && rightScore !== leftScore) return rightScore - leftScore;
    }
    return left.rank - right.rank;
  });
}

function listMetric(row: TierRow, mode: BoardMode): { value: string; detail: string } {
  if (mode === "blind") {
    return {
      value: signedScore(row.blind_score_pp),
      detail: evidenceLabel(row),
    };
  }
  if (mode === "counter") {
    return {
      value: row.countered_opponent_count === null ? "Unavailable" : `${row.countered_opponent_count} / 5`,
      detail: row.counter_score === null ? evidenceLabel(row) : `${row.counter_score.toFixed(1)} expected responses`,
    };
  }
  return {
    value: signedScore(row.tier_value_pp),
    detail: `${row.played_maps} observed maps · ${evidenceLabel(row)}`,
  };
}

function DraftRow({
  row,
  mode,
  onSelect,
}: {
  row: TierRow;
  mode: BoardMode;
  onSelect: (row: TierRow) => void;
}) {
  const metric = listMetric(row, mode);
  return (
    <button type="button" className={styles.draftRow} onClick={() => onSelect(row)}>
      <span className={styles.rowRank}>#{row.rank}</span>
      <ChampionThumb name={row.champion} imageUrl={row.champion_image_url} />
      <span className={styles.rowName}>
        <strong>{row.champion}</strong>
        <span>
          {roleLabel(row.role)} · {row.played_maps} maps · <span className={movementClass(row)}>{movementText(row)}</span>
        </span>
      </span>
      <span className={styles.rowMetric}>
        <strong>{metric.value}</strong>
        <span>{metric.detail}</span>
      </span>
    </button>
  );
}

function BoardList({
  rows,
  mode,
  onSelect,
}: {
  rows: TierRow[];
  mode: BoardMode;
  onSelect: (row: TierRow) => void;
}) {
  const sorted = useMemo(() => sortRows(rows, mode), [mode, rows]);
  return (
    <div className={styles.list}>
      {sorted.map((row) => (
        <DraftRow key={`${row.role}|${row.champion_id}`} row={row} mode={mode} onSelect={onSelect} />
      ))}
    </div>
  );
}

function RoleSnapshotGrid({ rows, onSelect }: { rows: TierRow[]; onSelect: (row: TierRow) => void }) {
  return (
    <div className={styles.roleGrid}>
      {ROLE_ORDER.map((role) => {
        const roleRows = rows.filter((row) => row.role === role).slice(0, 5);
        return (
          <article className={styles.roleCard} key={role}>
            <header>
              <p className={styles.cardLabel}>{roleLabel(role)}</p>
              <span>top overall strength</span>
            </header>
            <div className={styles.roleRows}>
              {roleRows.map((row) => (
                <button type="button" className={styles.roleRow} key={row.champion_id} onClick={() => onSelect(row)}>
                  <span>#{row.rank}</span>
                  <ChampionThumb name={row.champion} imageUrl={row.champion_image_url} />
                  <strong>{row.champion}</strong>
                  <em>{signedScore(row.tier_value_pp)}</em>
                </button>
              ))}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function ResponseBoard({
  rows,
  targetId,
  onTargetChange,
}: {
  rows: TierRow[];
  targetId: string;
  onTargetChange: (value: string) => void;
}) {
  const targets = useMemo(() => {
    const byId = new Map<string, string>();
    for (const row of rows) {
      for (const matchup of row.matchup_profile ?? []) byId.set(matchup.champion_id, matchup.champion);
    }
    return [...byId.entries()].sort((left, right) => left[1].localeCompare(right[1]));
  }, [rows]);
  const selectedTarget = targets.some(([id]) => id === targetId) ? targetId : targets[0]?.[0] || "";
  const targetName = targets.find(([id]) => id === selectedTarget)?.[1] ?? "the selected champion";
  const responses = rows
    .flatMap((row) => {
      const matchup = row.matchup_profile?.find((candidate) => candidate.champion_id === selectedTarget);
      return matchup ? [{ row, matchup }] : [];
    })
    .sort((left, right) => right.matchup.model_edge_pp - left.matchup.model_edge_pp);

  return (
    <section className={styles.questionPanel}>
      <div className={styles.questionControls}>
        <label className={styles.field}>
          <span>Enemy champion</span>
          <select value={selectedTarget} onChange={(event) => onTargetChange(event.target.value)}>
            {!targets.length ? <option value="">No response data</option> : null}
            {targets.map(([id, name]) => (
              <option key={id} value={id}>{name}</option>
            ))}
          </select>
        </label>
        <p>Responses use the same patch-wide model. The edge is a model comparison, not a raw win rate.</p>
      </div>
      {responses.length ? (
        <div className={styles.responseList}>
          <div className={styles.panelHeading}>
            <div>
              <p className={styles.cardLabel}>Best responses to</p>
              <h2>{targetName}</h2>
            </div>
            <span>{responses.length} modeled responses</span>
          </div>
          {responses.map(({ row, matchup }) => (
            <div className={styles.responseRow} key={`${row.role}|${row.champion_id}`}>
              <span className={styles.rowRank}>#{row.rank}</span>
              <ChampionThumb name={row.champion} imageUrl={row.champion_image_url} />
              <span className={styles.rowName}>
                <strong>{row.champion}</strong>
                <span>{matchup.evidence_status === "supported" ? "Supported pair" : "Limited pair"} · {matchup.effective_maps.toFixed(1)} effective maps</span>
              </span>
              <span className={styles.rowMetric}>
                <strong>{signedScore(matchup.model_edge_pp)}</strong>
                <span>{matchup.posterior_interval_pp.low.toFixed(1)} to {matchup.posterior_interval_pp.high.toFixed(1)} pp interval</span>
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className={styles.unavailable}>
          <p>No accepted response evidence is available for this role and patch.</p>
          <span>The page keeps the result unavailable until the pair sample passes its evidence checks.</span>
        </div>
      )}
    </section>
  );
}

function RegionalBoard({
  scope,
  rows,
  regionId,
  onRegionChange,
}: {
  scope?: Scope;
  rows: TierRow[];
  regionId: string;
  onRegionChange: (value: string) => void;
}) {
  const views = scope?.regional_views ?? [];
  const selectedRegion = views.some((candidate) => candidate.id === regionId) ? regionId : views[0]?.id || "";
  const view = views.find((candidate) => candidate.id === selectedRegion);
  const imageById = new Map(rows.map((row) => [row.champion_id, row.champion_image_url]));
  return (
    <section className={styles.questionPanel}>
      <div className={styles.questionControls}>
        <label className={styles.field}>
          <span>Region or league</span>
          <select value={selectedRegion} onChange={(event) => onRegionChange(event.target.value)}>
            {!views.length ? <option value="">No regional view</option> : null}
            {views.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>{candidate.label}</option>
            ))}
          </select>
        </label>
        <p>The patch-wide fit stays fixed. This view changes the observed league pool and shows its sample size.</p>
      </div>
      {view ? (
        <div className={styles.responseList}>
          <div className={styles.panelHeading}>
            <div>
              <p className={styles.cardLabel}>{view.label}</p>
              <h2>Strongest observed picks</h2>
            </div>
            <span>{view.maps} maps in this patch</span>
          </div>
          {view.rows.map((regionalRow) => (
            <div className={styles.responseRow} key={regionalRow.champion_id}>
              <span className={styles.rowRank}>#{regionalRow.regional_rank}</span>
              <ChampionThumb name={regionalRow.champion} imageUrl={imageById.get(regionalRow.champion_id)} />
              <span className={styles.rowName}>
                <strong>{regionalRow.champion}</strong>
                <span>Patch-wide rank #{regionalRow.global_rank} · {regionalRow.played_maps} maps in {view.label}</span>
              </span>
              <span className={styles.rowMetric}>
                <strong>{signedScore(regionalRow.strength_score_pp)}</strong>
                <span>{regionalRow.sample_status === "thin" ? "Thin regional sample" : "Observed regional sample"}</span>
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className={styles.unavailable}>
          <p>No regional context is available for this patch and role.</p>
          <span>The canonical board remains patch-wide.</span>
        </div>
      )}
    </section>
  );
}

export function TierListExplorer() {
  const [role, setRole] = useState("");
  const [patch, setPatch] = useState("");
  const [mode, setMode] = useState<BoardMode>("first_pick");
  const [responseChampion, setResponseChampion] = useState("");
  const [region, setRegion] = useState("");
  const [data, setData] = useState<TierResponse>(EMPTY);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async (signal?: AbortSignal, showLoading = false) => {
    if (showLoading) setLoading(true);
    try {
      const response = await fetch(TIER_LIST_URL, { signal });
      if (!response.ok) throw new Error("tier-list file is unavailable");
      setData((await response.json()) as TierResponse);
    } catch {
      if (!signal?.aborted) setData(EMPTY);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch(TIER_LIST_URL, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("tier-list file is unavailable");
        return response.json() as Promise<TierResponse>;
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
        <p>Patch-wide draft boards are waiting for an accepted local rebuild.</p>
        <span>{data.reason ?? "The previous league-specific boards are retired."}</span>
      </div>
    );
  }

  const scopes = data.scopes ?? [];
  const patchOptions = [...new Set(data.options?.patches ?? scopes.map((scope) => scope.patch))]
    .sort((left, right) => patchOrder(right) - patchOrder(left));
  const activePatch = patch || patchOptions[0] || "";
  const roleOptions = (data.options?.roles ?? [...ROLE_ORDER]).map((value) => ({ value, label: roleLabel(value) }));
  const rows = (data.rows ?? []).filter((row) => row.patch === activePatch);
  const selectedRows = role ? rows.filter((row) => row.role === role) : [];
  const selectedScope = role
    ? scopes.find((scope) => scope.patch === activePatch && scope.role === role)
    : undefined;
  const topRows = role ? selectedRows : rows;
  const firstPick = [...topRows].sort((left, right) => left.rank - right.rank)[0];
  const blindPick = [...topRows]
    .filter((row) => row.blind_score_pp !== null)
    .sort((left, right) => (right.blind_score_pp ?? -Infinity) - (left.blind_score_pp ?? -Infinity))[0];
  const counterPick = [...topRows]
    .filter((row) => row.countered_opponent_count !== null)
    .sort((left, right) => (right.countered_opponent_count ?? -Infinity) - (left.countered_opponent_count ?? -Infinity))[0];

  return (
    <section className={styles.section}>
      <div className={styles.filters}>
        <Select
          label="Patch"
          value={patch}
          options={patchOptions.map((value) => ({ value, label: value }))}
          onChange={setPatch}
          emptyLabel={`latest (${activePatch})`}
        />
        <Select
          label="Role"
          value={role}
          options={roleOptions}
          onChange={(value) => {
            setRole(value);
            setResponseChampion("");
            setRegion("");
          }}
          emptyLabel="all roles"
        />
        <button className={styles.button} onClick={() => void refresh(undefined, true)} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      <div className={styles.meta}>
        {role ? `${selectedRows.length} champions · ${roleLabel(role)}` : `${rows.length} champions across all roles`} · {freshnessLabel(data)} · updated {data.as_of ?? data.generated_at}
      </div>

      <nav className={styles.questionNav} aria-label="Draft questions">
        {BOARD_MODES.map((item) => (
          <button
            key={item.value}
            type="button"
            className={mode === item.value ? styles.questionTabActive : styles.questionTab}
            aria-pressed={mode === item.value}
            onClick={() => setMode(item.value)}
          >
            <strong>{item.label}</strong>
            <span>{item.note}</span>
          </button>
        ))}
      </nav>

      <div className={styles.summaryGrid}>
        <SummaryCard
          label="Top first-pick strength"
          row={firstPick}
          value={signedScore(firstPick?.tier_value_pp)}
          description="Overall patch-wide strength in the selected role."
        />
        <SummaryCard
          label="Safest blind"
          row={blindPick}
          value={signedScore(blindPick?.blind_score_pp)}
          description="Lower-tail matchup stability across the accepted opponent pool."
        />
        <SummaryCard
          label="Widest counter reach"
          row={counterPick}
          value={counterPick?.countered_opponent_count === null || counterPick?.countered_opponent_count === undefined ? "Unavailable" : `${counterPick.countered_opponent_count} / 5`}
          description="Modeled positive responses across five legal opponents."
        />
      </div>

      {mode === "responses" && !role ? (
        <div className={styles.unavailable}>
          <p>Choose a role to inspect champion responses.</p>
          <span>The response question compares champions within one role.</span>
        </div>
      ) : null}

      {mode === "regions" && !role ? (
        <div className={styles.unavailable}>
          <p>Choose a role to inspect regional context.</p>
          <span>The regional view keeps one role and one patch together.</span>
        </div>
      ) : null}

      {mode === "responses" && role ? (
        <ResponseBoard rows={selectedRows} targetId={responseChampion} onTargetChange={setResponseChampion} />
      ) : null}

      {mode === "regions" && role ? (
        <RegionalBoard scope={selectedScope} rows={selectedRows} regionId={region} onRegionChange={setRegion} />
      ) : null}

      {mode !== "responses" && mode !== "regions" ? (
        role ? (
          <section className={styles.boardPanel}>
            <header className={styles.panelHeading}>
              <div>
                <p className={styles.cardLabel}>{roleLabel(role)} · Patch {activePatch}</p>
                <h2>{mode === "blind" ? "Blind stability" : mode === "counter" ? "Counter reach" : "First-pick board"}</h2>
              </div>
              <span>{selectedRows.length} champions</span>
            </header>
            <BoardList rows={selectedRows} mode={mode} onSelect={() => {
              setResponseChampion("");
              setMode("responses");
            }} />
          </section>
        ) : mode === "first_pick" ? (
          <RoleSnapshotGrid rows={rows} onSelect={(row) => {
            setRole(row.role);
            setResponseChampion("");
          }} />
        ) : (
          <div className={styles.unavailable}>
            <p>Choose a role to inspect this question.</p>
            <span>Blind stability and counter reach compare champions within one role.</span>
          </div>
        )
      ) : null}

      <div className={styles.methodNote}>
        <strong>How to read this board</strong>
        <span>First pick uses the overall patch-wide model. Blind and counter views use matchup-shape fields only when their evidence checks pass. Regional context filters observed appearances while keeping the same patch-wide fit.</span>
      </div>
    </section>
  );
}
