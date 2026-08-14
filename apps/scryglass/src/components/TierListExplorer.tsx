"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  filterRowsByMinimumGames,
  filterRowsByRegion,
  firstPickMetric,
  matchupGrade,
  regionalOptions,
  rowsForMode,
  TIER_ROLE_ORDER,
  viableCandidates,
  type TierBoardMode,
  type TierBucket,
  type ResponseMatrix,
  type StructuralSimilarity,
  type TierRow,
  type TierRankedMode,
  type TierScope,
} from "@/lib/tierBoard";
import { publicPatchLabel } from "@/lib/patchIdentity";
import styles from "./TierListExplorer.module.css";

const TIER_LIST_URL = "/api/public-data/tierlists";
const LATEST_TIER_LIST_URL = `${TIER_LIST_URL}?view=latest`;
const ROLE_ORDER = TIER_ROLE_ORDER;
const ROLE_LABELS: Record<string, string> = {
  top: "Top",
  jungle: "Jungle",
  mid: "Mid",
  bot: "Bot",
  support: "Support",
};
const BOARD_MODES = [
  { value: "first_pick", label: "First pick", note: "best general pick" },
  { value: "blind", label: "Blind", note: "safest worst case" },
  { value: "counter", label: "Good into", note: "widest matchup edge" },
  { value: "responses", label: "Matchup matrix", note: "pick-by-pick answers" },
  { value: "unpicked", label: "Unpicked, but viable", note: "similar job, no games" },
] as const;
const MINIMUM_GAME_OPTIONS = [1, 3, 5, 10, 20] as const;
type BoardMode = TierBoardMode;
type RankedBoardMode = TierRankedMode;
type MatrixSize = "overview" | "standard" | "large";
type MatchupGrade = ReturnType<typeof matchupGrade>;
type MatchupBasis = NonNullable<ResponseMatrix["basis"]>[number][number];
type MatchupTooltip = {
  x: number;
  y: number;
  response: string;
  enemy: string;
  responseShare: number;
  enemyShare: number;
  grade: MatchupGrade;
  evidence: "supported" | "limited" | null | undefined;
  maps: number | null | undefined;
  basis: MatchupBasis | undefined;
};

export type TierResponse = {
  status: string;
  reason?: string;
  generated_at?: string;
  as_of?: string;
  source_freshness?: "oe_daily_export" | "oe_with_same_day_grid_bridge";
  options?: {
    roles: string[];
    patches: string[];
    regions?: string[];
    leagues?: string[];
    tiers?: string[];
    tier_buckets?: TierBucket[];
  };
  scopes?: TierScope[];
  rows?: TierRow[];
  structural_similarity?: StructuralSimilarity;
  champion_images?: Record<string, string>;
};

export type TierFilterState = {
  patch: string;
  role: string;
  region: string;
  league: string;
  tier: string;
  minimumGames: number;
};

const EMPTY: TierResponse = { status: "unavailable" };

function patchOrder(value: string): number {
  const [major, minor] = value.split(".").map(Number);
  return (Number.isFinite(major) ? major : 0) * 1000 + (Number.isFinite(minor) ? minor : 0);
}

function publicPatchScopeId(scopeId: string | undefined, sourcePatch: string, patch: string): string | undefined {
  if (!scopeId) return scopeId;
  const prefix = `patch:${sourcePatch}`;
  return scopeId.startsWith(prefix) ? `patch:${patch}${scopeId.slice(prefix.length)}` : scopeId;
}

function normalizeTierResponse(payload: TierResponse): TierResponse {
  const options = payload.options
    ? { ...payload.options, patches: payload.options.patches.map(publicPatchLabel) }
    : payload.options;
  const rows = payload.rows?.map((row) => {
    const sourcePatch = row.patch;
    const patch = publicPatchLabel(sourcePatch);
    return {
      ...row,
      patch,
      scope_id: publicPatchScopeId(row.scope_id, sourcePatch, patch) ?? row.scope_id,
    };
  });
  const scopes = payload.scopes?.map((scope) => {
    const sourcePatch = scope.patch;
    const patch = publicPatchLabel(sourcePatch);
    return {
      ...scope,
      patch,
      scope_id: publicPatchScopeId(scope.scope_id, sourcePatch, patch) ?? scope.scope_id,
    };
  });
  return { ...payload, options, rows, scopes };
}

function roleLabel(role: string): string {
  return ROLE_LABELS[role] ?? role;
}

function gameCount(value: number): string {
  return `${value} ${value === 1 ? "game" : "games"}`;
}

function signedScore(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Pending";
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function winShare(value: number): number {
  return Math.max(0, Math.min(100, 50 + value));
}

function matchupEvidence(
  evidence: "supported" | "limited" | null | undefined,
  maps: number | null | undefined,
  basis?: MatchupBasis,
): { label: string; detail: string } {
  if (basis === "atom_and_strength_inferred") {
    return { label: "Atom + archetype estimate", detail: "Champion strength included" };
  }
  if (basis === "strength_only_inferred") {
    return { label: "Strength-only estimate", detail: "No direct or atom matchup" };
  }
  if (!maps || maps < 0.05) return { label: "Inferred matchup", detail: "Basis refresh pending" };
  if (basis === "observed_pair_plus_model" || evidence === "supported") {
    return { label: "Direct + model", detail: `${maps.toFixed(1)} weighted games` };
  }
  return { label: "Thin sample", detail: `${maps.toFixed(1)} weighted games` };
}

function numericMetric(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function tierLevel(tier: TierBucket): number {
  if (tier.startsWith("Z") || tier.startsWith("S") || tier === "A") return 4;
  if (tier === "B") return 3;
  if (tier === "C") return 2;
  return 1;
}

function TierRail({ tier, compact = false }: { tier: TierBucket; compact?: boolean }) {
  const level = tierLevel(tier);
  return (
    <span className={compact ? styles.tierRailCompact : styles.tierRail} aria-hidden="true">
      {[1, 2, 3, 4].map((step) => (
        <i className={step <= level ? styles.tierStepActive : styles.tierStep} key={step} />
      ))}
    </span>
  );
}

function SignedRail({ value, compact = false }: { value: number | null | undefined; compact?: boolean }) {
  const metric = numericMetric(value);
  if (metric === null) return null;
  const width = Math.max(3, Math.min(50, Math.abs(metric) / 8 * 50));
  const style = { "--metric-width": `${width}%` } as CSSProperties;
  return (
    <span
      className={compact ? styles.signedRailCompact : styles.signedRail}
      data-direction={metric >= 0 ? "positive" : "negative"}
      style={style}
      aria-hidden="true"
    >
      <i />
    </span>
  );
}

function CountRail({ value, compact = false }: { value: number | null | undefined; compact?: boolean }) {
  const metric = numericMetric(value);
  if (metric === null) return null;
  const width = Math.max(3, Math.min(100, metric / 5 * 100));
  const style = { "--metric-width": `${width}%` } as CSSProperties;
  return (
    <span className={compact ? styles.countRailCompact : styles.countRail} style={style} aria-hidden="true">
      <i />
    </span>
  );
}

function MetricRail({ row, mode, compact = false }: { row: TierRow; mode: RankedBoardMode; compact?: boolean }) {
  if (mode === "blind") return <SignedRail value={row.blind_score_pp} compact={compact} />;
  if (mode === "counter") return <CountRail value={row.countered_opponent_count} compact={compact} />;
  return <TierRail tier={row.tier_bucket} compact={compact} />;
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

function TierLoadingState() {
  return (
    <div className={styles.loadingState} role="status" aria-live="polite">
      <div className={styles.loadingHeader}>
        <span>Loading accepted tier artifact</span>
        <i aria-hidden="true" />
      </div>
      <div className={styles.loadingRows} aria-hidden="true">
        {Array.from({ length: 5 }, (_, index) => <span key={index} style={{ "--loading-width": `${62 + index * 7}%` } as CSSProperties} />)}
      </div>
      <small>Preparing the patch, role, and evidence controls.</small>
    </div>
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
  if (row.matchup_opponents > 0) return "Some matchup evidence";
  return "Thin matchup sample";
}

function SummaryCard({
  label,
  description,
  row,
  value,
  mode,
}: {
  label: string;
  description: string;
  row?: TierRow;
  value: string;
  mode: RankedBoardMode;
}) {
  return (
    <article className={`${styles.summaryCard} ${row ? "" : styles.summaryCardPending}`}>
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
        <strong className={styles.summaryUnavailable}>{value}</strong>
      )}
      {row ? <strong className={styles.summaryValue}>{value}</strong> : null}
      {row ? <MetricRail row={row} mode={mode} /> : null}
      <p className={styles.summaryDescription}>{description}</p>
    </article>
  );
}

function sortRows(rows: TierRow[], mode: RankedBoardMode): TierRow[] {
  return [...rows].sort((left, right) => {
    if (mode === "blind") {
      const leftValue = numericMetric(left.blind_score_pp);
      const rightValue = numericMetric(right.blind_score_pp);
      if (leftValue === null && rightValue !== null) return 1;
      if (rightValue === null && leftValue !== null) return -1;
      if (leftValue !== null && rightValue !== null && rightValue !== leftValue) return rightValue - leftValue;
    }
    if (mode === "counter") {
      const leftCount = numericMetric(left.countered_opponent_count);
      const rightCount = numericMetric(right.countered_opponent_count);
      if (leftCount === null && rightCount !== null) return 1;
      if (rightCount === null && leftCount !== null) return -1;
      if (leftCount !== null && rightCount !== null && rightCount !== leftCount) return rightCount - leftCount;
      const leftScore = numericMetric(left.counter_score);
      const rightScore = numericMetric(right.counter_score);
      if (leftScore === null && rightScore !== null) return 1;
      if (rightScore === null && leftScore !== null) return -1;
      if (leftScore !== null && rightScore !== null && rightScore !== leftScore) return rightScore - leftScore;
    }
    return left.rank - right.rank;
  });
}

function listMetric(row: TierRow, mode: RankedBoardMode): { value: string; detail: string } {
  if (mode === "blind") {
    return {
      value: signedScore(row.blind_score_pp),
      detail: evidenceLabel(row),
    };
  }
  if (mode === "counter") {
    return {
      value: row.countered_opponent_count === null || row.countered_opponent_count === undefined
        ? "Pending"
        : `${row.countered_opponent_count} / 5`,
      detail: row.countered_opponent_count === null || row.countered_opponent_count === undefined
        ? evidenceLabel(row)
        : "common opponents with a positive edge",
    };
  }
  return {
    value: firstPickMetric(row),
    detail: `${gameCount(row.played_maps)} observed · patch-wide tier`,
  };
}

function compactMetricValue(row: TierRow, mode: RankedBoardMode): string {
  if (mode === "first_pick") return signedScore(row.tier_value_pp);
  return listMetric(row, mode).value;
}

function DraftRow({
  row,
  mode,
  onSelect,
}: {
  row: TierRow;
  mode: RankedBoardMode;
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
          {roleLabel(row.role)} · {gameCount(row.played_maps)} · <span className={movementClass(row)}>{movementText(row)}</span>
        </span>
      </span>
      <span className={styles.rowMetric}>
        <strong>{metric.value}</strong>
        <MetricRail row={row} mode={mode} />
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
  mode: RankedBoardMode;
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

function RoleBoardGrid({
  rows,
  mode,
  onSelect,
}: {
  rows: TierRow[];
  mode: RankedBoardMode;
  onSelect: (row: TierRow) => void;
}) {
  const title = mode === "blind"
    ? "Safest blind picks by role"
    : mode === "counter"
      ? "Widest matchup edges by role"
      : "Five-role draft sheet";
  return (
    <section className={styles.roleSnapshot}>
      <header className={styles.roleSnapshotHeader}>
        <div>
          <p className={styles.cardLabel}>Patch-wide role comparison</p>
          <h2>{title}</h2>
        </div>
        {mode === "first_pick" ? (
          <div className={styles.tierLegend} aria-label="Tier scale from strongest to developing">
            <span><i className={styles.legendA} />A</span>
            <span><i className={styles.legendB} />B</span>
            <span><i className={styles.legendC} />C</span>
            <span><i className={styles.legendD} />D</span>
          </div>
        ) : null}
      </header>
      <div className={styles.roleGrid}>
        {ROLE_ORDER.map((role) => {
          const roleRows = sortRows(
            rowsForMode(rows.filter((row) => row.role === role), mode),
            mode,
          ).slice(0, 5);
          return (
            <article className={styles.roleCard} data-role={role} key={role}>
              <header>
                <p className={styles.cardLabel}>{roleLabel(role)}</p>
                <span>top five</span>
              </header>
              <div className={styles.roleRows}>
                {roleRows.map((row) => (
                  <button type="button" className={styles.roleRow} key={row.champion_id} onClick={() => onSelect(row)}>
                    <span>#{row.rank}</span>
                    <ChampionThumb name={row.champion} imageUrl={row.champion_image_url} />
                    <strong>{row.champion}</strong>
                    <span className={styles.roleMetric}>
                      <em title={listMetric(row, mode).value}>{compactMetricValue(row, mode)}</em>
                      <MetricRail row={row} mode={mode} compact />
                    </span>
                  </button>
                ))}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function traitLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function similarityLabel(value: number): string {
  if (value >= 0.9) return "Very close";
  if (value >= 0.84) return "Close";
  return "Related";
}

function UnpickedBoard({
  library,
  patchRoleRows,
  referenceRows,
  role,
  patch,
  targetId,
  onTargetChange,
}: {
  library?: StructuralSimilarity;
  patchRoleRows: TierRow[];
  referenceRows: TierRow[];
  role: string;
  patch: string;
  targetId: string;
  onTargetChange: (value: string) => void;
}) {
  const references = useMemo(
    () => [...referenceRows].sort((left, right) => left.champion.localeCompare(right.champion)),
    [referenceRows],
  );
  const selectedTarget = references.some((row) => row.champion_id === targetId) ? targetId : "";
  const candidates = useMemo(
    () => viableCandidates(library, patchRoleRows, referenceRows, role, selectedTarget),
    [library, patchRoleRows, referenceRows, role, selectedTarget],
  );

  if (!library) {
    return (
      <section className={styles.viablePanel}>
        <div className={styles.unavailable}>
          <p>Structural alternatives are waiting for the next accepted tier artifact.</p>
          <span>The performance boards remain available.</span>
        </div>
      </section>
    );
  }

  return (
    <section className={styles.viablePanel}>
      <div className={styles.questionControls}>
        <label className={styles.field}>
          <span>Played reference</span>
          <select value={selectedTarget} onChange={(event) => onTargetChange(event.target.value)}>
            <option value="">all played champions</option>
            {references.map((reference) => (
              <option key={reference.champion_id} value={reference.champion_id}>{reference.champion}</option>
            ))}
          </select>
        </label>
        <p>These champions have zero accepted {roleLabel(role)} picks in patch {patch}. Similarity covers role, function, and mechanic structure.</p>
      </div>
      <header className={styles.viableHeading}>
        <div>
          <p className={styles.cardLabel}>{roleLabel(role)} · Patch {patch}</p>
          <h2>Unpicked structural alternatives</h2>
        </div>
        <span>{candidates.length} candidates</span>
      </header>
      {candidates.length ? (
        <div className={styles.viableList}>
          {candidates.slice(0, 40).map((item) => (
            <article className={styles.viableCard} key={item.candidate.champion_id}>
              <div className={styles.viableChampion}>
                <ChampionThumb name={item.candidate.champion} imageUrl={item.candidate.champion_image_url} />
                <span>
                  <strong>{item.candidate.champion}</strong>
                  <small>0 accepted games</small>
                </span>
              </div>
              <div
                className={styles.similarityBridge}
                style={{ "--similarity-position": `${Math.round(item.similarity * 100)}%` } as CSSProperties}
              >
                <strong>{similarityLabel(item.similarity)}</strong>
                <span className={styles.similarityTrack} aria-hidden="true"><i /></span>
                <span className={styles.similarityCaption}>{Math.round(item.similarity * 100)}% structural similarity</span>
              </div>
              <div className={styles.viableReference}>
                <span>
                  <small>Closest played</small>
                  <strong>{item.reference.champion}</strong>
                </span>
                <ChampionThumb name={item.reference.champion} imageUrl={item.reference.champion_image_url} />
              </div>
              <div className={styles.viableTraits}>
                {item.sharedRoles.slice(0, 2).map((value) => <span key={`role-${value}`}>{value}</span>)}
                {item.sharedTraits.slice(0, 3).map((trait) => (
                  <span key={`${trait.dimension}-${trait.label}`}>{traitLabel(trait.label)}</span>
                ))}
                <em>{item.candidate.profile_status === "atom_detail" ? "detailed atom profile" : "family profile"}</em>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className={styles.unavailable}>
          <p>{references.length ? "No unpicked champion is structurally close enough for this view." : "No played reference is available for this region and role."}</p>
          <span>Try all played champions, another role, or another region.</span>
        </div>
      )}
      <p className={styles.viableCaveat}>This view identifies plausible functional substitutes. It does not estimate their strength, win rate, or draft value.</p>
    </section>
  );
}

function ResponseBoard({
  matrix,
  rows,
  targetId,
  onTargetChange,
}: {
  matrix?: ResponseMatrix;
  rows: TierRow[];
  targetId: string;
  onTargetChange: (value: string) => void;
}) {
  const [matrixSize, setMatrixSize] = useState<MatrixSize>("overview");
  const [tooltip, setTooltip] = useState<MatchupTooltip | null>(null);
  const tooltipTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const allowedIds = useMemo(() => new Set(rows.map((row) => row.champion_id)), [rows]);
  const champions = useMemo(
    () => (matrix?.champions ?? []).filter((champion) => allowedIds.has(champion.champion_id)),
    [allowedIds, matrix],
  );
  const selectedTarget = champions.some((champion) => champion.champion_id === targetId) ? targetId : "";
  const columns = selectedTarget
    ? champions.filter((champion) => champion.champion_id === selectedTarget)
    : champions;
  const matrixIndex = new Map((matrix?.champions ?? []).map((champion, index) => [champion.champion_id, index]));
  const imageById = new Map(rows.map((row) => [row.champion_id, row.champion_image_url]));
  const rowById = new Map(rows.map((row) => [row.champion_id, row]));

  const gradeClass = {
    S: styles.matchupGradeS,
    A: styles.matchupGradeA,
    B: styles.matchupGradeB,
    C: styles.matchupGradeC,
    D: styles.matchupGradeD,
  } as const;

  const hideTooltip = useCallback(() => {
    if (tooltipTimer.current) clearTimeout(tooltipTimer.current);
    tooltipTimer.current = null;
    setTooltip(null);
  }, []);

  const queueTooltip = useCallback((
    event: ReactPointerEvent<HTMLTableCellElement>,
    details: Omit<MatchupTooltip, "x" | "y">,
  ) => {
    if (tooltipTimer.current) clearTimeout(tooltipTimer.current);
    setTooltip(null);
    const rect = event.currentTarget.getBoundingClientRect();
    const cardWidth = Math.min(272, window.innerWidth - 24);
    const cardHeight = 170;
    const x = Math.max(12, Math.min(window.innerWidth - cardWidth - 12, rect.left + rect.width / 2 - cardWidth / 2));
    const y = rect.bottom + cardHeight + 12 <= window.innerHeight
      ? rect.bottom + 8
      : Math.max(12, rect.top - cardHeight - 8);
    tooltipTimer.current = setTimeout(() => setTooltip({ ...details, x, y }), 1500);
  }, []);

  useEffect(() => () => {
    if (tooltipTimer.current) clearTimeout(tooltipTimer.current);
  }, []);

  return (
    <section className={styles.matchupMatrixPanel}>
      <div className={styles.questionControls}>
        <label className={styles.field}>
          <span>Enemy champion</span>
          <select value={selectedTarget} onChange={(event) => onTargetChange(event.target.value)}>
            <option value="">all enemy champions</option>
            {champions.map((champion) => (
              <option key={champion.champion_id} value={champion.champion_id}>{champion.champion}</option>
            ))}
          </select>
        </label>
        <p>Read across your champion&apos;s row. Each column is the enemy pick. Hover a cell for 1.5 seconds to see both modeled win shares and the grade basis.</p>
      </div>
      {matrix && champions.length ? (
        <>
          <div className={styles.matrixToolbar}>
            <div className={styles.matchupLegend} aria-label="Matchup grade scale">
              <span className={styles.matchupGradeS}>S <small>strong counter</small></span>
              <span className={styles.matchupGradeA}>A <small>good response</small></span>
              <span className={styles.matchupGradeB}>B <small>close</small></span>
              <span className={styles.matchupGradeC}>C <small>unfavorable</small></span>
              <span className={styles.matchupGradeD}>D <small>heavily countered</small></span>
            </div>
            <div className={styles.matchupBasisLegend} aria-label="Estimate basis">
              <span><i data-basis="observed_pair_plus_model" />Direct</span>
              <span><i data-basis="atom_and_strength_inferred" />Atoms</span>
              <span><i data-basis="strength_only_inferred" />Strength</span>
            </div>
            <div className={styles.matrixSizeControl} aria-label="Matrix size">
              <span>Matrix size</span>
              <div>
                {([
                  ["overview", "Overview"],
                  ["standard", "Standard"],
                  ["large", "Large"],
                ] as const).map(([value, label]) => (
                  <button
                    type="button"
                    className={matrixSize === value ? styles.matrixSizeButtonActive : styles.matrixSizeButton}
                    aria-pressed={matrixSize === value}
                    key={value}
                    onClick={() => setMatrixSize(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className={styles.matchupMatrixScroll} data-native-scroll onScroll={hideTooltip}>
            <table className={styles.matchupMatrix} data-matrix-size={matrixSize}>
              <thead>
                <tr>
                  <th scope="col" className={styles.matchupCorner}>
                    <span>Enemy pick →</span>
                    <strong>Your pick ↓</strong>
                  </th>
                  {columns.map((champion) => (
                    <th scope="col" key={champion.champion_id} title={`Enemy pick: ${champion.champion}`}>
                      <ChampionThumb name={champion.champion} imageUrl={imageById.get(champion.champion_id)} />
                      <span className={styles.enemyChampionName}>{champion.champion}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {champions.map((response) => {
                  const responseIndex = matrixIndex.get(response.champion_id);
                  const responseRow = rowById.get(response.champion_id);
                  return (
                    <tr key={response.champion_id}>
                      <th scope="row" title={`Your pick: ${response.champion}`}>
                        <ChampionThumb name={response.champion} imageUrl={imageById.get(response.champion_id)} />
                        <span><strong>{response.champion}</strong><small>#{responseRow?.rank ?? "—"}</small></span>
                      </th>
                      {columns.map((enemy) => {
                        const enemyIndex = matrixIndex.get(enemy.champion_id);
                        if (responseIndex === undefined || enemyIndex === undefined || responseIndex === enemyIndex) {
                          return <td className={styles.matchupDiagonal} key={enemy.champion_id}>—</td>;
                        }
                        const edge = matrix.edge_pp[responseIndex]?.[enemyIndex];
                        const low = matrix.interval_low_pp[responseIndex]?.[enemyIndex];
                        const high = matrix.interval_high_pp[responseIndex]?.[enemyIndex];
                        const evidence = matrix.evidence[responseIndex]?.[enemyIndex];
                        const maps = matrix.effective_maps[responseIndex]?.[enemyIndex];
                        const basis = matrix.basis?.[responseIndex]?.[enemyIndex];
                        if (edge === null || edge === undefined) {
                          return <td className={styles.matchupDiagonal} key={enemy.champion_id}>—</td>;
                        }
                        const grade = matchupGrade(edge);
                        const evidenceCopy = matchupEvidence(evidence, maps, basis);
                        const detail = `${response.champion} into ${enemy.champion}: ${grade}, ${winShare(edge).toFixed(1)}% modeled win share. ${evidenceCopy.label}: ${evidenceCopy.detail}. Interval: ${low?.toFixed(1) ?? "—"} to ${high?.toFixed(1) ?? "—"} percentage points.`;
                        const tooltipDetails = {
                          response: response.champion,
                          enemy: enemy.champion,
                          responseShare: winShare(edge),
                          enemyShare: winShare(-edge),
                          grade,
                          evidence,
                          maps,
                          basis,
                        };
                        return (
                          <td
                            className={`${styles.matchupCell} ${gradeClass[grade]}`}
                            data-basis={basis ?? "pending"}
                            key={enemy.champion_id}
                            aria-label={detail}
                            onPointerEnter={(event) => queueTooltip(event, tooltipDetails)}
                            onPointerLeave={hideTooltip}
                          >
                            <strong>{grade}</strong>
                            <small>{signedScore(edge)}</small>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {tooltip ? (
            <aside
              className={styles.matchupTooltip}
              style={{ left: tooltip.x, top: tooltip.y }}
              role="tooltip"
            >
              <header>
                <strong>{tooltip.response}</strong>
                <span aria-hidden="true">→</span>
                <strong>{tooltip.enemy}</strong>
                <em className={gradeClass[tooltip.grade]}>{tooltip.grade}</em>
              </header>
              <div className={styles.tooltipWinLabels}>
                <span><strong>{tooltip.responseShare.toFixed(1)}%</strong><small>Your pick</small></span>
                <span><strong>{tooltip.enemyShare.toFixed(1)}%</strong><small>Enemy pick</small></span>
              </div>
              <div
                className={styles.tooltipWinBar}
                style={{ "--response-share": `${tooltip.responseShare}%` } as CSSProperties}
                aria-hidden="true"
              ><i /><i /></div>
              <footer>
                <strong>Why {tooltip.grade} · {tooltip.responseShare.toFixed(1)}% modeled WR</strong>
                <span>{matchupEvidence(tooltip.evidence, tooltip.maps, tooltip.basis).label} · {matchupEvidence(tooltip.evidence, tooltip.maps, tooltip.basis).detail}</span>
              </footer>
            </aside>
          ) : null}
        </>
      ) : (
        <div className={styles.unavailable}>
          <p>The matchup matrix is waiting for the next accepted tier artifact.</p>
          <span>First-pick, blind, and counter boards remain available.</span>
        </div>
      )}
    </section>
  );
}

export function TierListExplorer({
  initialData,
  initialFilters,
  serverFiltered = false,
}: {
  initialData?: TierResponse;
  initialFilters?: TierFilterState;
  serverFiltered?: boolean;
} = {}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [role, setRole] = useState(initialFilters?.role ?? "");
  const [patch, setPatch] = useState(publicPatchLabel(initialFilters?.patch ?? ""));
  const [minimumGames, setMinimumGames] = useState(initialFilters?.minimumGames ?? 5);
  const [mode, setMode] = useState<BoardMode>("first_pick");
  const [responseChampion, setResponseChampion] = useState("");
  const [region, setRegion] = useState(initialFilters?.region ?? "");
  const [league, setLeague] = useState(initialFilters?.league ?? "");
  const [tier, setTier] = useState(initialFilters?.tier ?? "");
  const [data, setData] = useState<TierResponse>(normalizeTierResponse(initialData ?? EMPTY));
  const [loading, setLoading] = useState(!initialData);
  const [fullHistoryLoaded, setFullHistoryLoaded] = useState(serverFiltered);

  const commitData = useCallback((payload: TierResponse, fullHistory: boolean) => {
    setData(normalizeTierResponse(payload));
    setFullHistoryLoaded(fullHistory);
  }, []);

  const load = useCallback(async (url: string, signal?: AbortSignal, showLoading = false) => {
    if (showLoading) setLoading(true);
    try {
      const response = await fetch(url, { signal });
      if (!response.ok) throw new Error("tier-list file is unavailable");
      commitData((await response.json()) as TierResponse, url === TIER_LIST_URL);
    } catch {
      if (!signal?.aborted) setData(EMPTY);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [commitData]);

  useEffect(() => {
    if (serverFiltered) return;
    const controller = new AbortController();
    fetch(LATEST_TIER_LIST_URL, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("tier-list file is unavailable");
        return response.json() as Promise<TierResponse>;
      })
      .then((payload) => {
        commitData(payload, false);
      })
      .catch(() => {
        if (!controller.signal.aborted) setData(EMPTY);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [commitData, serverFiltered]);

  useEffect(() => {
    if (!serverFiltered) return;
    const params = new URLSearchParams(searchParams.toString());
    if (patch) params.set("patch", patch); else params.delete("patch");
    if (role) params.set("role", role); else params.delete("role");
    if (region) params.set("region", region); else params.delete("region");
    if (league) params.set("league", league); else params.delete("league");
    if (tier) params.set("tier", tier); else params.delete("tier");
    if (minimumGames !== 5) params.set("min", String(minimumGames)); else params.delete("min");
    const suffix = params.toString();
    const next = `${pathname}${suffix ? `?${suffix}` : ""}`;
    const current = `${pathname}${searchParams.toString() ? `?${searchParams.toString()}` : ""}`;
    if (next !== current) router.replace(next, { scroll: false });
  }, [league, minimumGames, patch, pathname, region, role, router, searchParams, serverFiltered, tier]);

  if (loading && data.status !== "available") return <TierLoadingState />;
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
  const latestPatch = patchOptions[0] || "";
  const roleOptions = (data.options?.roles ?? [...ROLE_ORDER]).map((value) => ({ value, label: roleLabel(value) }));
  const rows = (data.rows ?? [])
    .filter((row) => row.patch === activePatch)
    .map((row) => ({
      ...row,
      champion_image_url: row.champion_image_url ?? data.champion_images?.[row.champion] ?? null,
    }));
  const scopeRegionOptions = regionalOptions(scopes, activePatch);
  const regionLabels = new Map(scopeRegionOptions.map((option) => [option.id, option.label]));
  const serverRegionIds = scopeRegionOptions.length
    ? scopeRegionOptions.map((option) => option.id)
    : data.options?.regions ?? [];
  const regionOptions = serverFiltered
    ? serverRegionIds.map((value) => ({
        id: value,
        label: regionLabels.get(value) ?? value,
      }))
    : scopeRegionOptions;
  const activeRegion = regionOptions.some((candidate) => candidate.id === region) ? region : "";
  const regionalRows = serverFiltered ? rows : filterRowsByRegion(rows, scopes, activePatch, activeRegion);
  const visibleRows = filterRowsByMinimumGames(regionalRows, minimumGames);
  const selectedRows = role ? visibleRows.filter((row) => row.role === role) : [];
  const selectedScope = role
    ? scopes.find((scope) => scope.patch === activePatch && scope.role === role)
    : undefined;
  const activeRegionLabel = regionOptions.find((candidate) => candidate.id === activeRegion)?.label;
  const topRows = role ? selectedRows : visibleRows;
  const firstPick = [...topRows].sort((left, right) => (
    (right.tier_value_pp ?? -Infinity) - (left.tier_value_pp ?? -Infinity)
  ))[0];
  const blindPick = rowsForMode(topRows, "blind")
    .sort((left, right) => (right.blind_score_pp ?? -Infinity) - (left.blind_score_pp ?? -Infinity))[0];
  const counterPick = rowsForMode(topRows, "counter")
    .sort((left, right) => (right.countered_opponent_count ?? -Infinity) - (left.countered_opponent_count ?? -Infinity))[0];
  const rankedMode: RankedBoardMode = mode === "blind" || mode === "counter" ? mode : "first_pick";
  const boardRows = rowsForMode(selectedRows, rankedMode);
  const patchRoleRows = role ? rows.filter((row) => row.role === role) : [];

  const changeMode = (value: BoardMode) => {
    if (value === mode) return;
    setMode(value);
  };

  return (
    <section className={styles.section} aria-busy={loading}>
      <div className={styles.filters}>
        <Select
          label="Patch"
          value={patch}
          options={patchOptions.map((value) => ({ value, label: publicPatchLabel(value) }))}
          onChange={(value) => {
            setPatch(value);
            setResponseChampion("");
            setRegion("");
            setLeague("");
            setTier("");
            if (value && value !== latestPatch && !fullHistoryLoaded) {
              void load(TIER_LIST_URL, undefined, true);
            }
          }}
          emptyLabel={latestPatch ? `latest (${publicPatchLabel(latestPatch)})` : "latest patch"}
        />
        <Select
          label="Role"
          value={role}
          options={roleOptions}
          onChange={(value) => {
            setRole(value);
            setResponseChampion("");
          }}
          emptyLabel="all roles"
        />
        <Select
          label="Region"
          value={activeRegion}
          options={regionOptions.map((candidate) => ({ value: candidate.id, label: candidate.label }))}
          onChange={(value) => {
            setRegion(value);
            setLeague("");
            setResponseChampion("");
          }}
          emptyLabel={regionOptions.length ? "all regions" : "regional refresh pending"}
        />
        {serverFiltered && data.options?.leagues?.length ? (
          <Select
            label="League"
            value={league}
            options={data.options.leagues.map((value) => ({ value, label: value }))}
            onChange={(value) => {
              setLeague(value);
              setRegion("");
              setResponseChampion("");
            }}
            emptyLabel="all leagues"
          />
        ) : null}
        {serverFiltered && data.options?.tiers?.length ? (
          <Select
            label="Tier"
            value={tier}
            options={data.options.tiers.map((value) => ({ value, label: value.replace(/^tier/, "Tier ") }))}
            onChange={(value) => {
              setTier(value);
              setResponseChampion("");
            }}
            emptyLabel="all tiers"
          />
        ) : null}
        <label className={styles.field}>
          <span>Minimum games</span>
          <select value={minimumGames} onChange={(event) => setMinimumGames(Number(event.target.value))}>
            {MINIMUM_GAME_OPTIONS.map((value) => (
              <option key={value} value={value}>{value}+</option>
            ))}
          </select>
        </label>
        <button
          className={styles.button}
          onClick={() => {
            if (serverFiltered) router.refresh();
            else void load(activePatch === latestPatch ? LATEST_TIER_LIST_URL : TIER_LIST_URL, undefined, true);
          }}
          disabled={loading}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      <div className={styles.meta}>
        {role ? `${selectedRows.length} champions · ${roleLabel(role)}` : `${visibleRows.length} champions across all roles`}
        {activeRegionLabel ? ` · ${activeRegionLabel}` : " · all regions"} · {minimumGames}+ games · {freshnessLabel(data)} · updated {data.as_of ?? data.generated_at}
      </div>

      <nav className={styles.questionNav} aria-label="Draft questions">
        {BOARD_MODES.map((item) => (
          <button
            key={item.value}
            type="button"
            className={mode === item.value ? styles.questionTabActive : styles.questionTab}
            aria-pressed={mode === item.value}
            aria-current={mode === item.value ? "page" : undefined}
            onClick={() => changeMode(item.value)}
          >
            <strong>{item.label}</strong>
            <span>{item.note}</span>
          </button>
        ))}
      </nav>

      {mode === "first_pick" ? (
        <div className={styles.activeSummary}>
          <SummaryCard
            label="Best first pick"
            row={firstPick}
            value={firstPickMetric(firstPick)}
            mode="first_pick"
            description={`Strongest general result among champions with at least ${minimumGames} accepted games.`}
          />
        </div>
      ) : null}

      {mode === "blind" ? (
        <div className={styles.activeSummary}>
          <SummaryCard
            label="Safest blind pick"
            row={blindPick}
            value={blindPick ? signedScore(blindPick.blind_score_pp) : "Matchup refresh pending"}
            mode="blind"
            description="Best expected result in the champion’s weakest common matchup."
          />
        </div>
      ) : null}

      {mode === "counter" ? (
        <div className={styles.activeSummary}>
          <SummaryCard
            label="Widest matchup edge"
            row={counterPick}
            value={counterPick?.countered_opponent_count === null || counterPick?.countered_opponent_count === undefined ? "Matchup refresh pending" : `${counterPick.countered_opponent_count} / 5`}
            mode="counter"
            description={counterPick ? `${counterPick.champion} is favored into ${counterPick.countered_opponent_count} of 5 common ${roleLabel(counterPick.role)} opponents.` : "Matchup refresh pending."}
          />
        </div>
      ) : null}

      {mode === "responses" && !role ? (
        <div className={styles.unavailable}>
          <p>Choose a role to inspect champion responses.</p>
          <span>The response question compares champions within one role.</span>
        </div>
      ) : null}

      {mode === "responses" && role ? (
        <ResponseBoard
          matrix={selectedScope?.response_matrix}
          rows={selectedRows}
          targetId={responseChampion}
          onTargetChange={setResponseChampion}
        />
      ) : null}

      {mode === "unpicked" && !role ? (
        <div className={styles.unavailable}>
          <p>Choose a role to find unpicked structural alternatives.</p>
          <span>Role compatibility is required for every candidate.</span>
        </div>
      ) : null}

      {mode === "unpicked" && role ? (
        <UnpickedBoard
          library={data.structural_similarity}
          patchRoleRows={patchRoleRows}
          referenceRows={selectedRows}
          role={role}
          patch={activePatch}
          targetId={responseChampion}
          onTargetChange={setResponseChampion}
        />
      ) : null}

      {mode !== "responses" && mode !== "unpicked" ? (
        role ? (
          <section className={styles.boardPanel}>
            <header className={styles.panelHeading}>
              <div>
                <p className={styles.cardLabel}>{roleLabel(role)} · Patch {activePatch}</p>
                <h2>{mode === "blind" ? "Blind stability" : mode === "counter" ? "Good into common picks" : "First-pick board"}</h2>
              </div>
              <span>{selectedRows.length} champions</span>
            </header>
            {boardRows.length ? (
            <BoardList rows={boardRows} mode={rankedMode} onSelect={(row) => {
                setResponseChampion(row.champion_id);
                setMode("responses");
              }} />
            ) : (
              <div className={styles.unavailable}>
                <p>{mode === "blind" ? "Blind stability" : "Good into common picks"} is waiting for the matchup refresh.</p>
                <span>The first-pick A–D board remains available for this role and patch.</span>
              </div>
            )}
          </section>
        ) : (
          <RoleBoardGrid rows={visibleRows} mode={rankedMode} onSelect={(row) => {
            setRole(row.role);
            setResponseChampion("");
          }} />
        )
      ) : null}

      <div className={styles.methodNote}>
        <strong>How to read this board</strong>
        <span>First pick uses the patch-wide model. Blind shows the expected weakest common matchup. Good into counts favorable modeled results against five common role opponents. Minimum games uses patch-wide accepted appearances. Unpicked alternatives use structural similarity and carry no performance claim. Region filters observed appearances and keeps the patch-wide fit fixed.</span>
      </div>
    </section>
  );
}
