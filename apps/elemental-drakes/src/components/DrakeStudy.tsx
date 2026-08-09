"use client";

import Image from "next/image";
import { useMemo, useState } from "react";

import championImages from "@/data/champion-images.json";
import {
  ELEMENTS,
  ROLES,
  buildCaptures,
  buildCurve,
  buildElementRankings,
  defaultPilot,
  enabledChampionResidualFeatures,
  formatClock,
  formatPp,
  legalElementChoice,
  ownerCounts,
  pilotSelections,
  randomDistinctElements,
  randomRoleTeam,
  sanitizeOwners,
  summarizeChampionEvidence,
  timelineStates,
  titleCaseTag,
  type Capture,
  type ChampionDifferentialSource,
  type ChampionEvidenceSummary,
  type CurvePoint,
  type ElementId,
  type ElementRanking,
  type ExplorerModel,
  type Inventory,
  type Mechanic,
  type OverallElementRanking,
  type PilotGame,
  type Role,
  type StudyArtifact,
  type TeamSide,
} from "@/lib/study";

const TEAM_COLORS = { A: "#66a1df", B: "#df8173" };
const ROLE_COLORS = ["#d76b58", "#b88a3b", "#35a4a8", "#8b77d2", "#7da55e"];
const TABS = ["analysis", "methodology", "sources"] as const;
type TabId = (typeof TABS)[number];
type ChartMode = "teams" | "champions";
type ChampionOrder = "role" | "pp-gain";

function championSourceLabel(source: ChampionDifferentialSource): string {
  if (source === "champion-informed") return "Direct";
  if (source === "mixed") return "Mixed";
  if (source === "archetype-prior-only") return "Archetype";
  return "Team only";
}

const EXPOSURE_RULE_LABELS: Record<string, string> = {
  "minimum-games": "team games",
  "minimum-series": "series",
  "minimum-ownership-games": "games with the dragon",
  "minimum-nonownership-games": "games without the dragon",
  "minimum-organizations": "organizations / team contexts",
  "minimum-organization-rosters": "organizations / team contexts",
};

function evidenceSupportCopy(summary: ChampionEvidenceSummary): string {
  if (!summary.activeElements.length) return "No inventory";
  if (summary.source === "archetype-prior-only") return "Prior";
  if (summary.source === "unsupported") return "Pooled";
  if (
    summary.minimumTrainingGames === null ||
    summary.leastSupportedElement === null
  ) {
    return "Sample n/a";
  }
  return `${summary.minimumTrainingGames.toLocaleString("en-US")} games`;
}

function evidenceSupportDetail(summary: ChampionEvidenceSummary): string {
  if (!summary.activeElements.length) return "No active dragon inventory.";
  const parts = [
    summary.minimumTrainingGames === null ||
    summary.minimumTrainingSeries === null ||
    summary.leastSupportedElement === null
      ? ""
      : `${titleCaseTag(
          summary.leastSupportedElement,
        )}: ${summary.minimumTrainingGames.toLocaleString(
          "en-US",
        )} team-games across ${summary.minimumTrainingSeries.toLocaleString(
          "en-US",
        )} series`,
    summary.failedExposureRules.length
      ? `Needs more ${summary.failedExposureRules
          .map(
            (rule) =>
              EXPOSURE_RULE_LABELS[rule] ??
              rule.replace(/^minimum-/, "").replaceAll("-", " "),
          )
          .join(", ")}`
      : "",
    summary.vocabularyProvenance === "post-audit-full-refit"
      ? "Added after the July family audit; not held out separately."
      : summary.vocabularyProvenance === "publication-audit-vocabulary"
        ? "Included in the July family audit; not held out separately."
        : "",
  ].filter(Boolean);
  return parts.join(". ");
}

function relativeFitLabel(value: number, exact = true): string {
  if (Math.abs(value) < 0.005) return exact ? "At pooled effect" : "Pooled";
  const direction = value > 0 ? "above" : "below";
  return exact
    ? `${Math.abs(value).toFixed(2)} pp ${direction} pooled`
    : `${direction === "above" ? "Above" : "Below"} pooled`;
}

const CHAMPION_PORTRAITS = championImages.images as Record<
  string,
  { path: string; sourceUrl: string }
>;

const DRAKE_FILE_PAGES: Record<ElementId, string> = {
  infernal:
    "https://wiki.leagueoflegends.com/en-us/File:Infernal_DrakeSquare.png",
  mountain:
    "https://wiki.leagueoflegends.com/en-us/File:Mountain_DrakeSquare.png",
  ocean: "https://wiki.leagueoflegends.com/en-us/File:Ocean_DrakeSquare.png",
  cloud: "https://wiki.leagueoflegends.com/en-us/File:Cloud_DrakeSquare.png",
  hextech:
    "https://wiki.leagueoflegends.com/en-us/File:Hextech_DrakeSquare.png",
  chemtech:
    "https://wiki.leagueoflegends.com/en-us/File:Chemtech_DrakeSquare.png",
};

function ChampionPortrait({
  champion,
  decorative = false,
}: {
  champion: string;
  decorative?: boolean;
}) {
  const portrait = CHAMPION_PORTRAITS[champion];
  if (!portrait) {
    return (
      <span className="champion-portrait champion-portrait--fallback" aria-hidden="true">
        {champion.slice(0, 1)}
      </span>
    );
  }
  return (
    <span className="champion-portrait">
      <Image
        src={portrait.path}
        alt={decorative ? "" : `${champion} portrait`}
        width={128}
        height={128}
        sizes="48px"
        unoptimized
      />
    </span>
  );
}

function DrakeMark({ id }: { id: ElementId }) {
  return (
    <span className={`drake-mark drake-mark--${id}`} aria-hidden="true">
      <Image
        src={`/drakes/${id}.png`}
        alt=""
        width={128}
        height={128}
        sizes="40px"
        unoptimized
      />
    </span>
  );
}

function ChampionPicker({
  side,
  role,
  champion,
  onSelect,
}: {
  side: TeamSide;
  role: Role;
  champion: string;
  onSelect: (champion: string) => boolean;
}) {
  const [draft, setDraft] = useState(champion);
  const [error, setError] = useState("");
  const errorId = `${side}-${role}-error`;

  function commit() {
    if (draft === champion) return;
    if (onSelect(draft)) {
      setError("");
    } else {
      setDraft(champion);
      setError("Choose an available champion from the list.");
    }
  }

  return (
    <div className="champion-picker">
      <ChampionPortrait champion={champion} decorative />
      <label htmlFor={`${side}-${role}`}>
        <span>{role}</span>
        <input
          id={`${side}-${role}`}
          list="champion-catalog"
          value={draft}
          aria-label={`${role} champion for Team ${side}`}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? errorId : undefined}
          autoComplete="off"
          onChange={(event) => {
            const next = event.target.value;
            setDraft(next);
            setError("");
            if (CHAMPION_PORTRAITS[next] && onSelect(next)) {
              setError("");
            }
          }}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commit();
              event.currentTarget.blur();
            }
          }}
        />
      </label>
      {error ? (
        <small id={errorId} role="status">
          {error}
        </small>
      ) : null}
    </div>
  );
}

function TeamEditor({
  side,
  name,
  champions,
  onChampion,
  onRandomize,
}: {
  side: TeamSide;
  name: string;
  champions: string[];
  onChampion: (index: number, champion: string) => boolean;
  onRandomize: () => void;
}) {
  return (
    <section className={`team-editor team-editor--${side.toLowerCase()}`}>
      <header>
        <div>
          <span className="team-letter">Team {side}</span>
          <h3>{name}</h3>
        </div>
        <button type="button" className="team-randomize" onClick={onRandomize}>
          Randomize
        </button>
      </header>
      <div className="champion-grid">
        {ROLES.map((role, index) => (
          <ChampionPicker
            key={`${side}-${role}-${champions[index] ?? ""}`}
            side={side}
            role={role}
            champion={champions[index] ?? ""}
            onSelect={(champion) => onChampion(index, champion)}
          />
        ))}
      </div>
    </section>
  );
}

function ExampleBrowser({
  games,
  selectedId,
  onSelectedId,
  onLoad,
}: {
  games: PilotGame[];
  selectedId: string;
  onSelectedId: (id: string) => void;
  onLoad: () => void;
}) {
  const groups = [
    {
      label: "Tier 1 regional leagues",
      games: games.filter((game) => game.competitionLevel === "tier1"),
    },
    {
      label: "International",
      games: games.filter((game) => game.competitionLevel === "international"),
    },
    {
      label: "Other professional",
      games: games.filter((game) => game.competitionLevel === "other-pro"),
    },
  ];
  return (
    <div className="example-browser">
      <label htmlFor="example-game">
        <span>Pro game</span>
        <select
          id="example-game"
          value={selectedId}
          onChange={(event) => onSelectedId(event.target.value)}
        >
          {groups.map((group) =>
            group.games.length ? (
              <optgroup key={group.label} label={group.label}>
                {group.games.map((game) => (
                  <option key={game.id} value={game.id}>
                    {game.league} · {game.regionLabel} ·{" "}
                    {game.teams.map((team) => team.name).join(" vs ")}
                  </option>
                ))}
              </optgroup>
            ) : null,
          )}
        </select>
      </label>
      <button type="button" className="button button--ink" onClick={onLoad}>
        Load
      </button>
    </div>
  );
}

function ElementSelect({
  id,
  label,
  value,
  disabled,
  mechanics,
  onChange,
}: {
  id: string;
  label: string;
  value: ElementId;
  disabled: ElementId[];
  mechanics: Mechanic[];
  onChange: (element: ElementId) => void;
}) {
  return (
    <label className={`element-select element-select--${value}`} htmlFor={id}>
      <span>{label}</span>
      <div>
        <DrakeMark id={value} />
        <select
          id={id}
          value={value}
          onChange={(event) => onChange(event.target.value as ElementId)}
        >
          {mechanics.map((mechanic) => (
            <option
              key={mechanic.id}
              value={mechanic.id}
              disabled={disabled.includes(mechanic.id)}
            >
              {mechanic.name}
            </option>
          ))}
        </select>
      </div>
    </label>
  );
}

function linePath(
  values: number[],
  x: (index: number) => number,
  y: (value: number) => number,
): string {
  return values
    .map((value, index) => `${index === 0 ? "M" : "L"}${x(index)},${y(value)}`)
    .join(" ");
}

type ChartSeries = {
  id: string;
  label: string;
  values: number[];
  color: string;
  side: TeamSide;
  kind: ChartMode;
  dashed?: boolean;
  champion?: string;
  role?: Role;
  championIndex?: number;
};

function niceDomain(values: number[], mode: ChartMode): number {
  const raw = Math.max(0.25, ...values.map((value) => Math.abs(value))) * 1.16;
  const step = mode === "teams" ? 5 : raw > 5 ? 1 : 0.5;
  return Math.max(step, Math.ceil(raw / step) * step);
}

function orderedChampionIndices(
  champions: string[],
  point: CurvePoint | undefined,
  order: ChampionOrder,
): number[] {
  const indices = champions.map((_, index) => index);
  if (order === "role") return indices;
  return indices.sort((left, right) => {
    const difference =
      (point?.championCumulativePp[right] ?? 0) -
      (point?.championCumulativePp[left] ?? 0);
    return difference || left - right;
  });
}

function heldInventoryCopy(
  inventory: Inventory,
  mechanics: Mechanic[],
): string {
  const held = ELEMENTS.flatMap((element) => {
    const count = inventory[element];
    if (!count) return [];
    const name =
      mechanics.find((mechanic) => mechanic.id === element)?.name ??
      titleCaseTag(element);
    return [`${name} ×${count}`];
  });
  return held.length ? held.join(" · ") : "No dragons";
}

function InventoryMarks({
  inventory,
  mechanics,
}: {
  inventory: Inventory;
  mechanics: Mechanic[];
}) {
  const active = ELEMENTS.filter((element) => inventory[element] > 0);
  return (
    <div
      className="inventory-marks"
      aria-label={heldInventoryCopy(inventory, mechanics)}
    >
      {active.length ? (
        active.map((element) => (
          <span key={element} title={titleCaseTag(element)}>
            <DrakeMark id={element} />
            <b>×{inventory[element]}</b>
          </span>
        ))
      ) : (
        <span className="inventory-empty">0 dragons</span>
      )}
    </div>
  );
}

function CurveChart({
  curveA,
  curveB,
  captures,
  mechanics,
  teamA,
  teamB,
  mode,
  onMode,
  activeStage,
  onActiveStage,
  championOrder,
  onChampionOrder,
}: {
  curveA: CurvePoint[];
  curveB: CurvePoint[];
  captures: Capture[];
  mechanics: Mechanic[];
  teamA: string[];
  teamB: string[];
  mode: ChartMode;
  onMode: (mode: ChartMode) => void;
  activeStage: number;
  onActiveStage: (stage: number) => void;
  championOrder: ChampionOrder;
  onChampionOrder: (order: ChampionOrder) => void;
}) {
  const width = 920;
  const height = 420;
  const margin = { top: 30, right: 28, bottom: 92, left: 70 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const [activeSeriesId, setActiveSeriesId] = useState("team-a");
  const teamSeries: ChartSeries[] = [
    {
      id: "team-a",
      label: "Team A total",
      values: curveA.map((point) => point.teamADeltaPp),
      color: TEAM_COLORS.A,
      side: "A",
      kind: "teams",
    },
    {
      id: "team-b",
      label: "Team B total",
      values: curveA.map((point) => point.teamBDeltaPp),
      color: TEAM_COLORS.B,
      side: "B",
      kind: "teams",
      dashed: true,
    },
  ];
  const championSeries: ChartSeries[] = (["A", "B"] as TeamSide[]).flatMap(
    (side) => {
      const champions = side === "A" ? teamA : teamB;
      const points = side === "A" ? curveA : curveB;
      return orderedChampionIndices(
        champions,
        points[activeStage],
        championOrder,
      ).map((index) => ({
        id: `champion-${side}-${index}`,
        label: `${side} · ${ROLES[index]} · ${champions[index]}`,
        values: points.map((point) => point.championCumulativePp[index] ?? 0),
        color: ROLE_COLORS[index],
        side,
        kind: "champions" as const,
        dashed: side === "B",
        champion: champions[index],
        role: ROLES[index],
        championIndex: index,
      }));
    },
  );
  const series = mode === "teams" ? teamSeries : championSeries;
  const activeSeries =
    series.find((candidate) => candidate.id === activeSeriesId) ?? series[0];
  const activeValue = activeSeries.values[activeStage] ?? 0;
  const plotted = series.flatMap((item) => item.values);
  const maxAbs = niceDomain(plotted, mode);
  const x = (index: number) =>
    margin.left +
    (curveA.length <= 1 ? 0 : (index / (curveA.length - 1)) * innerWidth);
  const y = (value: number) =>
    margin.top + ((maxAbs - value) / (2 * maxAbs)) * innerHeight;
  const ticks = [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs];

  function stageFromPointer(clientX: number, svg: SVGSVGElement): number {
    const bounds = svg.getBoundingClientRect();
    const viewX = ((clientX - bounds.left) / bounds.width) * width;
    const raw =
      curveA.length <= 1
        ? 0
        : ((viewX - margin.left) / innerWidth) * (curveA.length - 1);
    return Math.max(0, Math.min(curveA.length - 1, Math.round(raw)));
  }

  function nearestSeries(
    clientY: number,
    svg: SVGSVGElement,
    stage: number,
  ): ChartSeries {
    const bounds = svg.getBoundingClientRect();
    const viewY = ((clientY - bounds.top) / bounds.height) * height;
    return series.reduce((nearest, candidate) =>
      Math.abs(viewY - y(candidate.values[stage] ?? 0)) <
      Math.abs(viewY - y(nearest.values[stage] ?? 0))
        ? candidate
        : nearest,
    );
  }

  function setStageFromKey(key: string): boolean {
    if (key === "ArrowLeft") onActiveStage(Math.max(0, activeStage - 1));
    else if (key === "ArrowRight")
      onActiveStage(Math.min(curveA.length - 1, activeStage + 1));
    else if (key === "Home") onActiveStage(0);
    else if (key === "End") onActiveStage(curveA.length - 1);
    else return false;
    return true;
  }

  const activeCapture = activeStage ? captures[activeStage - 1] : null;
  const activePoint =
    activeSeries.side === "A" ? curveA[activeStage] : curveB[activeStage];
  const activeIndex = activeSeries.championIndex;
  const activeSource =
    activeIndex === undefined
      ? null
      : activePoint?.championDifferentialSource[activeIndex];
  const breakdown =
    activeIndex === undefined
      ? "Versus the same-time 0/0 reference"
      : `All held dragons · ${championSourceLabel(
          activeSource ?? "unsupported",
        )}`;

  return (
    <section className="trajectory" aria-labelledby="trajectory-title">
      <div className="trajectory-heading">
        <div>
          <h3 id="trajectory-title">
            {mode === "teams" ? "Inventory edge" : "Champion fit"}
          </h3>
          <p>
            {mode === "teams"
              ? "Versus the same-time no-dragon reference."
              : "Cumulative through the selected stage."}
          </p>
        </div>
        <div className="scale-toggle" role="group" aria-label="Chart scale">
          <button
            type="button"
            aria-pressed={mode === "teams"}
            onClick={() => {
              onMode("teams");
              setActiveSeriesId("team-a");
            }}
          >
            Teams
          </button>
          <button
            type="button"
            aria-pressed={mode === "champions"}
            onClick={() => {
              onMode("champions");
              setActiveSeriesId("champion-A-0");
            }}
          >
            Champions
          </button>
        </div>
      </div>
      {mode === "champions" ? (
        <div className="champion-order-control champion-sort">
          <button
            type="button"
            aria-pressed={championOrder === "role"}
            onClick={() => onChampionOrder("role")}
          >
            Role
          </button>
          <button
            type="button"
            aria-pressed={championOrder === "pp-gain"}
            onClick={() => onChampionOrder("pp-gain")}
          >
            Effect
          </button>
        </div>
      ) : null}

      <svg
        className="curve-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        tabIndex={0}
        aria-labelledby="curve-svg-title curve-svg-desc"
        onKeyDown={(event) => {
          if (setStageFromKey(event.key)) event.preventDefault();
        }}
      >
        <title id="curve-svg-title">
          {mode === "teams"
            ? "Inventory edge by capture"
            : "Champion fit by capture"}
        </title>
        <desc id="curve-svg-desc">
          {mode === "teams"
            ? "Two team lines. Arrow keys move between captures."
            : "Ten champion lines. Each point includes all dragons held."}
        </desc>
        <text
          className="axis-title"
          x="18"
          y={margin.top + innerHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 18 ${margin.top + innerHeight / 2})`}
        >
          {mode === "teams"
            ? "Team effect (pp)"
            : "Difference from pooled (pp)"}
        </text>
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={margin.left}
              x2={width - margin.right}
              y1={y(tick)}
              y2={y(tick)}
              className={Math.abs(tick) < 0.001 ? "axis-zero" : "axis-grid"}
            />
            <text x={margin.left - 10} y={y(tick) + 4} textAnchor="end">
              {tick.toFixed(1)}
            </text>
            {mode === "champions" && Math.abs(tick) < 0.001 ? (
              <text
                className="pooled-line-label"
                x={width - margin.right - 6}
                y={y(tick) - 7}
                textAnchor="end"
              >
                Pooled effect
              </text>
            ) : null}
          </g>
        ))}
        {curveA.map((point, index) => {
          const capture = index ? captures[index - 1] : null;
          const mechanic = capture
            ? mechanics.find((candidate) => candidate.id === capture.element)
            : null;
          return (
            <g key={point.stage}>
              <line
                x1={x(index)}
                x2={x(index)}
                y1={margin.top}
                y2={height - margin.bottom}
                className={index === activeStage ? "stage-current" : "stage-grid"}
              />
              <text x={x(index)} y={height - margin.bottom + 24} textAnchor="middle">
                {index === 0 ? "No dragons" : `#${index} → ${capture?.owner}`}
              </text>
              <text
                x={x(index)}
                y={height - margin.bottom + 42}
                textAnchor="middle"
                className="stage-time"
              >
                {index === 0 ? "0 / 0 stacks" : mechanic?.name}
              </text>
              <text
                x={x(index)}
                y={height - margin.bottom + 58}
                textAnchor="middle"
                className="stage-time"
              >
                {index === 0
                  ? "reference"
                  : `at ${formatClock(Math.round(point.minute * 60))}`}
              </text>
            </g>
          );
        })}
        {series.map((item) => (
          <g key={item.id}>
            <path
              d={linePath(item.values, x, y)}
              className={[
                item.kind === "teams" ? "team-line" : "champion-line",
                item.dashed ? "series-dashed" : "",
                item.id === activeSeries.id ? "series-active" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              style={{ stroke: item.color }}
            />
            {item.values.map((value, index) => (
              <circle
                key={`${item.id}-${index}`}
                className={[
                  "series-point",
                  curveA[index]?.supportStatus === "low-support"
                    ? "series-point--low"
                    : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                cx={x(index)}
                cy={y(value)}
                r={item.id === activeSeries.id && index === activeStage ? 5 : 2.7}
                style={{ stroke: item.color }}
              />
            ))}
          </g>
        ))}
        <rect
          className="plot-inspection-area"
          x={margin.left}
          y={margin.top}
          width={innerWidth}
          height={innerHeight}
          fill="transparent"
          onPointerMove={(event) => {
            const svg = event.currentTarget.ownerSVGElement;
            if (!svg) return;
            const stage = stageFromPointer(event.clientX, svg);
            onActiveStage(stage);
            setActiveSeriesId(nearestSeries(event.clientY, svg, stage).id);
          }}
        />
        <g className="active-marker" aria-hidden="true">
          <line
            x1={x(activeStage)}
            x2={x(activeStage)}
            y1={margin.top}
            y2={height - margin.bottom}
          />
          <circle
            cx={x(activeStage)}
            cy={y(activeValue)}
            r="6"
            style={{ fill: activeSeries.color }}
          />
        </g>
      </svg>

      <div className="trajectory-inspector">
        <div>
          {activeCapture ? <DrakeMark id={activeCapture.element} /> : null}
          <span>
            {activeCapture
              ? `Capture ${activeStage} · Team ${activeCapture.owner}`
              : "Display baseline"}
          </span>
        </div>
        {activeSeries.champion ? (
          <ChampionPortrait champion={activeSeries.champion} decorative />
        ) : null}
        <div>
          <strong>{activeSeries.label}</strong>
          <small>{breakdown}</small>
        </div>
        <b>
          {mode === "champions"
            ? relativeFitLabel(activeValue)
            : formatPp(activeValue)}
        </b>
      </div>

      <div
        className={`series-key series-key--${mode}`}
        role="group"
        aria-label={`${mode === "teams" ? "Team" : "Champion"} series`}
      >
        {mode === "champions" ? (
          <p className="series-key-note legend-subtitle">
            Solid A · dashed B · color by role
          </p>
        ) : null}
        {series.map((item) => (
          <button
            type="button"
            key={item.id}
            aria-pressed={item.id === activeSeries.id}
            onPointerEnter={() => setActiveSeriesId(item.id)}
            onFocus={() => setActiveSeriesId(item.id)}
            onClick={() => setActiveSeriesId(item.id)}
          >
            {item.champion ? (
              <ChampionPortrait champion={item.champion} decorative />
            ) : null}
            <i
              className={[
                "legend-line",
                item.dashed ? "legend-dashed" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              style={{ borderColor: item.color, background: item.color }}
              aria-hidden="true"
            />
            <span>
              <strong>{item.label}</strong>
              <small>
                {mode === "champions"
                  ? relativeFitLabel(item.values[activeStage] ?? 0)
                  : formatPp(item.values[activeStage] ?? 0)}
              </small>
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

function CaptureRail({
  side,
  teamName,
  captures,
  activeStage,
  mechanics,
  canAdd,
  onInspect,
  onMove,
  onAdd,
}: {
  side: TeamSide;
  teamName: string;
  captures: Capture[];
  activeStage: number;
  mechanics: Mechanic[];
  canAdd: boolean;
  onInspect: (stage: number) => void;
  onMove: (captureIndex: number, owner: TeamSide) => void;
  onAdd: (owner: TeamSide) => void;
}) {
  const owned = captures
    .map((capture, index) => ({ capture, index }))
    .filter(({ capture }) => capture.owner === side);
  const nextCapture = captures.length + 1;
  return (
    <aside className={`capture-rail capture-rail--${side.toLowerCase()}`}>
      <header>
        <span>Team {side}</span>
        <strong>{owned.length}/4</strong>
        <small>{teamName}</small>
      </header>
      <ol aria-label={`Team ${side} dragon inventory`}>
        {Array.from({ length: 4 }, (_, slot) => {
          const item = owned[slot];
          if (item) {
            const mechanic = mechanics.find(
              (candidate) => candidate.id === item.capture.element,
            );
            return (
              <li key={`${side}-${slot}`} className="capture-slot capture-slot--filled">
                <button
                  type="button"
                  className="capture-select"
                  aria-pressed={activeStage === item.index + 1}
                  onClick={() => onInspect(item.index + 1)}
                >
                  <DrakeMark id={item.capture.element} />
                  <span>
                    <small>#{item.index + 1}</small>
                    <strong>{mechanic?.name}</strong>
                  </span>
                </button>
                <button
                  type="button"
                  className="capture-move"
                  aria-label={`Move capture ${item.index + 1} to Team ${
                    side === "A" ? "B" : "A"
                  }`}
                  title={`Move to Team ${side === "A" ? "B" : "A"}`}
                  onClick={() =>
                    onMove(item.index, side === "A" ? "B" : "A")
                  }
                >
                  <span aria-hidden="true">
                    {side === "A" ? "→" : "←"}
                  </span>
                </button>
              </li>
            );
          }
          const isNext = slot === owned.length;
          return (
            <li key={`${side}-${slot}`} className="capture-slot capture-slot--empty">
              {isNext ? (
                <button
                  type="button"
                  disabled={!canAdd}
                  aria-label={`Assign capture ${nextCapture} to Team ${side}`}
                  title={`Assign capture ${nextCapture} to Team ${side}`}
                  onClick={() => onAdd(side)}
                >
                  <strong aria-hidden="true">+</strong>
                </button>
              ) : (
                <span aria-hidden="true">—</span>
              )}
            </li>
          );
        })}
      </ol>
    </aside>
  );
}

function ChampionDifferentials({
  model,
  curveA,
  curveB,
  captures,
  mechanics,
  teamA,
  teamB,
  teamAName,
  teamBName,
  activeStage,
  championCatalog,
  championOrder,
}: {
  model: ExplorerModel;
  curveA: CurvePoint[];
  curveB: CurvePoint[];
  captures: Capture[];
  mechanics: Mechanic[];
  teamA: string[];
  teamB: string[];
  teamAName: string;
  teamBName: string;
  activeStage: number;
  championCatalog: ExplorerModel["championCatalog"];
  championOrder: ChampionOrder;
}) {
  const pointA = curveA[activeStage] ?? curveA[0];
  const pointB = curveB[activeStage] ?? curveB[0];
  const capture = activeStage ? captures[activeStage - 1] : null;
  const selectedStates = timelineStates(captures);
  const selectedState = selectedStates[activeStage] ?? selectedStates[0];
  const maxChampion = niceDomain(
    [...curveA, ...curveB].flatMap(
      (point) => point.championCumulativePp,
    ),
    "champions",
  );
  const teams = [
    {
      side: "A" as const,
      name: teamAName,
      champions: teamA,
      point: pointA,
      inventory: selectedState.inventoryA,
    },
    {
      side: "B" as const,
      name: teamBName,
      champions: teamB,
      point: pointB,
      inventory: selectedState.inventoryB,
    },
  ];
  const catalog = useMemo(
    () => new Map(championCatalog.map((champion) => [champion.name, champion])),
    [championCatalog],
  );
  const enabledChampionFeatures = useMemo(
    () => enabledChampionResidualFeatures(model),
    [model],
  );
  const [showExact, setShowExact] = useState(false);
  const teamAEdge = pointA?.teamADeltaPp ?? 0;
  const edgeSide: TeamSide | null =
    Math.abs(teamAEdge) < 0.005 ? null : teamAEdge > 0 ? "A" : "B";
  const edgeMarker = 50 - Math.max(-46, Math.min(46, teamAEdge));

  return (
    <section
      className="champion-allocations"
      aria-labelledby="differential-stage-title"
    >
      <header>
        <div>
          <h3 id="differential-stage-title">
            {capture
              ? `Champion fit after capture ${activeStage}`
              : "Champion fit at baseline"}
          </h3>
          <p>
            Buffs apply to all five. Bars show fit above or below the pooled
            effect.
          </p>
        </div>
        <button
          type="button"
          className="fit-value-toggle"
          aria-pressed={showExact}
          onClick={() => setShowExact((current) => !current)}
        >
          {showExact ? "Hide pp" : "Show pp"}
        </button>
      </header>
      <div
        className="inventory-edge"
        aria-label={
          edgeSide
            ? `Full inventory edge: Team ${edgeSide}, ${Math.abs(teamAEdge).toFixed(2)} percentage points`
            : "Full inventory edge: even"
        }
      >
        <span>Team A</span>
        <div>
          <i />
          <b style={{ left: `${edgeMarker}%` }} />
        </div>
        <strong>
          {edgeSide
            ? `Team ${edgeSide} · ${Math.abs(teamAEdge).toFixed(2)} pp`
            : "Even"}
          <small>Full inventory edge</small>
        </strong>
        <span>Team B</span>
      </div>
      <div className="allocation-teams">
        {teams.map(({ side, name, champions, point, inventory }) => (
          <article key={side} className={`allocation-team allocation-team--${side.toLowerCase()}`}>
            <h4>
              <span>Team {side}</span>
              <small>{name}</small>
            </h4>
            <div className="allocation-inventory">
              <InventoryMarks inventory={inventory} mechanics={mechanics} />
              <strong>Buffs all 5</strong>
            </div>
            <ul>
              {orderedChampionIndices(
                champions,
                point,
                championOrder,
              ).map((index) => {
                const champion = champions[index];
                const value = point?.championCumulativePp[index] ?? 0;
                const width = Math.min(50, (Math.abs(value) / maxChampion) * 50);
                const entry = catalog.get(champion);
                const source =
                  point?.championDifferentialSource[index] ?? "unsupported";
                const evidenceSummary = summarizeChampionEvidence(
                  entry,
                  inventory,
                  enabledChampionFeatures,
                );
                const sourceCopy = championSourceLabel(source);
                const evidenceCopy = evidenceSupportCopy(evidenceSummary);
                const evidenceDetail = evidenceSupportDetail(evidenceSummary);
                return (
                  <li key={`${side}-${champion}`}>
                    <ChampionPortrait champion={champion} decorative />
                    <div className="allocation-label">
                      <strong>{ROLES[index]} · {champion}</strong>
                      <small
                        className="allocation-evidence"
                        title={evidenceDetail}
                      >
                        <span>{sourceCopy}</span>
                        {evidenceCopy}
                      </small>
                    </div>
                    <div
                      className="allocation-bar"
                      role="img"
                      aria-label={`${champion}: ${relativeFitLabel(value)}`}
                    >
                      <i />
                      <span
                        className={value < 0 ? "fit-lower" : "fit-higher"}
                        style={{ width: `${width}%` }}
                      />
                    </div>
                    <b title={relativeFitLabel(value)}>
                      {relativeFitLabel(value, showExact)}
                    </b>
                  </li>
                );
              })}
            </ul>
            <div
              className="allocation-reconcile"
              title="These terms reconcile exactly to the full team estimate. They are model accounting, not five independent causal effects."
            >
              <span>
                Shared + contextual
                <strong>{formatPp(point?.teamContextPp ?? 0)}</strong>
              </span>
              <span>
                Champion fit
                <strong>
                  {formatPp(
                    point?.championCumulativePp.reduce(
                      (sum, value) => sum + value,
                      0,
                    ) ?? 0,
                  )}
                </strong>
              </span>
              <span>
                Team
                <strong>
                  {formatPp(
                    side === "A"
                      ? point?.teamADeltaPp ?? 0
                      : point?.teamBDeltaPp ?? 0,
                  )}
                </strong>
              </span>
            </div>
          </article>
        ))}
      </div>
      <footer>
        <span>
          {pointA?.supportStatus === "low-support"
            ? `Low support · ${pointA.supportRows.toLocaleString("en-US")} stage observations`
            : pointA?.supportStatus === "baseline"
              ? "Unscored display baseline"
              : `${pointA?.supportRows.toLocaleString("en-US")} stage observations`}
        </span>
      </footer>
    </section>
  );
}

function DragonExplorer({
  model,
  first,
  second,
  rift,
  owners,
  mechanics,
  teamA,
  teamB,
  teamAName,
  teamBName,
  curveA,
  curveB,
  championCatalog,
  onFirst,
  onSecond,
  onRift,
  onOwners,
  onRandomizeSpawns,
}: {
  model: ExplorerModel;
  first: ElementId;
  second: ElementId;
  rift: ElementId;
  owners: TeamSide[];
  mechanics: Mechanic[];
  teamA: string[];
  teamB: string[];
  teamAName: string;
  teamBName: string;
  curveA: CurvePoint[];
  curveB: CurvePoint[];
  championCatalog: ExplorerModel["championCatalog"];
  onFirst: (element: ElementId) => void;
  onSecond: (element: ElementId) => void;
  onRift: (element: ElementId) => void;
  onOwners: (owners: TeamSide[]) => void;
  onRandomizeSpawns: () => void;
}) {
  const captures = buildCaptures(owners, first, second, rift);
  const states = timelineStates(captures);
  const latestState = states.at(-1);
  const hasSoul = Boolean(latestState?.soulA || latestState?.soulB);
  const counts = ownerCounts(owners);
  const [activeStageState, setActiveStageState] = useState(captures.length);
  const [mode, setMode] = useState<ChartMode>("teams");
  const [championOrder, setChampionOrder] = useState<ChampionOrder>("role");
  const activeStage = Math.min(activeStageState, captures.length);
  const soul = latestState?.soulA
    ? { side: "A" as const, element: latestState.soulA }
    : latestState?.soulB
      ? { side: "B" as const, element: latestState.soulB }
      : null;

  function changeOwner(index: number, side: TeamSide) {
    const changed = [...owners];
    changed[index] = side;
    const next = sanitizeOwners(changed);
    onOwners(next);
    setActiveStageState(Math.min(index + 1, next.length));
  }

  function addCapture(side: TeamSide) {
    if (hasSoul || owners.length >= 7 || counts[side] >= 4) return;
    const next = sanitizeOwners([...owners, side]);
    onOwners(next);
    setActiveStageState(next.length);
  }

  return (
    <section className="dragon-explorer" aria-labelledby="explorer-title">
      <header className="explorer-heading">
        <h2 id="explorer-title">Dragon path</h2>
      </header>

      <div className="spawn-toolbar">
        <ElementSelect
          id="first-element"
          label="First"
          value={first}
          disabled={[second, rift]}
          mechanics={mechanics}
          onChange={onFirst}
        />
        <ElementSelect
          id="second-element"
          label="Second"
          value={second}
          disabled={[first, rift]}
          mechanics={mechanics}
          onChange={onSecond}
        />
        <ElementSelect
          id="rift-element"
          label="Rift + soul"
          value={rift}
          disabled={[first, second]}
          mechanics={mechanics}
          onChange={onRift}
        />
        <button type="button" className="button spawn-randomize" onClick={onRandomizeSpawns}>
          Randomize
        </button>
      </div>

      <nav className="stage-selector" aria-label="Inspect capture stage">
        {curveA.map((point, index) => {
          const capture = index ? captures[index - 1] : null;
          return (
            <button
              type="button"
              key={point.stage}
              aria-current={index === activeStage ? "step" : undefined}
              onClick={() => setActiveStageState(index)}
            >
              {capture ? <DrakeMark id={capture.element} /> : <span className="baseline-dot" />}
              <span>{index === 0 ? "Baseline" : `Capture ${index}`}</span>
              <small>{capture ? `Team ${capture.owner}` : "0/0"}</small>
            </button>
          );
        })}
      </nav>

      <div className="dragon-board-grid">
        <CaptureRail
          side="A"
          teamName={teamAName}
          captures={captures}
          activeStage={activeStage}
          mechanics={mechanics}
          canAdd={!hasSoul && owners.length < 7 && counts.A < 4}
          onInspect={setActiveStageState}
          onMove={changeOwner}
          onAdd={addCapture}
        />
        <div className="plot-core">
          <CurveChart
            curveA={curveA}
            curveB={curveB}
            captures={captures}
            mechanics={mechanics}
            teamA={teamA}
            teamB={teamB}
            mode={mode}
            onMode={setMode}
            activeStage={activeStage}
            onActiveStage={setActiveStageState}
            championOrder={championOrder}
            onChampionOrder={setChampionOrder}
          />
          <div className={`soul-status ${soul ? "soul-status--earned" : ""}`}>
            {soul ? <DrakeMark id={soul.element} /> : <span className="soul-glyph">◇</span>}
            <div>
              <span>Soul</span>
              <strong>
                {soul
                  ? `${titleCaseTag(soul.element)} Soul · Team ${soul.side}`
                  : "No soul yet · first to four"}
              </strong>
            </div>
          </div>
        </div>
        <CaptureRail
          side="B"
          teamName={teamBName}
          captures={captures}
          activeStage={activeStage}
          mechanics={mechanics}
          canAdd={!hasSoul && owners.length < 7 && counts.B < 4}
          onInspect={setActiveStageState}
          onMove={changeOwner}
          onAdd={addCapture}
        />
      </div>

      <div className="path-actions">
        <p aria-live="polite">
          Team A {counts.A}/4 · Team B {counts.B}/4
          {hasSoul ? " · Soul" : ""}
        </p>
        <button
          type="button"
          className="button button--quiet"
          disabled={!owners.length}
          onClick={() => {
            const next = owners.slice(0, -1);
            onOwners(next);
            setActiveStageState(Math.min(activeStage, next.length));
          }}
        >
          Undo
        </button>
      </div>

      <ChampionDifferentials
        model={model}
        curveA={curveA}
        curveB={curveB}
        captures={captures}
        mechanics={mechanics}
        teamA={teamA}
        teamB={teamB}
        teamAName={teamAName}
        teamBName={teamBName}
        activeStage={activeStage}
        championCatalog={championCatalog}
        championOrder={championOrder}
      />
    </section>
  );
}

function OverallDragonFit({
  mechanics,
  rankingsOverall,
  modeledGames,
}: {
  mechanics: Mechanic[];
  rankingsOverall: OverallElementRanking[];
  modeledGames: number;
}) {
  const mechanicFor = (element: ElementId) =>
    mechanics.find((candidate) => candidate.id === element);
  const comparisonRows = [...rankingsOverall].sort((left, right) =>
    (mechanicFor(left.element)?.name ?? left.element).localeCompare(
      mechanicFor(right.element)?.name ?? right.element,
    ),
  );
  const stages = [
    {
      key: "first",
      label: "First",
      value: (ranking: OverallElementRanking) => ranking.firstCapturePp,
    },
    {
      key: "second",
      label: "Second",
      value: (ranking: OverallElementRanking) =>
        ranking.secondCapturePp ?? null,
    },
    {
      key: "map",
      label: "Map phase",
      helper: "3rd onward, incl. soul",
      value: (ranking: OverallElementRanking) =>
        ranking.mapPhaseCapturePp ?? null,
    },
  ] as const;
  const ranks = new Map<string, number>();
  for (const stage of stages) {
    const ordered = rankingsOverall
      .map((ranking) => ({
        element: ranking.element,
        value: stage.value(ranking),
      }))
      .filter(
        (entry): entry is { element: ElementId; value: number } =>
          entry.value !== null && Number.isFinite(entry.value),
      )
      .sort((left, right) => right.value - left.value);
    let previousValue: number | null = null;
    let previousRank = 0;
    ordered.forEach((entry, index) => {
      const rank =
        previousValue !== null && Math.abs(entry.value - previousValue) < 1e-9
          ? previousRank
          : index + 1;
      ranks.set(`${stage.key}:${entry.element}`, rank);
      previousValue = entry.value;
      previousRank = rank;
    });
  }
  const ordinal = (rank: number) => {
    const mod100 = rank % 100;
    if (mod100 >= 11 && mod100 <= 13) return `${rank}th`;
    if (rank % 10 === 1) return `${rank}st`;
    if (rank % 10 === 2) return `${rank}nd`;
    if (rank % 10 === 3) return `${rank}rd`;
    return `${rank}th`;
  };

  return (
    <section
      className="overall-dragon page-shell"
      aria-labelledby="overall-dragon-title"
    >
      <header>
        <div>
          <h2 id="overall-dragon-title">Dragon rankings by stage</h2>
          <p>Adjusted point estimates across pro drafts</p>
        </div>
        <p className="overall-rank-note">
          {modeledGames.toLocaleString("en-US")} modeled games · Ranks have no
          uncertainty intervals.
        </p>
      </header>
      <table className="overall-stage-matrix">
        <caption className="sr-only">
          Adjusted dragon point estimates and rank at first capture, second
          capture, and the repeated map phase
        </caption>
        <thead>
          <tr>
            <th scope="col">Dragon</th>
            {stages.map((stage) => (
              <th scope="col" key={stage.key}>
                <span>{stage.label}</span>
                {"helper" in stage ? <small>{stage.helper}</small> : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {comparisonRows.map((ranking) => (
            <tr key={ranking.element}>
              <th scope="row">
                <DrakeMark id={ranking.element} />
                <strong>
                  {mechanicFor(ranking.element)?.name ?? ranking.element}
                </strong>
              </th>
              {stages.map((stage) => {
                const value = stage.value(ranking);
                const rank = ranks.get(`${stage.key}:${ranking.element}`);
                const leader = rank === 1;
                return (
                  <td
                    className={leader ? "is-leader" : undefined}
                    data-stage={stage.label}
                    key={stage.key}
                  >
                    <span className="overall-mobile-stage" aria-hidden="true">
                      {stage.label}
                      {"helper" in stage ? <small>{stage.helper}</small> : null}
                    </span>
                    <div className="overall-stage-cell">
                      <span>
                        {rank ? ordinal(rank) : "—"}
                        {leader ? <em>Leader</em> : null}
                      </span>
                      <strong>{value === null ? "—" : formatPp(value)}</strong>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function DragonFit({
  mechanics,
  rankingsA,
  rankingsB,
  teamAName,
  teamBName,
}: {
  mechanics: Mechanic[];
  rankingsA: ElementRanking[];
  rankingsB: ElementRanking[];
  teamAName: string;
  teamBName: string;
}) {
  return (
    <section className="dragon-fit page-shell" aria-labelledby="dragon-fit-title">
      <header>
        <h2 id="dragon-fit-title">Best for these lineups</h2>
      </header>
      <div className="fit-grid">
        {[
          { side: "A" as const, name: teamAName, rankings: rankingsA },
          { side: "B" as const, name: teamBName, rankings: rankingsB },
        ].map(({ side, name, rankings }) => (
          <article key={side}>
            <h3>
              Team {side}
              <small>{name}</small>
            </h3>
            <div className="fit-column-headings" aria-hidden="true">
              <span>Dragon</span>
              <span>First</span>
              <span>4 + soul</span>
            </div>
            <ol>
              {rankings.map((ranking, index) => {
                const mechanic = mechanics.find(
                  (candidate) => candidate.id === ranking.element,
                );
                return (
                  <li key={ranking.element}>
                    <span>{index + 1}</span>
                    <DrakeMark id={ranking.element} />
                    <strong>{mechanic?.name}</strong>
                    <b>{formatPp(ranking.firstCapturePp)}</b>
                    <b>{formatPp(ranking.perfectControlSoulPp)}</b>
                  </li>
                );
              })}
            </ol>
          </article>
        ))}
      </div>
    </section>
  );
}

function MethodologyPanel({ study }: { study: StudyArtifact }) {
  const model = study.explorerModel;
  const diagnostics = model.models.jointState.diagnostics;
  const coverage = study.competitionCoverage;
  return (
    <section
      id="panel-methodology"
      className="evidence-panel page-shell"
      role="tabpanel"
      aria-labelledby="tab-methodology"
    >
      <header className="evidence-heading">
        <h1>Methodology</h1>
        <p>
          {model.cohort.modeledGames.toLocaleString("en-US")} of{" "}
          {coverage.eligibleGames.toLocaleString("en-US")} complete GRID games
          met model requirements.
        </p>
      </header>

      <div className="method-ledger">
        <article>
          <h2>Estimate</h2>
          <p>
            Each inventory is compared with a same-time 0/0 reference, adjusted
            for state, side, drafts, and prior strength. After stage 0, the
            reference is synthetic.
          </p>
        </article>
        <article>
          <h2>Champion fit</h2>
          <p>
            Cumulative terms pool roles, patches, leagues, and teams. Sparse
            cells shrink toward archetypes. Pair synergy and counters are not
            modeled.
          </p>
        </article>
        <article>
          <h2>Evidence gate</h2>
          <p>
            50 games, 25 series, 20 with and without the dragon, and three
            organizations. July tests the family, not each cell.
          </p>
        </article>
        <article>
          <h2>Validation</h2>
          <p>
            {diagnostics.holdoutGames.toLocaleString("en-US")} games · Brier{" "}
            {diagnostics.brier.toFixed(4)} (null{" "}
            {diagnostics.nullBrier.toFixed(4)}) · calibration{" "}
            {(diagnostics.ece10 * 100).toFixed(2)} pp.
          </p>
        </article>
        <article>
          <h2>Stage ranks</h2>
          <p>
            Both sides of every observed draft are averaged. Map phase is one
            repeated capture from the third onward; the final increment includes
            soul.
          </p>
        </article>
        <article>
          <h2>Limits</h2>
          <p>
            Associational, not causal. No path or cell interval. Tier 1 is LCK,
            LPL, LEC, LCS, CBLOL, and LCP; all levels share one model.
          </p>
        </article>
      </div>
    </section>
  );
}

function SourcesPanel({ study }: { study: StudyArtifact }) {
  return (
    <section
      id="panel-sources"
      className="evidence-panel page-shell"
      role="tabpanel"
      aria-labelledby="tab-sources"
    >
      <header className="evidence-heading">
        <h1>Sources</h1>
      </header>
      <div className="source-ledger">
        <a href="https://grid.gg/get-league-of-legends/">
          <strong>GRID Open Platform</strong>
          <span>6,504 complete · 6,382 modeled · raw archives unpublished</span>
        </a>
        <a href="https://lol.fandom.com/wiki/Leaguepedia:Community">
          <strong>Leaguepedia</strong>
          <span>{study.roleCatalog.games.toLocaleString("en-US")} pro drafts · role weights</span>
        </a>
        <a href={championImages.source.apiEndpoint}>
          <strong>League of Legends Wiki</strong>
          <span>Portraits · dragon art · mechanics</span>
        </a>
      </div>
      <details className="source-citations">
        <summary>Mechanics and image citations</summary>
        <div className="source-links">
          <div>
            <h2>Mechanics</h2>
            {study.mechanics.map((mechanic) => (
              <a key={mechanic.id} href={mechanic.sourceUrl}>
                {mechanic.name} · {mechanic.source}
              </a>
            ))}
          </div>
          <div>
            <h2>Images</h2>
            {study.mechanics.map((mechanic) => (
              <a key={mechanic.id} href={DRAKE_FILE_PAGES[mechanic.id]}>
                {mechanic.name} file
              </a>
            ))}
          </div>
        </div>
      </details>
      <p className="source-meta">
        Generated {new Date(study.metadata.generatedAt).toLocaleDateString("en-US", {
          year: "numeric",
          month: "long",
          day: "numeric",
        })}
        {study.metadata.explorerModelSource
          ? (
              <>
                {" · "}Model {study.metadata.explorerModelSource.schemaVersion.replace(
                  "elemental-drake-explorer-",
                  "",
                )}{" "}
                <span title={study.metadata.explorerModelSource.sha256}>
                  · SHA-256
                </span>
              </>
            )
          : null}
      </p>
    </section>
  );
}

export function DrakeStudy({ study }: { study: StudyArtifact }) {
  const initialGame = useMemo(() => defaultPilot(study), [study]);
  const initial = useMemo(() => pilotSelections(initialGame), [initialGame]);
  const [tab, setTab] = useState<TabId>("analysis");
  const [exampleId, setExampleId] = useState(initialGame.id);
  const [teamAName, setTeamAName] = useState(initial.teamAName);
  const [teamBName, setTeamBName] = useState(initial.teamBName);
  const [teamA, setTeamA] = useState(initial.teamA);
  const [teamB, setTeamB] = useState(initial.teamB);
  const [first, setFirst] = useState(initial.first);
  const [second, setSecond] = useState(initial.second);
  const [rift, setRift] = useState(initial.rift);
  const [owners, setOwners] = useState<TeamSide[]>(initial.owners);

  const model = study.explorerModel;
  const validNames = useMemo(
    () => new Set(model.championCatalog.map((champion) => champion.name)),
    [model.championCatalog],
  );
  const captures = useMemo(
    () => buildCaptures(owners, first, second, rift),
    [owners, first, second, rift],
  );
  const curveA = useMemo(
    () => buildCurve(model, teamA, teamB, captures, "A"),
    [model, teamA, teamB, captures],
  );
  const curveB = useMemo(
    () => buildCurve(model, teamA, teamB, captures, "B"),
    [model, teamA, teamB, captures],
  );
  const rankingsA = useMemo(
    () => buildElementRankings(model, teamA, teamB, "A"),
    [model, teamA, teamB],
  );
  const rankingsB = useMemo(
    () => buildElementRankings(model, teamA, teamB, "B"),
    [model, teamA, teamB],
  );
  const overallRankings = model.models.jointState.overallElementRankings;
  const rankingsOverall = overallRankings.rankings;
  const modeledGames = overallRankings.support.modeledGames;
  const activeExample =
    study.pilotGames.find((game) => game.id === exampleId) ?? initialGame;

  function loadGame(game: PilotGame) {
    const loaded = pilotSelections(game);
    setTeamAName(loaded.teamAName);
    setTeamBName(loaded.teamBName);
    setTeamA(loaded.teamA);
    setTeamB(loaded.teamB);
    setFirst(loaded.first);
    setSecond(loaded.second);
    setRift(loaded.rift);
    setOwners(loaded.owners);
  }

  function selectChampion(
    side: TeamSide,
    index: number,
    champion: string,
  ): boolean {
    if (!validNames.has(champion)) return false;
    const all = [...teamA, ...teamB];
    const ownGlobalIndex = side === "A" ? index : 5 + index;
    if (
      all.some(
        (selected, selectedIndex) =>
          selectedIndex !== ownGlobalIndex && selected === champion,
      )
    ) {
      return false;
    }
    if (side === "A") {
      setTeamA((current) =>
        current.map((selected, selectedIndex) =>
          selectedIndex === index ? champion : selected,
        ),
      );
    } else {
      setTeamB((current) =>
        current.map((selected, selectedIndex) =>
          selectedIndex === index ? champion : selected,
        ),
      );
    }
    return true;
  }

  function randomizeTeam(side: TeamSide) {
    const excluded = side === "A" ? teamB : teamA;
    const randomized = randomRoleTeam(study.roleCatalog, validNames, excluded);
    if (side === "A") {
      setTeamA(randomized);
      setTeamAName("Custom Team A");
    } else {
      setTeamB(randomized);
      setTeamBName("Custom Team B");
    }
  }

  function randomizeBoth() {
    const randomizedA = randomRoleTeam(study.roleCatalog, validNames);
    const randomizedB = randomRoleTeam(
      study.roleCatalog,
      validNames,
      randomizedA,
    );
    setTeamA(randomizedA);
    setTeamB(randomizedB);
    setTeamAName("Custom Team A");
    setTeamBName("Custom Team B");
  }

  function changeElement(
    slot: "first" | "second" | "rift",
    element: ElementId,
  ) {
    if (!legalElementChoice(slot, element, first, second, rift)) return;
    if (slot === "first") setFirst(element);
    else if (slot === "second") setSecond(element);
    else setRift(element);
  }

  function activateTab(nextTab: TabId) {
    setTab(nextTab);
    requestAnimationFrame(() => {
      const anchor = document.getElementById("study-tabs-anchor");
      if (anchor) {
        window.scrollTo({
          top: anchor.getBoundingClientRect().top + window.scrollY,
          behavior: "auto",
        });
      }
    });
  }

  function handleTabKey(event: React.KeyboardEvent<HTMLButtonElement>) {
    const current = TABS.indexOf(tab);
    let next = current;
    if (event.key === "ArrowLeft") next = (current + TABS.length - 1) % TABS.length;
    else if (event.key === "ArrowRight") next = (current + 1) % TABS.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = TABS.length - 1;
    else return;
    event.preventDefault();
    const nextTab = TABS[next];
    activateTab(nextTab);
    document.querySelector<HTMLButtonElement>(`[data-tab="${nextTab}"]`)?.focus();
  }

  return (
    <main id="top">
      <datalist id="champion-catalog">
        {model.championCatalog.map((champion) => (
          <option key={champion.name} value={champion.name}>
            {champion.proGameAppearances.toLocaleString("en-US")} pro appearances
          </option>
        ))}
      </datalist>

      <header className="site-header page-shell">
        <a className="wordmark" href="#top">SCRYGLASS</a>
        <span className="header-rule" />
        <span>Elemental drakes</span>
        <span>{model.cohort.modeledGames.toLocaleString("en-US")} games · GRID</span>
      </header>

      <section className="app-intro page-shell">
        <h1>What is each dragon worth?</h1>
        <p>Choose two lineups, then assign each dragon.</p>
      </section>

      <div id="study-tabs-anchor" aria-hidden="true" />
      <div className="view-tabs-wrap">
        <div className="view-tabs page-shell" role="tablist" aria-label="Study sections">
          {TABS.map((id) => (
            <button
              type="button"
              role="tab"
              id={`tab-${id}`}
              data-tab={id}
              key={id}
              aria-selected={tab === id}
              aria-controls={`panel-${id}`}
              tabIndex={tab === id ? 0 : -1}
              onClick={() => activateTab(id)}
              onKeyDown={handleTabKey}
            >
              {id}
            </button>
          ))}
        </div>
      </div>

      <article
        id="panel-analysis"
        role="tabpanel"
        aria-labelledby="tab-analysis"
        hidden={tab !== "analysis"}
      >
        <OverallDragonFit
          mechanics={study.mechanics}
          rankingsOverall={rankingsOverall}
          modeledGames={modeledGames}
        />

        <section className="worksheet">
          <div className="page-shell worksheet-inner">
            <div className="setup-heading">
              <h2>Lineups</h2>
            </div>

            <ExampleBrowser
              games={study.pilotGames}
              selectedId={exampleId}
              onSelectedId={setExampleId}
              onLoad={() => loadGame(activeExample)}
            />
            <div className="team-actions">
              <button
                className="button"
                type="button"
                onClick={() => {
                  setTeamAName(teamBName);
                  setTeamBName(teamAName);
                  setTeamA(teamB);
                  setTeamB(teamA);
                  setOwners((current) =>
                    current.map((owner) => (owner === "A" ? "B" : "A")),
                  );
                }}
              >
                Swap
              </button>
              <button className="button" type="button" onClick={randomizeBoth}>
                Randomize
              </button>
              <button
                className="button button--quiet"
                type="button"
                onClick={() => {
                  setExampleId(initialGame.id);
                  loadGame(initialGame);
                }}
              >
                Reset
              </button>
            </div>

            <div className="teams-grid">
              <TeamEditor
                side="A"
                name={teamAName}
                champions={teamA}
                onChampion={(index, champion) => selectChampion("A", index, champion)}
                onRandomize={() => randomizeTeam("A")}
              />
              <TeamEditor
                side="B"
                name={teamBName}
                champions={teamB}
                onChampion={(index, champion) => selectChampion("B", index, champion)}
                onRandomize={() => randomizeTeam("B")}
              />
            </div>

            <DragonExplorer
              model={model}
              first={first}
              second={second}
              rift={rift}
              owners={owners}
              mechanics={study.mechanics}
              teamA={teamA}
              teamB={teamB}
              teamAName={teamAName}
              teamBName={teamBName}
              curveA={curveA}
              curveB={curveB}
              championCatalog={model.championCatalog}
              onFirst={(element) => changeElement("first", element)}
              onSecond={(element) => changeElement("second", element)}
              onRift={(element) => changeElement("rift", element)}
              onOwners={setOwners}
              onRandomizeSpawns={() => {
                const [nextFirst, nextSecond, nextRift] = randomDistinctElements();
                setFirst(nextFirst);
                setSecond(nextSecond);
                setRift(nextRift);
              }}
            />
          </div>
        </section>

        <DragonFit
          mechanics={study.mechanics}
          rankingsA={rankingsA}
          rankingsB={rankingsB}
          teamAName={teamAName}
          teamBName={teamBName}
        />
      </article>

      <div hidden={tab !== "methodology"}>
        <MethodologyPanel study={study} />
      </div>
      <div hidden={tab !== "sources"}>
        <SourcesPanel study={study} />
      </div>

      <footer className="site-footer page-shell">
        <div>
          <a className="wordmark" href="#top">SCRYGLASS</a>
          <p>Non-betting research on professional League of Legends.</p>
        </div>
        <p className="riot-notice">
          Not endorsed by Riot Games. League of Legends and its assets are ©
          Riot Games, Inc.
        </p>
      </footer>
    </main>
  );
}
