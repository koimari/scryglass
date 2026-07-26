"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  DraftAction,
  DraftCandidateRole,
  DraftChampion,
  DraftRecommendation,
  DraftRole,
  DraftSandboxResult,
  DraftSide,
} from "@/lib/draftScore";
import { champIconUrl } from "@/lib/format";

const PICK_ORDER: DraftSide[] = [
  "blue",
  "red",
  "red",
  "blue",
  "blue",
  "red",
  "blue",
  "blue",
  "red",
  "red",
];

const DAYOS_EXAMPLE: DraftAction[] = [
  { side: "blue", champion: "Jarvan IV", role: "jng" },
  { side: "red", champion: "Ezreal", role: "bot" },
  { side: "red", champion: "Naafiri", role: "jng" },
  { side: "blue", champion: "Orianna", role: "mid" },
  { side: "blue", champion: "Jayce", role: "top" },
];

const ROLE_LABEL: Record<DraftRole, string> = {
  top: "Top",
  jng: "Jungle",
  mid: "Mid",
  bot: "ADC",
  sup: "Support",
};

const LEAGUES = ["LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP", "INTL"];

type Props = {
  catalog: DraftChampion[];
  initialActions?: DraftAction[];
  initialExcluded?: string[];
  initialPerspective?: DraftSide;
  initialLeague?: string;
};

function sideLabel(side: DraftSide): string {
  return side === "blue" ? "Blue" : "Red";
}

function signedPp(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} pp`;
}

function encodeScenario(actions: DraftAction[], excluded: string[]): string {
  return btoa(JSON.stringify({ v: 1, actions, excluded }))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

function DraftSideColumn({
  side,
  actions,
  nextSide,
  onBranch,
}: {
  side: DraftSide;
  actions: DraftAction[];
  nextSide: DraftSide | null;
  onBranch: (actionIndex: number) => void;
}) {
  const sideActions = actions
    .map((action, index) => ({ action, index }))
    .filter(({ action }) => action.side === side);
  return (
    <section className={`sandbox-side sandbox-side-${side}`} aria-label={`${sideLabel(side)} side picks`}>
      <header className="sandbox-side-head">
        <span>{sideLabel(side)} side</span>
        <strong>{sideActions.length}/5</strong>
      </header>
      <ol className="sandbox-pick-list">
        {Array.from({ length: 5 }, (_, slot) => {
          const selected = sideActions[slot];
          const isNext = !selected && slot === sideActions.length && nextSide === side;
          return (
            <li className={isNext ? "is-next" : ""} key={`${side}-${slot}`}>
              <span className="sandbox-pick-seat">{side === "blue" ? "B" : "R"}{slot + 1}</span>
              {selected ? (
                <>
                  <span
                    className="sandbox-champion-portrait"
                    aria-hidden
                    style={{ backgroundImage: `url("${champIconUrl(selected.action.champion)}")` }}
                  />
                  <span className="sandbox-pick-name">{selected.action.champion}</span>
                  <span className="sandbox-pick-role">
                    {selected.action.role ? ROLE_LABEL[selected.action.role] : "Role open"}
                  </span>
                  <button
                    type="button"
                    className="sandbox-branch-button"
                    onClick={() => onBranch(selected.index)}
                    aria-label={`Branch before ${selected.action.champion}`}
                  >
                    Branch
                  </button>
                </>
              ) : (
                <span className="sandbox-pick-empty">{isNext ? "Next decision" : "Open"}</span>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function RecommendationTable({
  rows,
  perspective,
  onPick,
}: {
  rows: DraftRecommendation[];
  perspective: DraftSide;
  onPick: (row: DraftRecommendation) => void;
}) {
  if (!rows.length) {
    return <p className="sandbox-empty">No legal candidates match this role filter.</p>;
  }
  return (
    <div className="table-scroll sandbox-recommendation-scroll">
      <table className="data-table sandbox-recommendations">
        <thead>
          <tr>
            <th>#</th>
            <th>Candidate</th>
            <th>Role</th>
            <th>{sideLabel(perspective)} projected WR</th>
            <th>Change</th>
            <th>Pro games</th>
            <th>Evidence</th>
            <th><span className="sr-only">Select</span></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.champion}-${row.role}`}>
              <td className="font-mono">{index + 1}</td>
              <td>
                <span className="sandbox-candidate-name">
                  <span
                    className="sandbox-champion-portrait"
                    aria-hidden
                    style={{ backgroundImage: `url("${champIconUrl(row.champion)}")` }}
                  />
                  <strong>{row.champion}</strong>
                </span>
              </td>
              <td>{row.role ? ROLE_LABEL[row.role] : "Unassigned"}</td>
              <td className="font-mono">{(100 * row.projected_wr).toFixed(2)}%</td>
              <td className={`font-mono ${row.delta_pp >= 0 ? "sandbox-positive" : "sandbox-negative"}`}>
                {signedPp(row.delta_pp)}
              </td>
              <td className="font-mono">{row.sample_games}</td>
              <td>{row.evidence}</td>
              <td>
                <button type="button" className="sandbox-use-pick" onClick={() => onPick(row)}>
                  Use pick
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DraftSandbox({
  catalog,
  initialActions = DAYOS_EXAMPLE,
  initialExcluded = [],
  initialPerspective = "red",
  initialLeague = "LCS",
}: Props) {
  const [actions, setActions] = useState<DraftAction[]>(initialActions);
  const [excluded, setExcluded] = useState<string[]>(initialExcluded);
  const [perspective, setPerspective] = useState<DraftSide>(initialPerspective);
  const [league, setLeague] = useState(initialLeague);
  const [candidateRole, setCandidateRole] = useState<DraftCandidateRole>("open");
  const [search, setSearch] = useState("");
  const [excludeSearch, setExcludeSearch] = useState("");
  const [analysis, setAnalysis] = useState<DraftSandboxResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [shareState, setShareState] = useState("Copy scenario");
  const nextSide = actions.length < PICK_ORDER.length ? PICK_ORDER[actions.length] : null;

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      if (!nextSide) {
        setLoading(false);
      }
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("/api/draft-sandbox", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            actions,
            perspective,
            next_side: nextSide ?? perspective,
            candidate_role: candidateRole,
            excluded,
            league,
            limit: nextSide ? 15 : 1,
          }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `draft sandbox ${response.status}`);
        setAnalysis(payload as DraftSandboxResult);
      } catch (requestError) {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setAnalysis(null);
        setError(requestError instanceof Error ? requestError.message : String(requestError));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 120);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [actions, candidateRole, excluded, league, nextSide, perspective]);

  const selected = useMemo(
    () => new Set(actions.map((action) => action.champion.toLocaleLowerCase())),
    [actions],
  );
  const available = useMemo(
    () => {
      const excludedNames = new Set(excluded.map((champion) => champion.toLocaleLowerCase()));
      return catalog.filter(
        (champion) =>
          !selected.has(champion.name.toLocaleLowerCase()) &&
          !excludedNames.has(champion.name.toLocaleLowerCase()),
      );
    },
    [catalog, excluded, selected],
  );
  const responseCheckpoint = analysis?.timeline[2] ?? null;
  const current = analysis?.current;

  const addPick = (champion: string, role: DraftRole | null) => {
    if (!nextSide) return;
    setActions((currentActions) => [...currentActions, { side: nextSide, champion, role }]);
    setSearch("");
  };

  const addSearchPick = () => {
    const champion = available.find(
      (candidate) => candidate.name.toLocaleLowerCase() === search.trim().toLocaleLowerCase(),
    );
    if (!champion) {
      setError("Choose a champion from the available model pool.");
      return;
    }
    const openRoles = analysis?.open_roles ?? [];
    const role =
      candidateRole !== "open" && candidateRole !== "any"
        ? candidateRole
        : champion.roles.find((candidate) => openRoles.includes(candidate)) ??
          champion.roles[0] ??
          null;
    addPick(champion.name, role);
  };

  const markUnavailable = () => {
    const champion = catalog.find(
      (candidate) =>
        candidate.name.toLocaleLowerCase() === excludeSearch.trim().toLocaleLowerCase(),
    );
    if (!champion || selected.has(champion.name.toLocaleLowerCase())) {
      setError("Choose an unselected champion from the model pool.");
      return;
    }
    setExcluded((current) =>
      current.includes(champion.name) ? current : [...current, champion.name],
    );
    setExcludeSearch("");
  };

  const copyScenario = async () => {
    const url = new URL(window.location.href);
    url.searchParams.set("draft", encodeScenario(actions, excluded));
    url.searchParams.set("side", perspective);
    url.searchParams.set("league", league);
    await navigator.clipboard.writeText(url.toString());
    setShareState("Copied");
    window.setTimeout(() => setShareState("Copy scenario"), 1600);
  };

  return (
    <div className="sandbox-page">
      <header className="page-header">
        <p className="blog-kicker">Model lab · Draft counterfactual</p>
        <h1 className="font-display mt-2 text-3xl">Draft sandbox</h1>
        <p className="lede">
          Replay a pick sequence, measure when the model moved, and rank the strongest legal next
          response. The model updates after every selection.
        </p>
        <div className="micro-log mt-4">
          <span><strong>Example question</strong> SEN Dayos</span>
          <span><strong>Model pool</strong> {catalog.length} pro-play champions</span>
          <span><strong>Output</strong> pick-by-pick ΔWR</span>
        </div>
        <div className="sandbox-header-actions">
          <button type="button" className="btn-primary" onClick={() => {
            setActions(DAYOS_EXAMPLE);
            setExcluded([]);
            setPerspective("red");
            setCandidateRole("open");
          }}>
            Load Dayos example
          </button>
          <button type="button" className="sandbox-secondary-button" onClick={() => {
            setActions([]);
            setExcluded([]);
          }}>
            New draft
          </button>
          <button type="button" className="sandbox-secondary-button" onClick={copyScenario}>
            {shareState}
          </button>
        </div>
      </header>

      <section className="sandbox-context" aria-label="Analysis context">
        <label>
          <span>Calibration</span>
          <select value={league} onChange={(event) => setLeague(event.target.value)}>
            {LEAGUES.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </label>
        <fieldset>
          <legend>Evaluate for</legend>
          <div className="sandbox-segmented">
            {(["blue", "red"] as DraftSide[]).map((side) => (
              <button
                type="button"
                className={perspective === side ? "is-selected" : ""}
                onClick={() => setPerspective(side)}
                key={side}
              >
                {sideLabel(side)}
              </button>
            ))}
          </div>
        </fieldset>
        <label>
          <span>Candidate role</span>
          <select
            value={candidateRole}
            onChange={(event) => setCandidateRole(event.target.value as DraftCandidateRole)}
          >
            <option value="open">Open roles</option>
            <option value="any">Any role</option>
            {Object.entries(ROLE_LABEL).map(([value, label]) => (
              <option value={value} key={value}>{label}</option>
            ))}
          </select>
        </label>
        <div className="sandbox-next">
          <span>Next seat</span>
          <strong>{nextSide ? `${sideLabel(nextSide)} pick` : "Draft complete"}</strong>
        </div>
      </section>

      <section className="sandbox-unavailable" aria-label="Unavailable champions">
        <div>
          <span>Unavailable / banned</span>
          <small>Removed from the legal next-pick ranking.</small>
        </div>
        <div className="sandbox-unavailable-control">
          <input
            type="search"
            list="sandbox-exclusions"
            value={excludeSearch}
            onChange={(event) => setExcludeSearch(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") markUnavailable();
            }}
            placeholder="Add banned champion"
            aria-label="Add unavailable champion"
          />
          <datalist id="sandbox-exclusions">
            {catalog
              .filter(
                (champion) =>
                  !selected.has(champion.name.toLocaleLowerCase()) &&
                  !excluded.includes(champion.name),
              )
              .map((champion) => <option value={champion.name} key={champion.name} />)}
          </datalist>
          <button type="button" className="sandbox-secondary-button" onClick={markUnavailable}>
            Exclude
          </button>
        </div>
        <div className="sandbox-excluded-list">
          {excluded.length ? excluded.map((champion) => (
            <button
              type="button"
              key={champion}
              onClick={() => setExcluded((current) => current.filter((item) => item !== champion))}
              aria-label={`Restore ${champion}`}
            >
              {champion} <span aria-hidden>×</span>
            </button>
          )) : <span>None</span>}
        </div>
      </section>

      <section className="sandbox-workbench">
        <div className="sandbox-board">
          <DraftSideColumn
            side="blue"
            actions={actions}
            nextSide={nextSide}
            onBranch={(index) => setActions((currentActions) => currentActions.slice(0, index))}
          />
          <div className="sandbox-versus" aria-hidden>VS</div>
          <DraftSideColumn
            side="red"
            actions={actions}
            nextSide={nextSide}
            onBranch={(index) => setActions((currentActions) => currentActions.slice(0, index))}
          />
        </div>

        <aside className="sandbox-read" aria-live="polite">
          <p className="blog-kicker">Current model read</p>
          {loading && !current ? (
            <div className="sandbox-skeleton" aria-label="Loading analysis" />
          ) : error ? (
            <p className="error-banner">{error}</p>
          ) : current ? (
            <>
              <div className="sandbox-current-number">
                <span>{sideLabel(perspective)} projected draft WR</span>
                <strong>{(100 * current.projected_wr).toFixed(2)}%</strong>
              </div>
              <div className="sandbox-balance" aria-label={`${sideLabel(perspective)} projected draft win rate`}>
                <span style={{ width: `${100 * current.projected_wr}%` }} />
              </div>
              {nextSide && analysis?.recommendations.length ? (
                <div className="sandbox-shortlist">
                  <div className="sandbox-shortlist-head">
                    <span>Best {sideLabel(analysis.recommendation_side)} responses</span>
                    <small>Projected change</small>
                  </div>
                  {analysis.recommendations.slice(0, 3).map((row) => (
                    <button
                      type="button"
                      key={`${row.champion}-${row.role}`}
                      onClick={() => addPick(row.champion, row.role)}
                    >
                      <span
                        className="sandbox-champion-portrait"
                        aria-hidden
                        style={{ backgroundImage: `url("${champIconUrl(row.champion)}")` }}
                      />
                      <span>
                        <strong>{row.champion}</strong>
                        <small>{row.role ? ROLE_LABEL[row.role] : "Role open"}</small>
                      </span>
                      <em>{signedPp(row.delta_pp)}</em>
                    </button>
                  ))}
                </div>
              ) : null}
              <dl className="sandbox-read-ledger">
                <div><dt>Model confidence</dt><dd>{(100 * current.confidence).toFixed(0)}%</dd></div>
                <div><dt>Calibration</dt><dd>{current.score.calibration.source}</dd></div>
                <div><dt>Chosen picks</dt><dd>{actions.length}/10</dd></div>
              </dl>
              {responseCheckpoint && actions.length >= 3 && (
                <p className="sandbox-answer">
                  After the first response pair, {sideLabel(perspective)} sat at{" "}
                  <strong>{(100 * responseCheckpoint.projected_wr).toFixed(2)}%</strong> in this
                  partial-draft model ({signedPp(100 * responseCheckpoint.projected_wr - 50)} versus
                  even).
                </p>
              )}
            </>
          ) : null}
        </aside>
      </section>

      {nextSide && (
        <section className="sandbox-picker" aria-labelledby="manual-pick-heading">
          <div>
            <p className="blog-kicker">Manual branch</p>
            <h2 id="manual-pick-heading" className="font-display text-lg">
              Add {sideLabel(nextSide)}&apos;s next pick
            </h2>
          </div>
          <div className="sandbox-picker-control">
            <input
              type="search"
              list="sandbox-champions"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") addSearchPick();
              }}
              placeholder="Champion name"
              aria-label="Champion name"
            />
            <datalist id="sandbox-champions">
              {available.map((champion) => <option value={champion.name} key={champion.name} />)}
            </datalist>
            <button type="button" className="btn-primary" onClick={addSearchPick}>Add pick</button>
          </div>
        </section>
      )}

      <section className="sandbox-ranking" aria-labelledby="ranking-heading">
        <div className="sandbox-section-head">
          <div>
            <p className="blog-kicker">Counterfactual ranking</p>
            <h2 id="ranking-heading" className="font-display text-xl">
              Best next responses for {sideLabel(analysis?.recommendation_side ?? nextSide ?? perspective)}
            </h2>
          </div>
          <p>
            Sorted by projected draft WR after the pick. Change is measured against the current
            state.
          </p>
        </div>
        {loading && analysis ? <p className="status-hint">Updating model…</p> : null}
        {error && analysis ? <p className="error-banner">{error}</p> : null}
        {analysis && nextSide ? (
          <RecommendationTable
            rows={analysis.recommendations}
            perspective={analysis.recommendation_side}
            onPick={(row) => addPick(row.champion, row.role)}
          />
        ) : (
          <p className="sandbox-empty">Complete draft. Branch from an earlier pick to test another response.</p>
        )}
      </section>

      <section className="sandbox-audit" aria-labelledby="audit-heading">
        <div className="sandbox-section-head">
          <div>
            <p className="blog-kicker">Decision trace</p>
            <h2 id="audit-heading" className="font-display text-xl">Where the model moved</h2>
          </div>
          <p>Every row uses the selected side&apos;s perspective.</p>
        </div>
        <div className="table-scroll">
          <table className="data-table sandbox-timeline">
            <thead>
              <tr><th>Seat</th><th>Pick</th><th>Role</th><th>Projected WR</th><th>Change</th><th>Confidence</th></tr>
            </thead>
            <tbody>
              {analysis?.timeline.map((row) => {
                const sidePick = actions
                  .slice(0, row.pick_number)
                  .filter((action) => action.side === row.side).length;
                return (
                  <tr key={`${row.pick_number}-${row.champion}`}>
                    <td className="font-mono">{row.side === "blue" ? "B" : "R"}{sidePick}</td>
                    <td><strong>{row.champion}</strong></td>
                    <td>{row.role ? ROLE_LABEL[row.role] : "Open"}</td>
                    <td className="font-mono">{(100 * row.projected_wr).toFixed(2)}%</td>
                    <td className={`font-mono ${row.delta_pp >= 0 ? "sandbox-positive" : "sandbox-negative"}`}>
                      {signedPp(row.delta_pp)}
                    </td>
                    <td className="font-mono">{(100 * row.confidence).toFixed(0)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <details className="sandbox-method">
        <summary>How to read this model</summary>
        <p>
          Unfilled seats are held at the fitted average, so changes compare one draft branch with
          another at the same point. Champion effects are regularized from professional games;
          known same-role matchups add a smaller lane-pair term. Partial pick states are not
          separately calibrated outcome probabilities, and the ranking does not know a player&apos;s
          champion pool, scrim plan, or hidden flex intent.
        </p>
        <p className="font-mono">{analysis?.note}</p>
      </details>
    </div>
  );
}
