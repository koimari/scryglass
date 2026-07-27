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

const ROLE_LABEL: Record<DraftRole, string> = {
  top: "Top",
  jng: "Jungle",
  mid: "Mid",
  bot: "ADC",
  sup: "Support",
};
const DRAFT_ROLES: DraftRole[] = ["top", "jng", "mid", "bot", "sup"];

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
            <th>{sideLabel(perspective)} model share</th>
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
                <button
                  type="button"
                  className="sandbox-use-pick"
                  onClick={() => onPick(row)}
                  aria-label={`Draft ${row.champion}${row.role ? ` as ${ROLE_LABEL[row.role]}` : ""}`}
                >
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
  initialActions = [],
  initialExcluded = [],
  initialPerspective = "blue",
  initialLeague = "LCS",
}: Props) {
  const [actions, setActions] = useState<DraftAction[]>(initialActions);
  const [excluded, setExcluded] = useState<string[]>(initialExcluded);
  const [perspective, setPerspective] = useState<DraftSide>(initialPerspective);
  const [league, setLeague] = useState(initialLeague);
  const [candidateRole, setCandidateRole] = useState<DraftCandidateRole>("open");
  const [search, setSearch] = useState("");
  const [pendingChampion, setPendingChampion] = useState<DraftChampion | null>(null);
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
  const openRoles = analysis?.open_roles ?? DRAFT_ROLES.slice();
  const pickerChampions = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return available.filter((champion) => {
      if (query && !champion.name.toLocaleLowerCase().includes(query)) return false;
      if (candidateRole === "open") {
        return champion.roles.some((role) => openRoles.includes(role));
      }
      return true;
    });
  }, [available, candidateRole, openRoles, search]);
  const responseCheckpoint = analysis?.timeline[2] ?? null;
  const current = analysis?.current;
  const pendingRoles = pendingChampion
    ? candidateRole !== "open" && candidateRole !== "any"
      ? openRoles.includes(candidateRole)
        ? [candidateRole]
        : []
      : candidateRole === "open"
        ? pendingChampion.roles.filter((role) => openRoles.includes(role))
        : openRoles
    : [];

  const addPick = (champion: string, role: DraftRole | null) => {
    if (!nextSide) return;
    setActions((currentActions) => [...currentActions, { side: nextSide, champion, role }]);
    setSearch("");
    setPendingChampion(null);
    setError(null);
  };

  const chooseChampion = (champion: DraftChampion) => {
    const legalRoles =
      candidateRole !== "open" && candidateRole !== "any"
        ? openRoles.includes(candidateRole)
          ? [candidateRole]
          : []
        : candidateRole === "open"
          ? champion.roles.filter((role) => openRoles.includes(role))
          : openRoles;
    if (!legalRoles.length) {
      setError(`${champion.name} has no legal role under this filter.`);
      return;
    }
    if (legalRoles.length === 1) {
      addPick(champion.name, legalRoles[0]);
      return;
    }
    setPendingChampion(champion);
    setError(null);
  };

  const addSearchPick = () => {
    const exact = pickerChampions.find(
      (candidate) => candidate.name.toLocaleLowerCase() === search.trim().toLocaleLowerCase(),
    );
    if (exact) {
      chooseChampion(exact);
      return;
    }
    if (pickerChampions.length === 1) {
      chooseChampion(pickerChampions[0]);
      return;
    }
    setError("Choose a champion from the visible grid.");
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
          <span><strong>Estimand</strong> partial-draft comparison</span>
          <span><strong>Model pool</strong> {catalog.length} pro-play champions</span>
          <span><strong>Complete board</strong> full-composition model</span>
        </div>
        <div className="sandbox-header-actions">
          <button type="button" className="btn-primary" onClick={() => {
            setActions([]);
            setExcluded([]);
            setPerspective("blue");
            setCandidateRole("open");
            setSearch("");
            setPendingChampion(null);
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
            onChange={(event) => {
              setCandidateRole(event.target.value as DraftCandidateRole);
              setPendingChampion(null);
            }}
          >
            <option value="open">Observed open roles</option>
            <option value="any">Any champion / open role</option>
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
            onBranch={(index) => {
              setActions((currentActions) => currentActions.slice(0, index));
              setPendingChampion(null);
            }}
          />
          <div className="sandbox-versus" aria-hidden>VS</div>
          <DraftSideColumn
            side="red"
            actions={actions}
            nextSide={nextSide}
            onBranch={(index) => {
              setActions((currentActions) => currentActions.slice(0, index));
              setPendingChampion(null);
            }}
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
                <span>{sideLabel(perspective)} model share</span>
                <strong>{(100 * current.projected_wr).toFixed(2)}%</strong>
              </div>
              <div className="sandbox-balance" aria-label={`${sideLabel(perspective)} model comparison share`}>
                <span style={{ width: `${100 * current.projected_wr}%` }} />
              </div>
              {nextSide && analysis?.recommendations.length ? (
                <div className="sandbox-shortlist">
                  <div className="sandbox-shortlist-head">
                    <span>Best {sideLabel(analysis.recommendation_side)} responses</span>
                    <small>Model change</small>
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
                  partial-draft comparison ({signedPp(100 * responseCheckpoint.projected_wr - 50)}
                  versus even).
                </p>
              )}
            </>
          ) : null}
        </aside>
      </section>

      {nextSide && (
        <section className="sandbox-picker" aria-labelledby="manual-pick-heading">
          <div className="sandbox-picker-head">
            <div>
              <p className="blog-kicker">Champion board</p>
              <h2 id="manual-pick-heading" className="font-display text-lg">
                Add {sideLabel(nextSide)}&apos;s next pick
              </h2>
            </div>
            <p>Choose a role, then click a champion portrait. Flex picks ask for a role before they are added.</p>
          </div>

          <div className="sandbox-role-filters" role="group" aria-label="Champion role filter">
            <button
              type="button"
              className={candidateRole === "open" ? "is-selected" : ""}
              aria-pressed={candidateRole === "open"}
              onClick={() => {
                setCandidateRole("open");
                setPendingChampion(null);
              }}
            >
              Pro roles
            </button>
            <button
              type="button"
              className={candidateRole === "any" ? "is-selected" : ""}
              aria-pressed={candidateRole === "any"}
              onClick={() => {
                setCandidateRole("any");
                setPendingChampion(null);
              }}
            >
              All open
            </button>
            {DRAFT_ROLES.map((role) => (
              <button
                type="button"
                className={candidateRole === role ? "is-selected" : ""}
                aria-pressed={candidateRole === role}
                disabled={!openRoles.includes(role)}
                onClick={() => {
                  setCandidateRole(role);
                  setPendingChampion(null);
                }}
                key={role}
              >
                {ROLE_LABEL[role]}
              </button>
            ))}
          </div>

          <div className="sandbox-picker-control">
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") addSearchPick();
              }}
              placeholder={`Search ${available.length} available champions`}
              aria-label="Search champions"
            />
            {search ? (
              <button
                type="button"
                className="sandbox-secondary-button"
                onClick={() => setSearch("")}
              >
                Clear
              </button>
            ) : null}
          </div>

          {pendingChampion ? (
            <div className="sandbox-role-choice" role="group" aria-label={`Choose ${pendingChampion.name}'s role`}>
              <span>
                Draft <strong>{pendingChampion.name}</strong> as
              </span>
              {pendingRoles.map((role) => (
                <button
                  type="button"
                  onClick={() => addPick(pendingChampion.name, role)}
                  key={role}
                >
                  {ROLE_LABEL[role]}
                </button>
              ))}
              <button type="button" onClick={() => setPendingChampion(null)}>Cancel</button>
            </div>
          ) : null}

          <div className="sandbox-champion-grid" aria-label="Available champions">
            {pickerChampions.map((champion) => {
              const explicitRole =
                candidateRole !== "open" && candidateRole !== "any" ? candidateRole : null;
              const roleGames = explicitRole ? Number(champion.role_games?.[explicitRole] ?? 0) : null;
              const observedOpenRoles = champion.roles.filter((role) => openRoles.includes(role));
              const detail = explicitRole
                ? roleGames
                  ? `${roleGames} pro games`
                  : "Unseen in role"
                : candidateRole === "any"
                  ? "Choose role"
                  : observedOpenRoles.map((role) => ROLE_LABEL[role]).join(" · ");
              return (
                <button
                  type="button"
                  className="sandbox-champion-button"
                  onClick={() => chooseChampion(champion)}
                  aria-label={`Draft ${champion.name}${explicitRole ? ` as ${ROLE_LABEL[explicitRole]}` : ""}`}
                  key={champion.name}
                >
                  <span
                    className="sandbox-champion-card-portrait"
                    aria-hidden
                    style={{ backgroundImage: `url("${champIconUrl(champion.name)}")` }}
                  />
                  <strong>{champion.name}</strong>
                  <small>{detail}</small>
                </button>
              );
            })}
          </div>
          {!pickerChampions.length ? (
            <p className="sandbox-empty">No available champion matches this search and role filter.</p>
          ) : null}
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
            Sorted by model share after the pick. Change is measured against the current state; these
            partial-draft comparisons are not calibrated live win probabilities.
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
        ) : nextSide ? (
          <p className="status-hint">{error ? "Ranking unavailable." : "Loading legal responses…"}</p>
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
              <tr><th>Seat</th><th>Pick</th><th>Role</th><th>Model share</th><th>Change</th><th>Confidence</th></tr>
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
          Until all ten picks are present, unfilled seats are neutralized so changes compare one
          draft branch with another at the same point. Those partial-draft values are not separately
          calibrated outcome probabilities. A complete board switches to the full-composition model:
          role-aware champion effects, within-team synergy, all 25 enemy interactions, and a sparse
          low-rank residual. The sandbox does not know a player&apos;s champion pool, scrim plan, or
          hidden flex intent. You can place any catalog champion in any still-open role; when that
          champion-role pair has no pro sample, its role-specific direct effect stays neutral while
          champion-level synergy and enemy interactions still apply, and the UI marks the role as
          unseen.
        </p>
        <p className="font-mono">{analysis?.note}</p>
      </details>
    </div>
  );
}
