"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DRAFT_PICK_ORDER,
  DRAFT_POLICY_MIN_ROLE_GAMES,
  DRAFT_ROLES,
} from "@/lib/draftRules";
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
import type { CompositionRuntimeMetadata } from "@/lib/draftComposition";
import {
  patchContractFromSource,
  patchContractsFromSource,
} from "@/lib/patch";

const ROLE_LABEL: Record<DraftRole, string> = {
  top: "Top",
  jng: "Jungle",
  mid: "Mid",
  bot: "ADC",
  sup: "Support",
};

const LIVE_RECOMMENDATION_LIMIT = 6;
const LIVE_ANALYSIS_DEBOUNCE_MS = 220;

const LEAGUES = [
  "LCK",
  "LPL",
  "LEC",
  "LCS",
  "CBLOL",
  "LCP",
  "MSI",
  "EWC",
];

type Props = {
  catalog: DraftChampion[];
  initialActions?: DraftAction[];
  initialExcluded?: string[];
  initialPerspective?: DraftSide;
  initialLeague?: string;
  initialPublicPatch?: string;
  modelMetadata?: CompositionRuntimeMetadata | null;
};

function sideLabel(side: DraftSide): string {
  return side === "blue" ? "Blue" : "Red";
}

function signedChange(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} points`;
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
                <span className="sandbox-pick-empty">
                  {isNext ? "Next decision" : "Open"}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function RecommendationList({
  rows,
  perspective,
  onPick,
}: {
  rows: DraftRecommendation[];
  perspective: DraftSide;
  onPick: (row: DraftRecommendation) => void;
}) {
  if (!rows.length) {
    return (
      <p className="sandbox-empty">
        No observed champion-role candidates match this role filter.
      </p>
    );
  }
  return (
    <div
      className="sandbox-shortlist"
      aria-label={`${sideLabel(perspective)} next best board actions`}
    >
      <div className="sandbox-shortlist-head">
        <span>Top next actions</span>
        <small>Projected value</small>
      </div>
      {rows.slice(0, 6).map((row, index) => (
        <button
          type="button"
          className="sandbox-use-pick"
          onClick={() => onPick(row)}
          key={`${row.champion}-${row.role}`}
          aria-label={`Draft ${row.champion}${row.role ? ` as ${ROLE_LABEL[row.role]}` : ""}`}
        >
          <span className="font-mono sandbox-rec-rank">{index + 1}</span>
          <span
            className="sandbox-champion-portrait"
            aria-hidden
            style={{ backgroundImage: `url("${champIconUrl(row.champion)}")` }}
          />
          <span>
            <strong>{row.champion}</strong>
            <small>{row.role ? ROLE_LABEL[row.role] : "Unassigned"}</small>
          </span>
          <span className="font-mono sandbox-rec-value">
            {(100 * row.projected_value).toFixed(1)}%
          </span>
          <em className={row.delta_points >= 0 ? "sandbox-positive" : "sandbox-negative"}>
            {signedChange(row.delta_points)}
          </em>
        </button>
      ))}
    </div>
  );
}

export function DraftSandbox({
  catalog,
  initialActions = [],
  initialExcluded = [],
  initialPerspective = "blue",
  initialLeague = "LCS",
  initialPublicPatch,
  modelMetadata = null,
}: Props) {
  const [actions, setActions] = useState<DraftAction[]>(initialActions);
  const [excluded, setExcluded] = useState<string[]>(initialExcluded);
  const [perspective, setPerspective] = useState<DraftSide>(initialPerspective);
  const [league, setLeague] = useState(initialLeague);
  const [candidateRole, setCandidateRole] = useState<DraftCandidateRole>("open");
  const [search, setSearch] = useState("");
  const [pendingChampion, setPendingChampion] = useState<DraftChampion | null>(null);
  const [pickerAnnouncement, setPickerAnnouncement] = useState("");
  const [excludeSearch, setExcludeSearch] = useState("");
  const [analysisResponse, setAnalysisResponse] = useState<{
    key: string;
    value: DraftSandboxResult;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shareState, setShareState] = useState("Copy scenario");
  const [includeRecommendations, setIncludeRecommendations] = useState(false);

  const analysisPatchContracts = patchContractsFromSource(
    modelMetadata?.analysis_patches ?? [],
  );
  const patchSpecificContracts = patchContractsFromSource(
    modelMetadata?.supported_patches ?? [],
  );
  const patchSpecificPublicPatches = new Set(
    patchSpecificContracts.map((contract) => contract.public_patch),
  );
  const latestPatchSpecific =
    patchSpecificContracts[patchSpecificContracts.length - 1]?.public_patch ??
    null;
  const latestObservedPatch = patchContractFromSource(
    modelMetadata?.latest_observed_patch,
  )?.public_patch ?? null;
  const [publicPatch, setPublicPatch] = useState(
    initialPublicPatch ?? latestObservedPatch ?? "",
  );

  const [analysisMode, setAnalysisMode] = useState<"manual" | "live">("manual");

  const sandboxHeadingRef = useRef<HTMLHeadingElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const firstRoleChoiceRef = useRef<HTMLButtonElement | null>(null);
  const roleChoiceTriggerRef = useRef<HTMLElement | null>(null);
  const roleChoiceReturnModeRef = useRef<"cancel" | "selection" | null>(
    null,
  );

  const nextSide =
    actions.length < DRAFT_PICK_ORDER.length ? DRAFT_PICK_ORDER[actions.length] : null;

  const openRoles = useMemo(() => {
    if (!nextSide) return [];
    const occupied = new Set(
      actions
        .filter((action) => action.side === nextSide)
        .map((action) => action.role)
        .filter((role): role is DraftRole => Boolean(role)),
    );
    return DRAFT_ROLES.filter((role) => !occupied.has(role));
  }, [actions, nextSide]);

  const buildRequest = useCallback(
    (recommendationMode: boolean) => ({
      actions,
      perspective,
      next_side: nextSide ?? perspective,
      candidate_role: candidateRole,
      excluded,
      league,
      public_patch: publicPatch,
      limit: nextSide
        ? recommendationMode
          ? (analysisMode === "live"
            ? LIVE_RECOMMENDATION_LIMIT
            : Math.max(10, LIVE_RECOMMENDATION_LIMIT))
          : 1
        : 1,
      include_recommendations: recommendationMode,
    }),
    [
      actions,
      candidateRole,
      excluded,
      league,
      nextSide,
      perspective,
      publicPatch,
      analysisMode,
    ],
  );

  const analysisRequest = useMemo(
    () => buildRequest(includeRecommendations),
    [buildRequest, includeRecommendations],
  );
  const requestKey = JSON.stringify(analysisRequest);
  const analysis = analysisResponse?.key === requestKey ? analysisResponse.value : null;
  const current = analysis?.current;

  const selected = useMemo(
    () =>
      new Set(actions.map((action) => action.champion.toLocaleLowerCase())),
    [actions],
  );

  const available = useMemo(() => {
    const excludedNames = new Set(excluded.map((champion) => champion.toLocaleLowerCase()));
    return catalog.filter(
      (champion) =>
        !selected.has(champion.name.toLocaleLowerCase()) &&
        !excludedNames.has(champion.name.toLocaleLowerCase()),
    );
  }, [catalog, excluded, selected]);

  const pickerChampions = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return available.filter((champion) => {
      if (query && !champion.name.toLocaleLowerCase().includes(query)) {
        return false;
      }
      if (candidateRole === "open") {
        return champion.roles.some((role) => openRoles.includes(role));
      }
      if (candidateRole === "any") {
        return champion.roles.some((role) => openRoles.includes(role));
      }
      return (
        openRoles.includes(candidateRole) && champion.roles.includes(candidateRole)
      );
    });
  }, [available, candidateRole, openRoles, search]);

  const responseCheckpoint = analysis?.timeline[2] ?? null;
  const pendingRoles = pendingChampion
    ? candidateRole !== "open" && candidateRole !== "any"
      ? openRoles.includes(candidateRole)
        ? [candidateRole]
        : []
      : candidateRole === "open"
        ? pendingChampion.roles.filter((role) => openRoles.includes(role))
        : pendingChampion.roles.filter((role) => openRoles.includes(role))
    : [];

  const fetchAnalysis = useCallback(
    async (
    recommendationMode: boolean,
    signal?: AbortSignal,
    force = false,
    ): Promise<void> => {
      if (!publicPatch) return;
      const request = buildRequest(recommendationMode);
      const payload = JSON.stringify(request);
      if (!force && analysisResponse?.key === payload) return;
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("/api/draft-sandbox", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal,
          body: payload,
        });
        const raw = await response.json();
        if (!response.ok) {
          throw new Error(raw.error || `draft sandbox ${response.status}`);
        }
        if (!signal?.aborted) {
          setAnalysisResponse({ key: payload, value: raw as DraftSandboxResult });
        }
      } catch (requestError) {
        if (
          requestError instanceof DOMException &&
          requestError.name === "AbortError"
        ) {
          return;
        }
        setError(
          requestError instanceof Error ? requestError.message : String(requestError),
        );
      } finally {
        if (!signal?.aborted) {
          setLoading(false);
        }
      }
    },
    [analysisResponse?.key, buildRequest, publicPatch],
  );

  const runAnalysisNow = async () => {
    if (!publicPatch) {
      setError("Select an observed patch context before running analysis.");
      return;
    }
    await fetchAnalysis(includeRecommendations, undefined, true);
  };

  useEffect(() => {
    if (analysisMode !== "live" || !publicPatch) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void fetchAnalysis(includeRecommendations, controller.signal);
    }, LIVE_ANALYSIS_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [
    actions,
    candidateRole,
    excluded,
    league,
    nextSide,
    perspective,
    publicPatch,
    analysisMode,
    includeRecommendations,
    fetchAnalysis,
  ]);

  useEffect(() => {
    if (pendingChampion) {
      firstRoleChoiceRef.current?.focus();
      return;
    }
    const returnMode = roleChoiceReturnModeRef.current;
    if (!returnMode) return;
    const trigger = roleChoiceTriggerRef.current;
    const fallback = searchInputRef.current ?? sandboxHeadingRef.current;
    const target = returnMode === "cancel" && trigger?.isConnected ? trigger : fallback;
    roleChoiceReturnModeRef.current = null;
    roleChoiceTriggerRef.current = null;
    target?.focus();
  }, [pendingChampion]);

  const addPick = (champion: string, role: DraftRole | null) => {
    setActions((currentActions) => {
      const side = DRAFT_PICK_ORDER[currentActions.length];
      if (!side || !role) return currentActions;
      const occupied = new Set(
        currentActions
          .filter((action) => action.side === side)
          .map((action) => action.role),
      );
      if (occupied.has(role)) return currentActions;
      return [...currentActions, { side, champion, role }];
    });
    setCandidateRole("open");
    setSearch("");
    setPendingChampion(null);
    setError(null);
  };

  const chooseChampion = (
    champion: DraftChampion,
    trigger: HTMLElement | null = null,
  ) => {
    const legalRoles =
      candidateRole !== "open" && candidateRole !== "any"
        ? openRoles.includes(candidateRole) && champion.roles.includes(candidateRole)
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
    roleChoiceTriggerRef.current = trigger;
    roleChoiceReturnModeRef.current = null;
    setPendingChampion(champion);
    setPickerAnnouncement(
      candidateRole === "any"
        ? `${champion.name} selected for a manual what-if. Choose an open role.`
        : `${champion.name} can fit in multiple open roles. Choose one now.`,
    );
    setError(null);
  };

  const addSearchPick = () => {
    const exact = pickerChampions.find(
      (candidate) =>
        candidate.name.toLocaleLowerCase() === search.trim().toLocaleLowerCase(),
    );
    if (exact) {
      chooseChampion(exact, searchInputRef.current);
      return;
    }
    if (pickerChampions.length === 1) {
      chooseChampion(pickerChampions[0], searchInputRef.current);
      return;
    }
    setError("Choose a champion from the visible grid.");
  };

  const selectPendingRole = (role: DraftRole) => {
    if (!pendingChampion) return;
    const champion = pendingChampion.name;
    roleChoiceReturnModeRef.current = "selection";
    setPickerAnnouncement(
      `${champion} added as ${ROLE_LABEL[role]}. Focus returned to champion search.`,
    );
    addPick(champion, role);
  };

  const cancelPendingRole = () => {
    if (!pendingChampion) return;
    roleChoiceReturnModeRef.current = "cancel";
    setPickerAnnouncement(
      `${pendingChampion.name} role selection canceled. Focus returned to the champion choice.`,
    );
    setPendingChampion(null);
  };

  const markUnavailable = () => {
    const champion = catalog.find(
      (candidate) =>
        candidate.name.toLocaleLowerCase() ===
        excludeSearch.trim().toLocaleLowerCase(),
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
    url.searchParams.delete("patch");
    if (publicPatch) {
      url.searchParams.set("public_patch", publicPatch);
    } else {
      url.searchParams.delete("public_patch");
    }
    await navigator.clipboard.writeText(url.toString());
    setShareState("Copied");
    window.setTimeout(() => setShareState("Copy scenario"), 1600);
  };

  const requestLabel = includeRecommendations ? "with" : "without";
  const modeLabel = analysisMode;
  const patchSupportText = latestObservedPatch
    ? `Patch ${latestObservedPatch} is current observed; values use ${latestPatchSpecific ? "patch-specific" : "pooled"} terms.`
    : "No observed patch is available for analysis.";
  const isExplicitRole = candidateRole !== "open" && candidateRole !== "any";
  const rolePolicyHint = isExplicitRole
    ? `Role fixed to an explicit position.`
    : candidateRole === "any"
      ? `Manual any role mode keeps recommendations in supported pro roles only (minimum ${DRAFT_POLICY_MIN_ROLE_GAMES} pro maps).`
      : `Supported pro roles require at least ${DRAFT_POLICY_MIN_ROLE_GAMES} pro maps.`;
  const isDraftLive = analysisMode === "live";
  const boardStateLabel = nextSide ? `${sideLabel(nextSide)} pick` : "Draft complete";

  return (
    <div className="sandbox-page sandbox-draft-shell">
      <p
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {pickerAnnouncement}
      </p>

      <header className="sandbox-hero">
        <div className="sandbox-hero-copy">
          <p className="blog-kicker">Draft sandbox</p>
          <h1
            ref={sandboxHeadingRef}
            className="font-display"
            tabIndex={-1}
          >
            Draft workspace
          </h1>
        </div>
        <div className="sandbox-header-actions">
          <button
            type="button"
            className="sandbox-primary-button"
            onClick={() => {
              setActions([]);
              setExcluded([]);
              setPerspective("blue");
              setCandidateRole("open");
              setSearch("");
              setPendingChampion(null);
              setAnalysisResponse(null);
              setError(null);
            }}
          >
            New draft
          </button>
          <button
            type="button"
            className="sandbox-secondary-button"
            onClick={copyScenario}
          >
            {shareState}
          </button>
        </div>
      </header>

      <section className="sandbox-toolbar" aria-label="Sandbox controls">
        <label className="sandbox-control-block">
          <span>League</span>
          <select value={league} onChange={(event) => setLeague(event.target.value)}>
            {LEAGUES.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </label>

        <label className="sandbox-control-block">
          <span>Patch</span>
          <select
            value={publicPatch}
            onChange={(event) => setPublicPatch(event.target.value)}
            aria-describedby="sandbox-patch-support"
            required
          >
            <option value="" disabled>
              Select patch
            </option>
            {analysisPatchContracts.map((contract) => (
              <option value={contract.public_patch} key={contract.public_patch}>
                {contract.public_patch}
                {contract.public_patch === latestObservedPatch
                  ? " · current"
                  : patchSpecificPublicPatches.has(contract.public_patch)
                    ? " · patch-specific"
                    : " · pooled"}
              </option>
            ))}
          </select>
          <small id="sandbox-patch-support" className="sandbox-mini-copy">
            {patchSupportText}
          </small>
        </label>

        <fieldset className="sandbox-control-block sandbox-side-switch">
          <legend>Perspective</legend>
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

        <label className="sandbox-control-block sandbox-inline-toggle" aria-label="Auto draft analysis">
          <span>Auto update</span>
          <input
            type="checkbox"
            checked={analysisMode === "live"}
            onChange={(event) =>
              setAnalysisMode(event.target.checked ? "live" : "manual")
            }
          />
        </label>

        <label className="sandbox-control-block sandbox-inline-toggle">
          <span>Live recommendations</span>
          <input
            type="checkbox"
            checked={includeRecommendations}
            onChange={(event) => setIncludeRecommendations(event.target.checked)}
          />
        </label>

        <div className="sandbox-next sandbox-control-block">
          <span>Next seat</span>
          <strong>{boardStateLabel}</strong>
        </div>

        {analysisMode === "manual" ? (
          <button
            type="button"
            className="sandbox-primary-button"
            onClick={runAnalysisNow}
            disabled={!publicPatch || loading}
          >
            Evaluate now
          </button>
        ) : null}
      </section>

      <details className="sandbox-unavailable" aria-label="Unavailable champions">
        <summary>
          <span>Unavailable champions</span>
          <small>{excluded.length ? `${excluded.length} excluded` : "None excluded"}</small>
        </summary>
        <div className="sandbox-unavailable-body">
          <div className="sandbox-unavailable-control">
            <input
              type="search"
              list="sandbox-exclusions"
              value={excludeSearch}
              onChange={(event) => setExcludeSearch(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") markUnavailable();
              }}
              placeholder="Add unavailable champion"
              aria-label="Add unavailable champion"
            />
            <datalist id="sandbox-exclusions">
              {catalog
                .filter(
                  (champion) =>
                    !selected.has(champion.name.toLocaleLowerCase()) &&
                    !excluded.includes(champion.name),
                )
                .map((champion) => (
                  <option value={champion.name} key={champion.name} />
                ))}
            </datalist>
            <button
              type="button"
              className="sandbox-secondary-button"
              onClick={markUnavailable}
            >
              Add
            </button>
          </div>
          <div className="sandbox-excluded-list">
            {excluded.length ? (
              excluded.map((champion) => (
                <button
                  type="button"
                  key={champion}
                  onClick={() =>
                    setExcluded((current) =>
                      current.filter((item) => item !== champion),
                    )
                  }
                  aria-label={`Restore ${champion}`}
                >
                  {champion} <span aria-hidden>×</span>
                </button>
              ))
            ) : (
              <span>No exclusions.</span>
            )}
          </div>
        </div>
      </details>

      <section className="sandbox-layout">
        <section className="sandbox-workbench">
          <div className="sandbox-board">
            <DraftSideColumn
              side="blue"
              actions={actions}
              nextSide={nextSide}
              onBranch={(index) => {
                setActions((currentActions) => currentActions.slice(0, index));
                setCandidateRole("open");
                setPendingChampion(null);
              }}
            />
            <div className="sandbox-versus" aria-hidden>
              VS
            </div>
            <DraftSideColumn
              side="red"
              actions={actions}
              nextSide={nextSide}
              onBranch={(index) => {
                setActions((currentActions) => currentActions.slice(0, index));
                setCandidateRole("open");
                setPendingChampion(null);
              }}
            />
          </div>

          <aside className="sandbox-read" aria-live="polite">
            <div className="sandbox-read-head">
              <p className="blog-kicker">Board read</p>
              <h2 id="manual-pick-heading" className="font-display">
                {sideLabel(perspective)} projection
              </h2>
            </div>
            {!publicPatch ? (
              <p className="status-hint">
                Select a patch to run analysis.
              </p>
            ) : loading && !current ? (
              <div className="sandbox-skeleton" aria-label="Loading analysis" />
            ) : error ? (
              <p className="error-banner">{error}</p>
            ) : current ? (
              <>
                <div className="sandbox-current-number">
                  <span>{sideLabel(perspective)} composition score</span>
                  <strong>
                    {(100 * current.projected_value).toFixed(1)}%
                  </strong>
                </div>
                <p className="sandbox-read-footnote">
                  Live mode is {isDraftLive ? "on" : "off"} · {includeRecommendations ? "recommendations" : "board only"}
                </p>
                <div
                  className="sandbox-balance"
                  aria-label={`${sideLabel(perspective)} model comparison value`}
                >
                  <span style={{ width: `${100 * current.projected_value}%` }} />
                </div>
                <details className="sandbox-read-metrics">
                  <summary>Projection details</summary>
                  <dl className="sandbox-read-ledger">
                    <div>
                      <dt>Mode</dt>
                      <dd>{modeLabel} · {requestLabel} recs</dd>
                    </div>
                    <div>
                      <dt>Scope</dt>
                      <dd>{analysis?.candidate_role_policy ?? "supported pro roles"}</dd>
                    </div>
                    <div>
                      <dt>Picks</dt>
                      <dd>{actions.length}/10</dd>
                    </div>
                  </dl>
                  {actions.length >= 3 && responseCheckpoint ? (
                    <p className="sandbox-answer">
                      {sideLabel(perspective)} moved{" "}
                      {signedChange(100 * (responseCheckpoint.projected_value - 0.5))}
                      {" "}
                      after the first full response pair.
                    </p>
                  ) : null}
                </details>
                {nextSide && analysis?.recommendations.length ? (
                  <RecommendationList
                    rows={analysis.recommendations}
                    perspective={analysis.recommendation_side}
                    onPick={(row) => addPick(row.champion, row.role)}
                  />
                ) : null}
                {nextSide ? null : (
                  <p className="sandbox-empty">
                    Draft complete — branch from an earlier pick to compare alternates.
                  </p>
                )}
              </>
            ) : null}
          </aside>
        </section>

        {nextSide && (
          <section className="sandbox-picker" aria-label="Pick next champion">
            <div className="sandbox-picker-head">
              <p className="blog-kicker">Champion board</p>
              <h2 className="font-display text-lg">
                Next: {sideLabel(nextSide)} pick
              </h2>
            </div>

            <div
              className="sandbox-role-filters"
              role="group"
              aria-label="Champion role filter"
            >
              <button
                type="button"
                className={candidateRole === "open" ? "is-selected" : ""}
                aria-pressed={candidateRole === "open"}
                onClick={() => {
                  setCandidateRole("open");
                  setPendingChampion(null);
                }}
              >
                Supported roles
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
                Manual role
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

            <p className="sandbox-role-policy-hint" id="sandbox-role-policy-note">
              {rolePolicyHint}
            </p>

            <div className="sandbox-picker-control">
              <input
                ref={searchInputRef}
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
              <div
                className="sandbox-role-choice"
                role="group"
                aria-labelledby="sandbox-role-choice-label"
              >
                <span id="sandbox-role-choice-label">
                  Draft <strong>{pendingChampion.name}</strong> as
                </span>
                {pendingRoles.map((role, index) => (
                  <button
                    ref={index === 0 ? firstRoleChoiceRef : undefined}
                    type="button"
                    onClick={() => selectPendingRole(role)}
                    key={role}
                  >
                    {ROLE_LABEL[role]}
                  </button>
                ))}
                <button type="button" onClick={cancelPendingRole}>
                  Cancel
                </button>
              </div>
            ) : null}

            <div className="sandbox-champion-grid" aria-label="Available champions">
              {pickerChampions.map((champion) => {
                const explicitRole =
                  candidateRole !== "open" && candidateRole !== "any"
                    ? candidateRole
                    : null;
                const roleGames =
                  explicitRole ? Number(champion.role_games?.[explicitRole] ?? 0) : null;
                const observedOpenRoles = champion.roles.filter((role) =>
                  openRoles.includes(role),
                );
                const detail = explicitRole
                  ? roleGames
                    ? `${roleGames} pro games`
                    : "Unseen in role"
                  : candidateRole === "any"
                    ? "Manual role choice"
                    : observedOpenRoles.map((role) => ROLE_LABEL[role]).join(" · ");
                return (
                  <button
                    type="button"
                    className="sandbox-champion-button"
                    onClick={(event) => chooseChampion(champion, event.currentTarget)}
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
              <p className="sandbox-empty">
                No available champion matches this search and role filter.
              </p>
            ) : null}
          </section>
        )}
      </section>

      <details className="sandbox-method">
        <summary>How to read this model</summary>
        <p>
          Unfilled seats are neutralized so each comparison is made at the same draft depth.
          This is a draft-coverage model output, not a final map outcome prediction.
        </p>
        <p className="font-mono">{analysis?.note}</p>
      </details>
    </div>
  );
}
