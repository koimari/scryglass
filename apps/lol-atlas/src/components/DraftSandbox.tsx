"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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
    return (
      <p className="sandbox-empty">
        No observed champion-role candidates match this role filter.
      </p>
    );
  }
  return (
    <div className="table-scroll sandbox-recommendation-scroll">
      <table className="data-table sandbox-recommendations">
        <thead>
          <tr>
            <th>#</th>
            <th>Candidate</th>
            <th>Role</th>
            <th>
              {sideLabel(perspective)} policy value
            </th>
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
              <td
                className="font-mono"
                title={`Immediately after this pick: ${(100 * row.immediate_value).toFixed(2)} / 100`}
              >
                {(100 * row.projected_value).toFixed(2)}/100
                {row.principal_variation.length ? (
                  <small className="block text-[var(--ink-faint)]">
                    then{" "}
                    {row.principal_variation
                      .map(
                        (action) =>
                          `${action.side === "blue" ? "B" : "R"} ${action.champion}`,
                      )
                      .join(" → ")}
                  </small>
                ) : null}
              </td>
              <td className={`font-mono ${row.delta_points >= 0 ? "sandbox-positive" : "sandbox-negative"}`}>
                {signedChange(row.delta_points)}
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
  initialPublicPatch,
  modelMetadata = null,
}: Props) {
  const [actions, setActions] = useState<DraftAction[]>(initialActions);
  const [excluded, setExcluded] = useState<string[]>(initialExcluded);
  const [perspective, setPerspective] = useState<DraftSide>(initialPerspective);
  const [league, setLeague] = useState(initialLeague);
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
  const [candidateRole, setCandidateRole] = useState<DraftCandidateRole>("open");
  const [search, setSearch] = useState("");
  const [pendingChampion, setPendingChampion] = useState<DraftChampion | null>(null);
  const [pickerAnnouncement, setPickerAnnouncement] = useState("");
  const [excludeSearch, setExcludeSearch] = useState("");
  const [analysisResponse, setAnalysisResponse] = useState<{
    key: string;
    value: DraftSandboxResult;
  } | null>(null);
  const [loading, setLoading] = useState(Boolean(initialPublicPatch));
  const [error, setError] = useState<string | null>(null);
  const [shareState, setShareState] = useState("Copy scenario");
  const sandboxHeadingRef = useRef<HTMLHeadingElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const firstRoleChoiceRef = useRef<HTMLButtonElement | null>(null);
  const roleChoiceTriggerRef = useRef<HTMLElement | null>(null);
  const roleChoiceReturnModeRef = useRef<"cancel" | "selection" | null>(
    null,
  );
  const nextSide =
    actions.length < DRAFT_PICK_ORDER.length
      ? DRAFT_PICK_ORDER[actions.length]
      : null;
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
  const requestKey = useMemo(
    () =>
      JSON.stringify({
        actions,
        candidateRole,
        excluded,
        league,
        publicPatch,
        nextSide,
        perspective,
      }),
    [
      actions,
      candidateRole,
      excluded,
      league,
      nextSide,
      perspective,
      publicPatch,
    ],
  );
  const analysis =
    analysisResponse?.key === requestKey ? analysisResponse.value : null;

  useEffect(() => {
    if (!publicPatch) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
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
            public_patch: publicPatch,
            limit: nextSide ? 15 : 1,
          }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `draft sandbox ${response.status}`);
        if (controller.signal.aborted) return;
        setAnalysisResponse({
          key: requestKey,
          value: payload as DraftSandboxResult,
        });
      } catch (requestError) {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(requestError instanceof Error ? requestError.message : String(requestError));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 120);
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
    requestKey,
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
    const target =
      returnMode === "cancel" && trigger?.isConnected ? trigger : fallback;
    roleChoiceReturnModeRef.current = null;
    roleChoiceTriggerRef.current = null;
    target?.focus();
  }, [pendingChampion]);

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
  const pickerChampions = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return available.filter((champion) => {
      if (query && !champion.name.toLocaleLowerCase().includes(query)) return false;
      if (candidateRole === "open") {
        return champion.roles.some((role) => openRoles.includes(role));
      }
      if (candidateRole !== "any") {
        return (
          openRoles.includes(candidateRole) &&
          champion.roles.includes(candidateRole)
        );
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
        ? openRoles.includes(candidateRole) &&
          champion.roles.includes(candidateRole)
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
        ? `${champion.name} selected for a manual what-if. Choose an open role. Policy rankings remain limited to supported pro roles.`
        : `${champion.name} is a flex pick. Choose one of its supported open roles.`,
    );
    setError(null);
  };

  const addSearchPick = () => {
    const exact = pickerChampions.find(
      (candidate) => candidate.name.toLocaleLowerCase() === search.trim().toLocaleLowerCase(),
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

  return (
    <div className="sandbox-page">
      <p
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {pickerAnnouncement}
      </p>
      <header className="page-header">
        <p className="blog-kicker">Model lab · Draft counterfactual</p>
        <h1
          ref={sandboxHeadingRef}
          className="font-display mt-2 text-3xl"
          tabIndex={-1}
        >
          Draft sandbox
        </h1>
        <p className="lede">
          Replay a pick sequence, measure when the model moved, and compare legal branches under a
          bounded response policy. A root beam advances the strongest immediate legal actions through
          up to two later pro-role picks.
        </p>
        <div className="micro-log mt-4">
          <span><strong>Estimand</strong> partial-draft comparison</span>
          <span><strong>Model pool</strong> {catalog.length} pro-play champions</span>
          <span><strong>Complete board</strong> experimental composition value</span>
          <span>
            <strong>Training maps</strong> {modelMetadata?.n_games_fit ?? "unverified"}
          </span>
          <span>
            <strong>Through</strong> {modelMetadata?.date_max?.slice(0, 10) ?? "unverified"}
          </span>
        </div>
        {modelMetadata?.validation?.future_patch_holdout ? (
          <p className="status-hint">
            Shift warning: future-patch holdout ECE{" "}
            {(
              100 *
              Number(modelMetadata.validation.future_patch_holdout.ece_10 ?? 0)
            ).toFixed(2)}
            %. Treat patch transfer as experimental until the replacement model passes the
            predeclared drift gate.
          </p>
        ) : null}
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
          <span>League context</span>
          <select value={league} onChange={(event) => setLeague(event.target.value)}>
            {LEAGUES.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </label>
        <label>
          <span>Patch context</span>
          <select
            value={publicPatch}
            onChange={(event) => setPublicPatch(event.target.value)}
            aria-describedby="sandbox-patch-support"
            required
          >
            <option value="" disabled>
              Select a patch
            </option>
            {analysisPatchContracts.map((contract) => (
              <option
                value={contract.public_patch}
                key={contract.public_patch}
              >
                {contract.public_patch}
                {contract.public_patch === latestObservedPatch
                  ? " · current observed"
                  : patchSpecificPublicPatches.has(contract.public_patch)
                    ? " · patch-specific"
                    : " · pooled holdout"}
              </option>
            ))}
          </select>
          <small id="sandbox-patch-support">
            {latestObservedPatch ? (
              <>
                <strong>
                  Current observed patch {latestObservedPatch} uses pooled
                  composition terms.
                </strong>{" "}
                It is an uncalibrated recommendation utility, not a
                patch-specific win rate.
                {latestPatchSpecific
                  ? ` Patch-specific terms end at ${latestPatchSpecific}.`
                  : ""}
              </>
            ) : (
              "No observed patch is available for analysis."
            )}
          </small>
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
            aria-describedby="sandbox-role-policy-note"
            onChange={(event) => {
              setCandidateRole(event.target.value as DraftCandidateRole);
              setPendingChampion(null);
            }}
          >
            <option value="open">
              Supported pro roles · {DRAFT_POLICY_MIN_ROLE_GAMES}+ maps
            </option>
            <option value="any">Manual what-if · any open role</option>
            {Object.entries(ROLE_LABEL).map(([value, label]) => (
              <option
                value={value}
                disabled={!openRoles.includes(value as DraftRole)}
                key={value}
              >
                {label}
              </option>
            ))}
          </select>
          <small id="sandbox-role-policy-note">
            Policy rankings and look-ahead search require at least{" "}
            {DRAFT_POLICY_MIN_ROLE_GAMES} pro maps in that champion-role pair.
            Manual what-if mode can place an unsupported pair on the board, but
            never adds it to policy search.
          </small>
        </label>
        <div className="sandbox-next">
          <span>Next seat</span>
          <strong>{nextSide ? `${sideLabel(nextSide)} pick` : "Draft complete"}</strong>
        </div>
      </section>

      <section className="sandbox-unavailable" aria-label="Unavailable champions">
        <div>
          <span>Unavailable champions</span>
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
              setCandidateRole("open");
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
              setCandidateRole("open");
              setPendingChampion(null);
            }}
          />
        </div>

        <aside className="sandbox-read" aria-live="polite">
          <p className="blog-kicker">Current model read</p>
          {!publicPatch ? (
            <p className="status-hint">
              Select an observed patch context to begin analysis.
            </p>
          ) : loading && !current ? (
            <div className="sandbox-skeleton" aria-label="Loading analysis" />
          ) : error ? (
            <p className="error-banner">{error}</p>
          ) : current ? (
            <>
              <div className="sandbox-current-number">
                <span>
                  {sideLabel(perspective)}{" "}
                  experimental policy value
                </span>
                <strong>
                  {(100 * current.projected_value).toFixed(2)}/100
                </strong>
              </div>
              <div className="sandbox-balance" aria-label={`${sideLabel(perspective)} model comparison value`}>
                <span style={{ width: `${100 * current.projected_value}%` }} />
              </div>
              {nextSide && analysis?.recommendations.length ? (
                <div className="sandbox-shortlist">
                  <div className="sandbox-shortlist-head">
                    <span>Highest response-aware branches for {sideLabel(analysis.recommendation_side)}</span>
                    <small>Value change</small>
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
                      <em>{signedChange(row.delta_points)}</em>
                    </button>
                  ))}
                </div>
              ) : null}
              <dl className="sandbox-read-ledger">
                <div>
                  <dt>Uncertainty</dt>
                  <dd>
                    Not a probability interval
                  </dd>
                </div>
                <div><dt>Probability gate</dt><dd>Withheld · chronological benchmark failed</dd></div>
                <div><dt>Chosen picks</dt><dd>{actions.length}/10</dd></div>
              </dl>
              {responseCheckpoint && actions.length >= 3 && (
                <p className="sandbox-answer">
                  After the first response pair, {sideLabel(perspective)} sat at{" "}
                  <strong>{(100 * responseCheckpoint.projected_value).toFixed(2)}/100</strong> in this
                  partial-draft comparison ({signedChange(100 * responseCheckpoint.projected_value - 50)}
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
            <p>
              Choose a role, then click a champion portrait. Flex picks move
              focus to their role choices before they are added.
            </p>
          </div>

          <div
            className="sandbox-role-filters"
            role="group"
            aria-label="Champion role filter"
            aria-describedby="sandbox-role-policy-note"
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
              Supported pro roles
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
              Manual any role
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
                candidateRole !== "open" && candidateRole !== "any" ? candidateRole : null;
              const roleGames = explicitRole ? Number(champion.role_games?.[explicitRole] ?? 0) : null;
              const observedOpenRoles = champion.roles.filter((role) => openRoles.includes(role));
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
                  onClick={(event) =>
                    chooseChampion(champion, event.currentTarget)
                  }
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
              Response-aware comparisons for {sideLabel(analysis?.recommendation_side ?? nextSide ?? perspective)}
            </h2>
          </div>
          <p>
            {analysis
              ? `The bounded search deep-evaluated ${analysis.search.root_evaluated_actions} of ${analysis.search.root_legal_actions} legal champion-role actions, then followed each through up to two later picks. `
              : "The bounded search retains the strongest immediate legal actions, then follows each through up to two later picks. "}
            Change is measured against the current state. The composition probability pipeline failed
            its chronological promotion gate, so these values are not win probabilities or a solved
            draft game.
          </p>
        </div>
        {loading && analysis ? <p className="status-hint">Updating model…</p> : null}
        {error && analysis ? <p className="error-banner">{error}</p> : null}
        {!publicPatch ? (
          <p className="status-hint">
            Select an observed patch context to load policy rankings.
          </p>
        ) : analysis && nextSide ? (
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
              <tr><th>Seat</th><th>Pick</th><th>Role</th><th>Model value</th><th>Change</th><th>Status</th></tr>
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
                    <td className="font-mono">
                      {(100 * row.projected_value).toFixed(2)}/100
                    </td>
                    <td className={`font-mono ${row.delta_points >= 0 ? "sandbox-positive" : "sandbox-negative"}`}>
                      {signedChange(row.delta_points)}
                    </td>
                    <td>
                      Experimental policy value
                    </td>
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
          Unfilled seats are neutralized so changes compare one draft branch with another at the same
          point. The same experimental composition value is used on a complete board: role-aware
          champion effects, within-team synergy, and all 25 explicit enemy interactions. Sparse
          low-rank residuals are disabled because their uncertainty is not estimated. This terminal
          model did not beat the chronological base-rate benchmark, so the
          values are deliberately not labelled as outcome probabilities. Candidate rankings use
          bounded beam minimax: the acting side maximizes
          its model share and the opposing side minimizes it over up to two later picks. This is not
          exhaustive search. The sandbox does not know a player&apos;s champion pool, scrim plan, or
          hidden flex intent. A role is locked when a champion is placed; this
          version does not preserve unresolved flex assignments. Manual
          what-if mode can place any catalog champion in any still-open
          role; when that champion-role pair has no pro sample, its role-specific direct effect stays
          neutral while champion-level synergy and enemy interactions still apply, and the UI marks
          the role as unseen. That manual placement does not expand policy search: recommendations
          and look-ahead picks remain restricted to champion-role pairs with at
          least {DRAFT_POLICY_MIN_ROLE_GAMES} pro maps.
        </p>
        <p className="font-mono">{analysis?.note}</p>
      </details>
    </div>
  );
}
