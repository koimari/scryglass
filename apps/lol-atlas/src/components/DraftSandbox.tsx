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

const DRAFT_ROLES: DraftRole[] = ["top", "jng", "mid", "bot", "sup"];

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

const LEAGUES = ["LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP", "INTL"];

type Props = {
  catalog: DraftChampion[];
  teams: DraftTeamOption[];
  initialActions?: DraftAction[];
  initialExcluded?: string[];
  initialPerspective?: DraftSide;
  initialLeague?: string;
  initialBlueTeam?: string;
  initialRedTeam?: string;
};

type DraftRosterOption = {
  player: string;
  role: DraftRole | null;
  rating: number;
  n_maps: number;
};

export type DraftTeamOption = {
  team: string;
  league: string | null;
  tier: "tier1" | "tier2" | "tier3" | null;
  rating: number;
  roster: DraftRosterOption[];
};

type LineupSelection = Partial<Record<DraftRole, string>>;

function sideLabel(side: DraftSide): string {
  return side === "blue" ? "Blue" : "Red";
}

function signedPp(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} pp`;
}

function signedModel(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function componentEntries(row: DraftRecommendation) {
  return [
    ["Base", row.components.champion],
    ["Synergy", row.components.synergy],
    ["Vs enemy", row.components.counters],
    ["Lane", row.components.lane],
    ["Comfort", row.components.comfort],
  ] as const;
}

function strongestDriver(row: DraftRecommendation): string {
  const entries = componentEntries(row);
  const [label, value] = [...entries].sort(
    (first, second) => Math.abs(second[1]) - Math.abs(first[1]),
  )[0];
  return `${label} ${signedModel(value)}`;
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
            <th>Model decomposition <small>log-odds</small></th>
            <th>Player fit</th>
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
              <td>
                <div className="sandbox-driver-ledger">
                  {componentEntries(row).map(([label, value]) => (
                    <span
                      className={value > 0 ? "is-positive" : value < 0 ? "is-negative" : ""}
                      key={label}
                    >
                      <em>{label}</em>
                      <strong>{signedModel(value)}</strong>
                    </span>
                  ))}
                </div>
                <small>{row.evidence} champion prior</small>
              </td>
              <td>
                {row.player ? (
                  <>
                    <strong>{row.player}</strong>
                    <small>
                      {row.player_games > 0
                        ? `${row.player_games} games on champion`
                        : "Neutral comfort prior"}
                    </small>
                  </>
                ) : "No lineup"}
              </td>
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

function teamTierLabel(tier: DraftTeamOption["tier"]): string {
  if (tier === "tier1") return "Tier 1";
  if (tier === "tier2") return "Tier 2";
  if (tier === "tier3") return "Tier 3";
  return "Unclassified";
}

function defaultLineup(team: DraftTeamOption | null): LineupSelection {
  if (!team) return {};
  const output: LineupSelection = {};
  for (const role of DRAFT_ROLES) {
    const player = team.roster.find((candidate) => candidate.role === role);
    if (player) output[role] = player.player;
  }
  return output;
}

function TeamContextPanel({
  side,
  teams,
  selected,
  lineup,
  onTeam,
  onPlayer,
}: {
  side: DraftSide;
  teams: DraftTeamOption[];
  selected: string;
  lineup: LineupSelection;
  onTeam: (team: string) => void;
  onPlayer: (role: DraftRole, player: string) => void;
}) {
  const team = teams.find((candidate) => candidate.team === selected) ?? null;
  const grouped = {
    tier1: teams.filter((candidate) => candidate.tier === "tier1"),
    tier2: teams.filter((candidate) => candidate.tier === "tier2"),
    tier3: teams.filter((candidate) => candidate.tier === "tier3"),
  };
  return (
    <section className={`sandbox-team-context sandbox-team-context-${side}`}>
      <header>
        <div>
          <span>{sideLabel(side)} context</span>
          <strong>{team?.team ?? "Even-strength baseline"}</strong>
        </div>
        {team ? (
          <span className="sandbox-team-rating">
            {team.rating.toFixed(0)} <small>adjusted Elo</small>
          </span>
        ) : null}
      </header>
      <label>
        <span className="sr-only">{sideLabel(side)} team</span>
        <select value={selected} onChange={(event) => onTeam(event.target.value)}>
          <option value="">No team selected</option>
          {(["tier1", "tier2", "tier3"] as const).map((tier) => (
            <optgroup label={teamTierLabel(tier)} key={tier}>
              {grouped[tier].map((candidate) => (
                <option value={candidate.team} key={candidate.team}>
                  {candidate.team} · {candidate.league ?? "—"}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      {team ? (
        <div className="sandbox-lineup">
          {DRAFT_ROLES.map((role) => {
            const candidates = team.roster.filter((player) => player.role === role);
            const current = candidates.find((player) => player.player === lineup[role]);
            return (
              <label key={role}>
                <span>{ROLE_LABEL[role]}</span>
                <select
                  value={lineup[role] ?? ""}
                  onChange={(event) => onPlayer(role, event.target.value)}
                >
                  <option value="">Unassigned</option>
                  {candidates.map((player) => (
                    <option value={player.player} key={player.player}>
                      {player.player}
                    </option>
                  ))}
                </select>
                <small>
                  {current
                    ? `${current.rating.toFixed(0)} Elo · ${current.n_maps} maps`
                    : "Neutral player prior"}
                </small>
              </label>
            );
          })}
        </div>
      ) : (
        <p className="sandbox-team-empty">
          Select both teams to add the calibrated strength prior. Lineups supply role-specific
          player Elo and champion comfort.
        </p>
      )}
    </section>
  );
}

function BanWorkbench({
  catalog,
  selected,
  excluded,
  open,
  search,
  role,
  onOpen,
  onSearch,
  onRole,
  onToggle,
}: {
  catalog: DraftChampion[];
  selected: Set<string>;
  excluded: string[];
  open: boolean;
  search: string;
  role: DraftCandidateRole;
  onOpen: () => void;
  onSearch: (value: string) => void;
  onRole: (value: DraftCandidateRole) => void;
  onToggle: (champion: string) => void;
}) {
  const excludedNames = new Set(excluded.map((champion) => champion.toLocaleLowerCase()));
  const filtered = catalog
    .filter((champion) => !selected.has(champion.name.toLocaleLowerCase()))
    .filter((champion) =>
      role === "any" || role === "open" ? true : champion.roles.includes(role),
    )
    .filter((champion) =>
      champion.name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()),
    )
    .slice(0, 48);
  return (
    <section className="sandbox-bans" aria-labelledby="sandbox-bans-heading">
      <header>
        <div>
          <p className="sandbox-step">Draft constraint</p>
          <h2 id="sandbox-bans-heading">Banned and unavailable</h2>
        </div>
        <button type="button" className="sandbox-secondary-button" onClick={onOpen}>
          {open ? "Close champion pool" : "Manage champion pool"}
        </button>
      </header>
      <div className="sandbox-ban-tray">
        {Array.from({ length: Math.max(10, excluded.length) }, (_, index) => {
          const champion = excluded[index];
          return champion ? (
            <button
              type="button"
              className="sandbox-ban-token"
              key={champion}
              onClick={() => onToggle(champion)}
              aria-label={`Restore ${champion}`}
            >
              <span
                className="sandbox-champion-portrait"
                aria-hidden
                style={{ backgroundImage: `url("${champIconUrl(champion)}")` }}
              />
              <span>{champion}</span>
              <em aria-hidden>×</em>
            </button>
          ) : (
            <span className="sandbox-ban-empty" aria-hidden key={`empty-${index}`}>
              <span>Ban {index + 1}</span>
            </span>
          );
        })}
      </div>
      {open ? (
        <div className="sandbox-ban-pool">
          <div className="sandbox-ban-tools">
            <label>
              <span>Find champion</span>
              <input
                type="search"
                value={search}
                onChange={(event) => onSearch(event.target.value)}
                placeholder="Search all champions"
              />
            </label>
            <label>
              <span>Role evidence</span>
              <select
                value={role}
                onChange={(event) => onRole(event.target.value as DraftCandidateRole)}
              >
                <option value="any">Every role</option>
                {Object.entries(ROLE_LABEL).map(([value, label]) => (
                  <option value={value} key={value}>{label}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="sandbox-champion-pool">
            {filtered.map((champion) => {
              const isBanned = excludedNames.has(champion.name.toLocaleLowerCase());
              const evidenceGames =
                role !== "any" && role !== "open"
                  ? Number(champion.role_games[role] ?? 0)
                  : champion.games;
              return (
                <button
                  type="button"
                  className={isBanned ? "is-banned" : ""}
                  onClick={() => onToggle(champion.name)}
                  key={champion.name}
                >
                  <span
                    className="sandbox-champion-portrait"
                    aria-hidden
                    style={{ backgroundImage: `url("${champIconUrl(champion.name)}")` }}
                  />
                  <span>{champion.name}</span>
                  <small>
                    {evidenceGames > 0 ? `${evidenceGames} pro games` : "Neutral prior"}
                  </small>
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
}

export function DraftSandbox({
  catalog,
  teams,
  initialActions = [],
  initialExcluded = [],
  initialPerspective = "red",
  initialLeague = "LCS",
  initialBlueTeam = "",
  initialRedTeam = "",
}: Props) {
  const [actions, setActions] = useState<DraftAction[]>(initialActions);
  const [excluded, setExcluded] = useState<string[]>(initialExcluded);
  const [perspective, setPerspective] = useState<DraftSide>(initialPerspective);
  const [league, setLeague] = useState(initialLeague);
  const [candidateRole, setCandidateRole] = useState<DraftCandidateRole>("open");
  const [search, setSearch] = useState("");
  const [banSearch, setBanSearch] = useState("");
  const [banRole, setBanRole] = useState<DraftCandidateRole>("any");
  const [banPoolOpen, setBanPoolOpen] = useState(false);
  const [blueTeam, setBlueTeam] = useState(initialBlueTeam);
  const [redTeam, setRedTeam] = useState(initialRedTeam);
  const [bluePlayers, setBluePlayers] = useState<LineupSelection>(() =>
    defaultLineup(teams.find((team) => team.team === initialBlueTeam) ?? null),
  );
  const [redPlayers, setRedPlayers] = useState<LineupSelection>(() =>
    defaultLineup(teams.find((team) => team.team === initialRedTeam) ?? null),
  );
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
            blue_team: blueTeam || null,
            red_team: redTeam || null,
            blue_players: bluePlayers,
            red_players: redPlayers,
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
  }, [
    actions,
    bluePlayers,
    blueTeam,
    candidateRole,
    excluded,
    league,
    nextSide,
    perspective,
    redPlayers,
    redTeam,
  ]);

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

  const toggleUnavailable = (champion: string) => {
    if (selected.has(champion.toLocaleLowerCase())) return;
    setExcluded((current) =>
      current.includes(champion)
        ? current.filter((item) => item !== champion)
        : [...current, champion],
    );
  };

  const changeTeam = (side: DraftSide, teamName: string) => {
    const team = teams.find((candidate) => candidate.team === teamName) ?? null;
    if (side === "blue") {
      setBlueTeam(teamName);
      setBluePlayers(defaultLineup(team));
    } else {
      setRedTeam(teamName);
      setRedPlayers(defaultLineup(team));
    }
  };

  const copyScenario = async () => {
    const url = new URL(window.location.href);
    url.searchParams.set("draft", encodeScenario(actions, excluded));
    url.searchParams.set("side", perspective);
    url.searchParams.set("league", league);
    if (blueTeam) url.searchParams.set("blueTeam", blueTeam);
    else url.searchParams.delete("blueTeam");
    if (redTeam) url.searchParams.set("redTeam", redTeam);
    else url.searchParams.delete("redTeam");
    await navigator.clipboard.writeText(url.toString());
    setShareState("Copied");
    window.setTimeout(() => setShareState("Copy scenario"), 1600);
  };

  return (
    <div className="sandbox-page">
      <header className="page-header">
        <p className="blog-kicker">Professional draft analysis</p>
        <h1 className="font-display mt-2 text-3xl">Draft sandbox</h1>
        <p className="lede">
          Compare legal responses against the exact draft state. Recommendations separate champion
          strength, allied synergy, enemy counters, player comfort, and team strength.
        </p>
        <div className="sandbox-model-stamp">
          <span><strong>{catalog.length}</strong> current champions</span>
          <span><strong>{analysis?.model.maps?.toLocaleString() ?? "16,334"}</strong> professional maps</span>
          <span><strong>Rolling-time</strong> interaction validation</span>
        </div>
        <div className="sandbox-header-actions">
          <button type="button" className="sandbox-secondary-button" onClick={() => {
            setActions([]);
            setExcluded([]);
            setBlueTeam("");
            setRedTeam("");
            setBluePlayers({});
            setRedPlayers({});
          }}>
            New draft
          </button>
          <button type="button" className="sandbox-secondary-button" onClick={copyScenario}>
            {shareState}
          </button>
        </div>
      </header>

      <section className="sandbox-strength-section" aria-labelledby="strength-context-heading">
        <div className="sandbox-section-intro">
          <p className="sandbox-step">01 · Strength context</p>
          <div>
            <h2 id="strength-context-heading">Who is playing?</h2>
            <p>
              Team Elo sets the match prior. The selected lineup adds role-weighted player Elo and
              champion-specific comfort.
            </p>
          </div>
        </div>
        <div className="sandbox-team-grid">
          <TeamContextPanel
            side="blue"
            teams={teams}
            selected={blueTeam}
            lineup={bluePlayers}
            onTeam={(team) => changeTeam("blue", team)}
            onPlayer={(role, player) =>
              setBluePlayers((currentPlayers) => ({ ...currentPlayers, [role]: player }))
            }
          />
          <TeamContextPanel
            side="red"
            teams={teams}
            selected={redTeam}
            lineup={redPlayers}
            onTeam={(team) => changeTeam("red", team)}
            onPlayer={(role, player) =>
              setRedPlayers((currentPlayers) => ({ ...currentPlayers, [role]: player }))
            }
          />
        </div>
      </section>

      <section className="sandbox-draft-section" aria-labelledby="draft-state-heading">
        <div className="sandbox-section-intro">
          <p className="sandbox-step">02 · Draft state</p>
          <div>
            <h2 id="draft-state-heading">Build the current board</h2>
            <p>The next legal seat and the evaluation perspective stay visible while you branch.</p>
          </div>
        </div>
        <div className="sandbox-context" aria-label="Analysis context">
          <div className="sandbox-next">
            <span>On the clock</span>
            <strong>{nextSide ? `${sideLabel(nextSide)} pick` : "Draft complete"}</strong>
          </div>
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
            <span>Recommendation role</span>
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
          <label>
            <span>Calibration field</span>
            <select value={league} onChange={(event) => setLeague(event.target.value)}>
              {LEAGUES.map((item) => <option value={item} key={item}>{item}</option>)}
            </select>
          </label>
        </div>
      </section>

      <BanWorkbench
        catalog={catalog}
        selected={selected}
        excluded={excluded}
        open={banPoolOpen}
        search={banSearch}
        role={banRole}
        onOpen={() => setBanPoolOpen((currentOpen) => !currentOpen)}
        onSearch={setBanSearch}
        onRole={setBanRole}
        onToggle={toggleUnavailable}
      />

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
          <p className="sandbox-step">Model position</p>
          {loading && !current ? (
            <div className="sandbox-skeleton" aria-label="Loading analysis" />
          ) : error ? (
            <p className="error-banner">{error}</p>
          ) : current ? (
            <>
              <div className="sandbox-current-number">
                <span>{sideLabel(perspective)} projected win chance</span>
                <strong>{(100 * current.projected_wr).toFixed(2)}%</strong>
                <small>
                  {current.score.calibration.strength_source === "even-strength assumption"
                    ? "Even-strength prior"
                    : current.score.calibration.strength_source}
                </small>
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
                  {analysis.recommendations.slice(0, 3).map((row, index) => (
                    <button
                      type="button"
                      className={index === 0 ? "is-primary" : ""}
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
                        <small>
                          {row.role ? ROLE_LABEL[row.role] : "Role open"} · {strongestDriver(row)}
                        </small>
                      </span>
                      <em>{signedPp(row.delta_pp)}</em>
                    </button>
                  ))}
                </div>
              ) : null}
              <dl className="sandbox-read-ledger">
                <div><dt>Model confidence</dt><dd>{(100 * current.confidence).toFixed(0)}%</dd></div>
                <div><dt>Draft calibration</dt><dd>{current.score.calibration.source}</dd></div>
                <div>
                  <dt>Interaction gates</dt>
                  <dd>
                    {analysis
                      ? `R ${analysis.model.interaction_gates.role} · S ${analysis.model.interaction_gates.synergy} · C ${analysis.model.interaction_gates.counters}/${analysis.model.interaction_gates.composition} · L ${analysis.model.interaction_gates.lane}`
                      : "—"}
                  </dd>
                </div>
                <div><dt>Chosen picks</dt><dd>{actions.length}/10</dd></div>
              </dl>
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
            <p className="sandbox-step">03 · Counterfactual ranking</p>
            <h2 id="ranking-heading">
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
            <p className="sandbox-step">04 · Decision trace</p>
            <h2 id="audit-heading">Where the model moved</h2>
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
          another at the same point. The recommendation model jointly estimates champion strength,
          allied pair synergy, cross-team counters, and same-role counters while controlling for
          team and player identity during training. Current team and lineup Elo set the strength
          prior; recency-weighted player champion evidence adds a separately shrunk comfort term.
          Partial states are not independently calibrated outcome probabilities.
        </p>
        {analysis ? (
          <p>
            The serving gates are champion-by-role {analysis.model.interaction_gates.role},
            synergy {analysis.model.interaction_gates.synergy},
            exact counter {analysis.model.interaction_gates.counters}, composition response{" "}
            {analysis.model.interaction_gates.composition}, and same-role{" "}
            {analysis.model.interaction_gates.lane}. A zero means that family failed the rolling
            validation test and is deliberately withheld from EV. Final chronological Brier score:
            {" "}
            {analysis.model.interaction_brier?.toFixed(4) ?? "not available"} with interactions,
            {" "}
            {analysis.model.baseline_brier?.toFixed(4) ?? "not available"} without them.
          </p>
        ) : null}
        <p className="font-mono">{analysis?.note}</p>
      </details>
    </div>
  );
}
