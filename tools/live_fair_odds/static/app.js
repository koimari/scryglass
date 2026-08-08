const STORAGE_KEY = "scryglass-personal-live-v1";
const ROLES = ["Top", "Jungle", "Mid", "Bottom", "Support"];
const OBJECTIVES = [
  ["dragons", "Dragons"],
  ["towers", "Towers"],
  ["grubs", "Grubs"],
  ["heralds", "Heralds"],
  ["barons", "Barons"],
  ["inhibitors", "Inhibitors"],
];

const EXAMPLE = {
  league: "LPL",
  blue_team: "Bilibili Gaming",
  red_team: "LGD Gaming",
  blue_picks: ["K'Sante", "Skarner", "Orianna", "Yunara", "Lulu"],
  red_picks: ["Ambessa", "Lee Sin", "Ryze", "Varus", "Bard"],
  minute: 15,
  patch: "",
  event_id: "",
  event_start: "",
  draft_source_available_at: "",
  blue_kills: 5,
  red_kills: 7,
  blue_gold: "",
  red_gold: "",
  blue_win_odds: "",
  red_win_odds: "",
  blue_dragons: 0,
  red_dragons: 0,
  blue_towers: 0,
  red_towers: 0,
  blue_grubs: 0,
  red_grubs: 0,
  blue_heralds: 0,
  red_heralds: 0,
  blue_barons: 0,
  red_barons: 0,
  blue_inhibitors: 0,
  red_inhibitors: 0,
  lines: [
    { line: 27.5, over_odds: "", under_odds: "" },
    { line: 28.5, over_odds: "", under_odds: "" },
    { line: 29.5, over_odds: "", under_odds: "" },
    { line: 30.5, over_odds: "", under_odds: "" },
    { line: 34.5, over_odds: "", under_odds: 1.83 },
  ],
};

let options = null;
let current = loadState();

function loadState() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY));
    return parsed && typeof parsed === "object" ? { ...EXAMPLE, ...parsed } : structuredClone(EXAMPLE);
  } catch {
    return structuredClone(EXAMPLE);
  }
}

function saveState(value) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
}

function element(id) {
  return document.getElementById(id);
}

function setSelect(select, rows, selected, valueKey = null, labelKey = null) {
  select.replaceChildren();
  rows.forEach((row) => {
    const option = document.createElement("option");
    option.value = valueKey ? row[valueKey] : row;
    option.textContent = labelKey ? row[labelKey] : row;
    option.selected = option.value === selected;
    select.append(option);
  });
}

function setTeamSelect(select, selected, league) {
  const matching = options.teams.filter((team) => team.league === league);
  const selectedTeam = options.teams.find((team) => team.name === selected);
  const rows = selectedTeam && !matching.some((team) => team.name === selected)
    ? [selectedTeam, ...matching]
    : matching;
  setSelect(select, rows.length ? rows : options.teams, selected, "name", "name");
}

function renderPickInputs(side, picks) {
  const container = element(`${side}-picks`);
  container.replaceChildren();
  ROLES.forEach((role, index) => {
    const fragment = element("pick-template").content.cloneNode(true);
    const label = fragment.querySelector("label");
    const roleLabel = fragment.querySelector("span");
    const input = fragment.querySelector("input");
    roleLabel.textContent = role;
    input.value = picks[index] || "";
    input.dataset.side = side;
    input.dataset.index = String(index);
    input.setAttribute("aria-label", `${side} ${role} champion`);
    label.dataset.role = role.toLowerCase();
    container.append(fragment);
  });
}

function renderObjectives(side) {
  const container = element(`${side}-objectives`);
  container.replaceChildren();
  OBJECTIVES.forEach(([key, label]) => {
    const fragment = element("objective-template").content.cloneNode(true);
    const text = fragment.querySelector("span");
    const input = fragment.querySelector("input");
    text.textContent = label;
    input.id = `${side}-${key}`;
    input.value = current[`${side}_${key}`] ?? 0;
    input.max = key === "towers" ? "11" : key === "dragons" ? "7" : "6";
    input.setAttribute("aria-label", `${side} ${label}`);
    container.append(fragment);
  });
}

function renderLines(lines) {
  const container = element("market-lines");
  container.replaceChildren();
  lines.forEach((line, index) => {
    const fragment = element("line-template").content.cloneNode(true);
    const row = fragment.querySelector(".market-input-row");
    row.dataset.index = String(index);
    fragment.querySelector(".market-line").value = line.line ?? "";
    fragment.querySelector(".market-over").value = line.over_odds ?? "";
    fragment.querySelector(".market-under").value = line.under_odds ?? "";
    fragment.querySelector(".remove-line").addEventListener("click", () => {
      const next = collectForm();
      next.lines.splice(index, 1);
      current = next;
      renderLines(current.lines);
      saveState(current);
    });
    container.append(fragment);
  });
}

function updateWinnerLabels() {
  const blueName = element("blue-team").value || "Blue";
  const redName = element("red-team").value || "Red";
  element("blue-winner-name").textContent = `${blueName} winner`;
  element("red-winner-name").textContent = `${redName} winner`;
}

function fillForm() {
  setSelect(element("league"), options.leagues, current.league);
  setTeamSelect(element("blue-team"), current.blue_team, current.league);
  setTeamSelect(element("red-team"), current.red_team, current.league);
  const championList = element("champions");
  championList.replaceChildren(
    ...options.champions.map((champion) => {
      const option = document.createElement("option");
      option.value = champion;
      return option;
    }),
  );
  renderPickInputs("blue", current.blue_picks);
  renderPickInputs("red", current.red_picks);
  [
    "minute",
    "patch",
    "event_id",
    "event_start",
    "draft_source_available_at",
    "blue_kills",
    "red_kills",
    "blue_gold",
    "red_gold",
    "blue_win_odds",
    "red_win_odds",
  ].forEach((key) => {
    element(key.replaceAll("_", "-")).value = current[key] ?? "";
  });
  renderObjectives("blue");
  renderObjectives("red");
  renderLines(current.lines);
  updateWinnerLabels();
  updateClock();
  element("model-date").textContent = `Local model snapshot · ${options.model_as_of || "date unavailable"}`;
}

function optionalNumber(value) {
  return value === "" ? "" : Number(value);
}

function collectForm() {
  const bluePicks = [...element("blue-picks").querySelectorAll("input")].map((input) => input.value.trim());
  const redPicks = [...element("red-picks").querySelectorAll("input")].map((input) => input.value.trim());
  const value = {
    league: element("league").value,
    blue_team: element("blue-team").value,
    red_team: element("red-team").value,
    blue_picks: bluePicks,
    red_picks: redPicks,
    minute: Number(element("minute").value),
    patch: element("patch").value.trim(),
    event_id: element("event-id").value.trim(),
    event_start: element("event-start").value.trim(),
    draft_source_available_at: element("draft-source-available-at").value.trim(),
    blue_kills: Number(element("blue-kills").value),
    red_kills: Number(element("red-kills").value),
    blue_gold: optionalNumber(element("blue-gold").value),
    red_gold: optionalNumber(element("red-gold").value),
    blue_win_odds: optionalNumber(element("blue-win-odds").value),
    red_win_odds: optionalNumber(element("red-win-odds").value),
    lines: [...element("market-lines").querySelectorAll(".market-input-row")].map((row) => ({
      line: Number(row.querySelector(".market-line").value),
      over_odds: optionalNumber(row.querySelector(".market-over").value),
      under_odds: optionalNumber(row.querySelector(".market-under").value),
    })),
  };
  ["blue", "red"].forEach((side) => {
    OBJECTIVES.forEach(([key]) => {
      value[`${side}_${key}`] = Number(element(`${side}-${key}`).value);
    });
  });
  return value;
}

function updateClock() {
  const value = Number(element("minute").value || 0);
  const minutes = Math.floor(value);
  const seconds = Math.round((value - minutes) * 60);
  element("clock-readout").textContent = `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function percent(value) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function odds(value) {
  return value == null ? "—" : Number(value).toFixed(2);
}

function number(value, digits = 1) {
  return value == null ? "—" : Number(value).toFixed(digits);
}

function decisionLabel(value) {
  return value === "NO_AUTHORIZED_BET" ? "NOT AUTHORIZED" : value;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderResults(result) {
  const totals = result.live_totals;
  const win = result.live_win;
  const winner = result.winner_reprice;
  const draftDiagnostic = result.pregame_win?.draft_score ?? null;
  const ratingRegistration = result.rating_registration ?? {};
  const ratingGap = ratingRegistration.ratings?.strength_difference?.posterior_mean ?? null;
  const favorite = winner.p_blue == null
    ? "No authorized winner price"
    : winner.p_blue >= winner.p_red
      ? `${result.teams.blue} ${percent(winner.p_blue)}`
      : `${result.teams.red} ${percent(winner.p_red)}`;
  const winnerCard = (side, team, view) => {
    const ev = view.expected_return_pct == null
      ? "—"
      : `${view.expected_return_pct >= 0 ? "+" : ""}${Number(view.expected_return_pct).toFixed(1)}%`;
    const evTone = view.expected_return_pct == null
      ? "neutral"
      : view.expected_return_pct >= 5
        ? "positive"
        : view.expected_return_pct >= 0
          ? "marginal"
          : "negative";
    return `
      <article class="winner-card is-${side}">
        <header>
          <span>${side} side</span>
          <h3>${escapeHtml(team)}</h3>
        </header>
        <div class="winner-probability">
          <strong>${percent(view.probability)}</strong>
          <span>Authorized probability</span>
          <small>Research diagnostic ${percent(view.diagnostic_probability)}</small>
        </div>
        <dl>
          <div><dt>Offered</dt><dd>${odds(view.offered_odds)}</dd></div>
          <div><dt>No-vig market</dt><dd>${percent(view.no_vig_break_even_probability)}</dd></div>
          <div><dt>Authorized fair</dt><dd>${odds(view.fair_odds)}</dd></div>
          <div class="winner-ev" data-tone="${evTone}"><dt>Authorized EV</dt><dd>${ev}</dd></div>
        </dl>
        <span class="winner-decision" data-decision="${view.decision}">${decisionLabel(view.decision)}</span>
      </article>
    `;
  };
  const marketRow = (side, line, view) => {
    const diagnostic = view.diagnostic_probability;
    const offered = view.offered_odds == null ? "No offered price" : `Offered ${odds(view.offered_odds)}`;
    const blockers = (view.blockers || []).join(", ") || "No blockers";
    const detail = view.expected_return_pct == null
      ? offered
      : `${view.expected_return_pct >= 0 ? "+" : ""}${Number(view.expected_return_pct).toFixed(1)}% EV`;
    return `
      <div class="price-row" data-side="${side}">
        <strong class="line-mark">${side === "under" ? "U" : "O"} ${Number(line).toFixed(1)}</strong>
        <div class="prob-track" aria-label="${side} research diagnostic ${percent(diagnostic)}"><span style="width:${diagnostic == null ? 0 : diagnostic * 100}%"></span></div>
        <span class="prob-label">${percent(diagnostic)}</span>
        <span class="fair-label">${percent(view.no_vig_break_even_probability)}</span>
        <span class="decision" data-decision="${view.decision}" title="${escapeHtml(blockers)}">
          <strong>${decisionLabel(view.decision)}</strong>
          <small>${detail}</small>
        </span>
      </div>
    `;
  };
  const lineRows = totals.lines.map((row) => [
    marketRow("under", row.line, row.under),
    marketRow("over", row.line, row.over),
  ].join("")).join("");
  const warnings = [
    ...(totals.warnings || []),
    ...(winner.warnings || []),
    ...(win.warnings || []).slice(0, 2),
    win.personal_soft_extension?.note,
  ].filter(Boolean);

  element("results").innerHTML = `
    <div class="result-head">
      <div>
        <p class="section-kicker">Manual snapshot · ${Number(result.minute).toFixed(1)} minutes</p>
        <h2>${escapeHtml(favorite)}</h2>
        <p>${escapeHtml(result.teams.blue)} ${result.state.blue_kills}–${result.state.red_kills} ${escapeHtml(result.teams.red)}</p>
      </div>
      <div class="projection-number">
        <strong>${number(totals.projected_mean)}</strong>
        <span>Research kill diagnostic</span>
      </div>
    </div>
    <section class="winner-reprice" aria-label="Winner market audit">
      <div class="winner-reprice-head">
        <div>
          <p class="section-kicker">Winner market audit</p>
          <h3>Market context versus research diagnostic</h3>
        </div>
        <span class="winner-mode">${escapeHtml(winner.source)}</span>
      </div>
      <div class="winner-card-grid">
        ${winnerCard("blue", result.teams.blue, winner.blue)}
        ${winnerCard("red", result.teams.red, winner.red)}
      </div>
    </section>
    <div class="price-ladder">
      <div class="price-ladder-heading"><span>Side</span><span>Diagnostic</span><span>Model</span><span>No-vig</span><span>Decision</span></div>
      ${lineRows}
    </div>
    <div class="result-ledger">
      <div><span>Checkpoint</span><strong>${totals.checkpoint == null ? "—" : `${totals.checkpoint}:00`}</strong></div>
      <div><span>Effective series</span><strong>${number(totals.effective_n, 0)}</strong></div>
      <div><span>Pregame kills</span><strong>${number(result.pregame_kills.mean)}</strong></div>
      <div><span>Draft-only dev · blue</span><strong>${percent(draftDiagnostic?.blue)}</strong></div>
      <div><span>Registered rating gap · blue</span><strong>${ratingGap == null ? "—" : `${ratingGap >= 0 ? "+" : ""}${number(ratingGap, 0)}`}</strong></div>
      <div><span>Rating authority</span><strong>${ratingRegistration.status === "registered" ? "Registered" : "Withheld"}</strong></div>
    </div>
    <ul class="warnings">${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>
  `;
  element("results").dataset.ready = "true";
}

async function score(event) {
  event.preventDefault();
  const button = document.querySelector(".price-button");
  const results = element("results");
  element("form-error").textContent = "";
  current = collectForm();
  saveState(current);
  button.disabled = true;
  results.setAttribute("aria-busy", "true");
  results.dataset.ready = "false";
  try {
    const response = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(current),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "The state could not be priced.");
    renderResults(result);
  } catch (error) {
    element("form-error").textContent = error.message;
  } finally {
    button.disabled = false;
    results.setAttribute("aria-busy", "false");
  }
}

async function initialise() {
  try {
    const response = await fetch("/api/options", { cache: "no-store" });
    if (!response.ok) throw new Error("Local model options are unavailable.");
    options = await response.json();
    fillForm();
  } catch (error) {
    element("form-error").textContent = error.message;
  }
}

element("worksheet").addEventListener("submit", score);
element("minute").addEventListener("input", updateClock);
element("league").addEventListener("change", () => {
  const league = element("league").value;
  setTeamSelect(element("blue-team"), element("blue-team").value, league);
  setTeamSelect(element("red-team"), element("red-team").value, league);
  updateWinnerLabels();
});
element("blue-team").addEventListener("change", updateWinnerLabels);
element("red-team").addEventListener("change", updateWinnerLabels);
element("add-line").addEventListener("click", () => {
  current = collectForm();
  const last = current.lines.at(-1)?.line ?? 29.5;
  current.lines.push({ line: Number(last) + 1, over_odds: "", under_odds: "" });
  renderLines(current.lines);
  saveState(current);
});
element("reset-example").addEventListener("click", () => {
  current = structuredClone(EXAMPLE);
  saveState(current);
  fillForm();
  element("results").removeAttribute("data-ready");
});

initialise();
