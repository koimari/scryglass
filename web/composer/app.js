const ROLES = ["Top", "Jng", "Mid", "Bot", "Sup"];

const CHAMPS = [
  "Aatrox","Ahri","Akali","Akshan","Alistar","Ambessa","Amumu","Anivia","Annie","Aphelios","Ashe","Aurelion Sol","Aurora",
  "Azir","Bard","Bel'Veth","Blitzcrank","Brand","Braum","Briar","Caitlyn","Camille","Cassiopeia","Cho'Gath","Corki",
  "Darius","Diana","Draven","Dr. Mundo","Ekko","Elise","Evelynn","Ezreal","Fiddlesticks","Fiora","Fizz","Galio","Gangplank",
  "Garen","Gnar","Gragas","Graves","Gwen","Hecarim","Heimerdinger","Hwei","Illaoi","Irelia","Ivern","Janna","Jarvan IV",
  "Jax","Jayce","Jhin","Jinx","K'Sante","Kai'Sa","Kalista","Karma","Karthus","Kassadin","Katarina","Kayle","Kayn","Kennen",
  "Kha'Zix","Kindred","Kled","Kog'Maw","LeBlanc","Lee Sin","Leona","Lillia","Lissandra","Locke","Lucian","Lulu","Lux",
  "Malphite","Malzahar","Maokai","Master Yi","Mel","Milio","Miss Fortune","Mordekaiser","Morgana","Naafiri","Nami","Nasus",
  "Nautilus","Neeko","Nidalee","Nilah","Nocturne","Nunu & Willump","Olaf","Orianna","Ornn","Pantheon","Poppy","Pyke",
  "Qiyana","Quinn","Rakan","Rammus","Rek'Sai","Rell","Renata Glasc","Renekton","Rengar","Riven","Rumble","Ryze","Samira",
  "Sejuani","Senna","Seraphine","Sett","Shaco","Shen","Shyvana","Singed","Sion","Sivir","Skarner","Smolder","Sona","Soraka",
  "Swain","Sylas","Syndra","Tahm Kench","Taliyah","Talon","Taric","Teemo","Thresh","Tristana","Trundle","Tryndamere",
  "Twisted Fate","Twitch","Udyr","Urgot","Varus","Vayne","Veigar","Vel'Koz","Vex","Vi","Viego","Viktor","Vladimir",
  "Volibear","Warwick","Wukong","Xayah","Xerath","Xin Zhao","Yasuo","Yone","Yorick","Yunara","Yuumi","Zac","Zed","Zeri",
  "Ziggs","Zilean","Zoe","Zyra",
];

const MARKET_PRESETS = [
  "Winner", "Total kills O/U", "Team kills handicap", "First Blood",
  "First Inhibitor", "Race to 10", "Race to 15", "Other",
];

const LEAGUE_WORDS = new Set([
  "LCK","LPL","LEC","LCS","CBLOL","MSI","WORLDS","EWC","FST","PCS","VCS","LJL","LCP","TCL","LTA","LTA N","LTA S",
]);

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

/** @type {{teams: string[], aliases: Record<string,string>, leagues: string[]}} */
let TEAM_DATA = {
  teams: [],
  aliases: {
    KC: "Karmine Corp",
    MKOI: "Movistar KOI",
    KOI: "Movistar KOI",
    DK: "Dplus Kia",
    HLE: "Hanwha Life Esports",
    G2: "G2 Esports",
    FNC: "Fnatic",
    GEN: "Gen.G",
    GENG: "Gen.G",
    VIT: "Team Vitality",
    TL: "Team Liquid",
    C9: "Cloud9",
    FLY: "FlyQuest",
  },
  leagues: ["LEC", "LCK", "LPL", "LCS", "EWC", "MSI", "Worlds", "FST", "CBLOL"],
};

let CHAMP_INDEX = new Map();
let MODE = "nl"; // nl | form
let SUPPRESS_FORM = false;
let SUPPRESS_LIVE_PAD = false;
let LIVE_PAD_DIRTY = false; // pad edits win over dump live until dump rewrite

function buildChampIndex() {
  CHAMP_INDEX = new Map();
  for (const c of CHAMPS) {
    CHAMP_INDEX.set(c.toLowerCase(), c);
    CHAMP_INDEX.set(c.replace(/['.]/g, "").toLowerCase(), c);
  }
  // common aliases
  const extra = {
    "ksante": "K'Sante",
    "kaisa": "Kai'Sa",
    "jarvan": "Jarvan IV",
    "j4": "Jarvan IV",
    "lee": "Lee Sin",
    "leesin": "Lee Sin",
    "xin": "Xin Zhao",
    "xinzhao": "Xin Zhao",
    "mf": "Miss Fortune",
    "missfortune": "Miss Fortune",
    "renata": "Renata Glasc",
    "nunu": "Nunu & Willump",
    "monkeyking": "Wukong",
    "mundo": "Dr. Mundo",
    "drmundo": "Dr. Mundo",
    "asol": "Aurelion Sol",
    "tf": "Twisted Fate",
    "locke": "Locke",
  };
  for (const [k, v] of Object.entries(extra)) CHAMP_INDEX.set(k, v);
}

function resolveTeam(name) {
  const raw = (name || "").trim();
  if (!raw) return raw;
  const hit = TEAM_DATA.aliases[raw] || TEAM_DATA.aliases[raw.toUpperCase()];
  if (hit) return hit;
  const lower = raw.toLowerCase();
  const found = TEAM_DATA.teams.find((t) => t.toLowerCase() === lower);
  return found || raw;
}

function resolveChamp(token) {
  const t = (token || "").trim();
  if (!t) return null;
  const key = t.toLowerCase().replace(/['.]/g, "");
  return CHAMP_INDEX.get(t.toLowerCase()) || CHAMP_INDEX.get(key) || null;
}

function extractChampsFromText(text) {
  /** Left-to-right scan so role order (Top→Sup) is preserved. */
  const sorted = [...CHAMPS].sort((a, b) => b.length - a.length);
  const found = [];
  const used = new Set();
  const s = String(text || "");
  let i = 0;
  while (i < s.length) {
    while (i < s.length && /[\s,|/·•:;]+/.test(s[i])) i++;
    if (i >= s.length) break;
    let matched = null;
    let matchLen = 0;
    for (const c of sorted) {
      if (i + c.length > s.length) continue;
      const slice = s.slice(i, i + c.length);
      if (slice.toLowerCase() !== c.toLowerCase()) continue;
      const after = s[i + c.length];
      if (after && /[A-Za-z0-9']/.test(after)) continue;
      matched = c;
      matchLen = c.length;
      break;
    }
    if (matched) {
      if (!used.has(matched.toLowerCase())) {
        found.push(matched);
        used.add(matched.toLowerCase());
      }
      i += matchLen;
      continue;
    }
    const m = s.slice(i).match(/^[A-Za-z0-9.'’&\-]+/);
    if (!m) {
      i += 1;
      continue;
    }
    const c = resolveChamp(m[0]);
    if (c && !used.has(c.toLowerCase())) {
      found.push(c);
      used.add(c.toLowerCase());
    }
    i += m[0].length;
  }
  return found;
}

function findTeamsInText(text) {
  const hits = [];
  const lower = text.toLowerCase();
  const candidates = [
    ...Object.keys(TEAM_DATA.aliases).map((a) => ({ raw: a, canon: TEAM_DATA.aliases[a] })),
    ...TEAM_DATA.teams.map((t) => ({ raw: t, canon: t })),
  ].sort((a, b) => b.raw.length - a.raw.length);

  const seen = new Set();
  for (const { raw, canon } of candidates) {
    const re = new RegExp(`(?:^|[^\\w.])${raw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?=$|[^\\w.])`, "i");
    if (re.test(text)) {
      const key = canon.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        hits.push({ raw, canon, index: lower.indexOf(raw.toLowerCase()) });
      }
    }
  }
  hits.sort((a, b) => a.index - b.index);
  return hits;
}

/**
 * Parse free-form dump → structured slip state.
 */
function parseDump(raw) {
  const text = (raw || "").replace(/\r/g, "").trim();
  const out = {
    team1: "",
    team2: "",
    league: "LEC",
    map: "1",
    blueTeam: "team1",
    series: "",
    blue: [],
    red: [],
    includeLive: false,
    includeBets: true,
    live: {},
    bets: [],
    warnings: [],
  };
  if (!text) return out;

  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const joined = lines.join("\n");

  // League
  for (const lg of TEAM_DATA.leagues) {
    if (new RegExp(`\\b${lg}\\b`, "i").test(text)) {
      out.league = lg;
      break;
    }
  }
  // Map
  const mapM = text.match(/\bmap\s*#?\s*([1-5])\b/i) || text.match(/\bg([1-5])\b/i);
  if (mapM) out.map = mapM[1];

  // Teams via "A vs B"
  const vsM = text.match(/([^\n|]+?)\s+vs\.?\s+([^\n|,]+?)(?:\s*[|·•,]|\s+map\b|\s*$)/i);
  if (vsM) {
    out.team1 = resolveTeam(vsM[1].replace(/^(match|game)\s+/i, "").trim());
    out.team2 = resolveTeam(vsM[2].split(/[|·•]/)[0].trim());
  } else {
    const teams = findTeamsInText(text);
    if (teams.length >= 2) {
      out.team1 = teams[0].canon;
      out.team2 = teams[1].canon;
    } else if (teams.length === 1) {
      out.team1 = teams[0].canon;
      out.warnings.push("Only one team found — add the opponent.");
    }
  }

  // BLUE / RED labeled drafts
  const blueLine = joined.match(/(?:^|\n)\s*(?:BLUE|Blue)\s*(?:\([^)]*\)|[^:\n]*)?[:\s]+([^\n]+)/);
  const redLine = joined.match(/(?:^|\n)\s*(?:RED|Red)\s*(?:\([^)]*\)|[^:\n]*)?[:\s]+([^\n]+)/);
  if (blueLine) out.blue = extractChampsFromText(blueLine[1]).slice(0, 5);
  if (redLine) out.red = extractChampsFromText(redLine[1]).slice(0, 5);

  // "draft A / B" or two blocks separated by vs
  if (out.blue.length < 5 || out.red.length < 5) {
    const sideSplit = text.split(/\n\s*vs\.?\s*\n/i);
    if (sideSplit.length === 2) {
      const b = extractChampsFromText(sideSplit[0]).slice(0, 5);
      const r = extractChampsFromText(sideSplit[1]).slice(0, 5);
      if (b.length >= 3) out.blue = b;
      if (r.length >= 3) out.red = r;
    }
  }

  // Fallback: first 5 + next 5 champs in document order (skip if already have)
  if (out.blue.length < 3 || out.red.length < 3) {
    const all = extractChampsFromText(text);
    if (all.length >= 8) {
      if (out.blue.length < 3) out.blue = all.slice(0, 5);
      if (out.red.length < 3) out.red = all.slice(5, 10);
    } else if (all.length >= 5 && out.blue.length < 3) {
      out.blue = all.slice(0, 5);
      out.warnings.push("Only one draft side clear — check Red.");
    }
  }

  // Blue side assignment from "BLUE TeamName"
  const blueTeamM = text.match(/\bBLUE\s+([A-Za-z0-9.'’ \-]+?)(?:\s*:|\s*$)/im);
  if (blueTeamM && out.team1 && out.team2) {
    const bt = resolveTeam(blueTeamM[1].split(":")[0].trim());
    if (bt.toLowerCase() === out.team2.toLowerCase()) out.blueTeam = "team2";
    else if (bt.toLowerCase() === out.team1.toLowerCase()) out.blueTeam = "team1";
  }

  // Odds: Team @ 1.50 (same-line only — \s would glue "Rell\\nG2")
  const oddsRe = /([A-Za-z0-9.'’]+(?:[ \t]+[A-Za-z0-9.'’]+)?)[ \t]*@[ \t]*(\d+(?:\.\d+)?)/gi;
  let om;
  const oddsHits = [];
  const knownTeam = (raw) => {
    const t = (raw || "").trim();
    if (!t) return null;
    if (TEAM_DATA.aliases[t] || TEAM_DATA.aliases[t.toUpperCase()]) return resolveTeam(t);
    if (TEAM_DATA.teams.some((x) => x.toLowerCase() === t.toLowerCase())) return resolveTeam(t);
    return null;
  };
  while ((om = oddsRe.exec(text))) {
    const sel = knownTeam(om[1]);
    if (!sel) continue;
    oddsHits.push({ selection: sel, odds: om[2], market: "Winner" });
  }
  // kills lines: over/under 29.5 @ 1.87
  const ouRe = /\b(over|under)\s+(\d+(?:\.\d)?)\s*@\s*(\d+(?:\.\d+)?)/gi;
  let ou;
  while ((ou = ouRe.exec(text))) {
    oddsHits.push({
      market: "Total kills O/U",
      selection: `${ou[1][0].toUpperCase()}${ou[1].slice(1).toLowerCase()} ${ou[2]}`,
      odds: ou[3],
    });
  }
  out.bets = oddsHits;

  // Live — labeled fields + shorthand HUD line
  // e.g. LIVE 18:00 4-8 31.8-36.3 T0-1 D0/3 G1/2
  const liveClock =
    text.match(/\bLIVE\s+(\d{1,2}:\d{2})\b/i) || text.match(/\b(\d{1,2}:\d{2})\b/);
  const killsM =
    text.match(/\bkills?\s+(\d+)\s*[-–]\s*(\d+)/i) ||
    text.match(/\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b(?=[^\n]*\bgold\b|\s+\d+(?:\.\d)?k)/i);
  const goldM =
    text.match(/\bgold\s+(\d+(?:\.\d)?)\s*k?\s*[-–]\s*(\d+(?:\.\d)?)\s*k?/i) ||
    text.match(/\b(\d{2,3}(?:\.\d)?)\s*k\s*[-–]\s*(\d{2,3}(?:\.\d)?)\s*k\b/i);
  const towersM =
    text.match(/\btowers?\s+(\d+)\s*[-–\/|]\s*(\d+)/i) ||
    text.match(/\bT\s*(\d+)\s*[-–\/|]\s*(\d+)\b/i);
  const dragonsM =
    text.match(/\bdragons?\s+(\d+)\s*[-–\/|]\s*(\d+)/i) ||
    text.match(/\bdrakes?\s+(\d+)\s*[-–\/|]\s*(\d+)/i) ||
    text.match(/\bD\s*(\d+)\s*[-–\/|]\s*(\d+)\b/i);
  const grubsM =
    text.match(/\bgrubs?\s+(\d+)\s*[-–\/|]\s*(\d+)/i) ||
    text.match(/\bG\s*(\d+)\s*[-–\/|]\s*(\d+)\b/i);
  // Bare scoreboard after clock: "18:00 4-8 31.8-36.3"
  const bareLive = text.match(
    /\b(\d{1,2}:\d{2})\s+(\d+)\s*[-–]\s*(\d+)\s+(\d+(?:\.\d)?)\s*k?\s*[-–]\s*(\d+(?:\.\d)?)\s*k?\b/i
  );
  if (/\bLIVE\b/i.test(text) || killsM || goldM || bareLive) {
    out.includeLive = true;
    if (bareLive) {
      out.live.clock = bareLive[1];
      out.live.killsBlue = bareLive[2];
      out.live.killsRed = bareLive[3];
      out.live.goldBlue = bareLive[4];
      out.live.goldRed = bareLive[5];
    }
    if (liveClock) out.live.clock = liveClock[1];
    if (killsM) {
      out.live.killsBlue = killsM[1];
      out.live.killsRed = killsM[2];
    }
    if (goldM) {
      out.live.goldBlue = goldM[1];
      out.live.goldRed = goldM[2];
    }
    if (towersM) {
      out.live.towersBlue = towersM[1];
      out.live.towersRed = towersM[2];
    }
    if (dragonsM) {
      out.live.dragonsBlue = dragonsM[1];
      out.live.dragonsRed = dragonsM[2];
    }
    if (grubsM) {
      out.live.grubsBlue = grubsM[1];
      out.live.grubsRed = grubsM[2];
    }
  }

  // Pregame / draft edge for glance (optional)
  // pre 37% | p_pre 0.37 | pregame MKOI 37% | fair MKOI 2.70
  const prePct = text.match(/\b(?:pre(?:game)?|p[_ ]?pre)\s+(?:([A-Za-z0-9.'’ ]+?)\s+)?(\d+(?:\.\d+)?)\s*%/i);
  const preFrac = text.match(/\bp[_ ]?pre\s+(\d?\.\d+)/i);
  if (prePct) {
    out.live.pPre = String(+prePct[2] / 100);
    if (prePct[1]) out.live.pPreSide = resolveTeam(prePct[1].trim());
  } else if (preFrac) {
    out.live.pPre = preFrac[1];
  }
  const draftEdgeM = text.match(/\bdraft[_ ]?edge\s+([+-]?\d+(?:\.\d+)?)/i);
  if (draftEdgeM) out.live.draftEdge = draftEdgeM[1];

  // Ticket cashout / stake (for HOLD vs CASHOUT)
  const cashM = text.match(/\bcashout\s*(?:R\$\s*)?(\d+(?:\.\d+)?)/i);
  if (cashM) out.live.cashout = cashM[1];
  const stakeM = text.match(/\bstake\s*(?:R\$\s*)?(\d+(?:\.\d+)?)/i);
  if (stakeM) {
    // Prefer attaching to first winner bet
    if (out.bets.length) out.bets[0].stake = stakeM[1];
    out.live.stake = stakeM[1];
  }

  if (!out.team1 || !out.team2) out.warnings.push("Teams incomplete.");
  if (out.blue.length && out.blue.length < 5) out.warnings.push(`Blue draft ${out.blue.length}/5.`);
  if (out.red.length && out.red.length < 5) out.warnings.push(`Red draft ${out.red.length}/5.`);

  return out;
}

function sideNames(state) {
  const t1 = resolveTeam(state.team1) || "Team1";
  const t2 = resolveTeam(state.team2) || "Team2";
  const blueIsT1 = state.blueTeam !== "team2";
  return {
    t1,
    t2,
    blue: blueIsT1 ? t1 : t2,
    red: blueIsT1 ? t2 : t1,
  };
}

function fmtGoldLead(blue, red, blueName, redName) {
  if (blue === "" || red === "" || Number.isNaN(+blue) || Number.isNaN(+red)) return "";
  const d = +red - +blue;
  if (Math.abs(d) < 0.05) return "even";
  if (d > 0) return `${redName} +${d.toFixed(1)}k`;
  return `${blueName} +${Math.abs(d).toFixed(1)}k`;
}

/** LIVE block lines only (clock, scoreboard, cashout/ticket). No draft / bets. */
function composeLiveLines(state) {
  const { blue, red } = sideNames(state);
  const L = state.live || {};
  const lines = [];
  const clock = (L.clock || "").trim() || "??:??";
  lines.push(`LIVE ${clock}`);
  const parts = [];
  if (L.killsBlue !== undefined && L.killsBlue !== "" && L.killsRed !== undefined && L.killsRed !== "") {
    parts.push(`Kills ${L.killsBlue}-${L.killsRed}`);
  }
  if (L.goldBlue !== undefined && L.goldBlue !== "" && L.goldRed !== undefined && L.goldRed !== "") {
    const lead = fmtGoldLead(L.goldBlue, L.goldRed, blue, red);
    parts.push(`Gold ${L.goldBlue}k-${L.goldRed}k${lead ? ` (${lead})` : ""}`);
  }
  if (L.towersBlue !== undefined && L.towersBlue !== "" && L.towersRed !== undefined && L.towersRed !== "") {
    parts.push(`Towers ${L.towersBlue}-${L.towersRed}`);
  }
  if (parts.length) lines.push(parts.join(" | "));
  const obj = [];
  if (L.dragonsBlue !== undefined && L.dragonsBlue !== "" || L.dragonsRed !== undefined && L.dragonsRed !== "") {
    obj.push(`Dragons ${blue} ${L.dragonsBlue || "0"} / ${red} ${L.dragonsRed || "0"}`);
  }
  if (L.grubsBlue !== undefined && L.grubsBlue !== "" || L.grubsRed !== undefined && L.grubsRed !== "") {
    obj.push(`Grubs ${blue} ${L.grubsBlue || "0"} / ${red} ${L.grubsRed || "0"}`);
  }
  if (L.baron === "none") obj.push("Baron none");
  else if (L.baron === "blue") obj.push(`Baron ${blue}`);
  else if (L.baron === "red") obj.push(`Baron ${red}`);
  if (L.herald === "none") obj.push("Herald none");
  else if (L.herald === "blue") obj.push(`Herald ${blue}`);
  else if (L.herald === "red") obj.push(`Herald ${red}`);
  else if (L.herald === "channeling_blue") obj.push(`Herald channeling ${blue}`);
  else if (L.herald === "channeling_red") obj.push(`Herald channeling ${red}`);
  if (obj.length) lines.push(obj.join(" | "));
  if (L.pPre) {
    const side = L.pPreSide || blue;
    lines.push(`pre ${side} ${(100 * +L.pPre).toFixed(0)}%`);
  }
  if (L.draftEdge !== undefined && L.draftEdge !== "") lines.push(`draft_edge ${L.draftEdge}`);
  if (L.cashout) lines.push(`cashout R$${L.cashout}`);
  if (L.stake || L.ticketOdds) {
    const bits = ["Ticket"];
    if (L.ticketOdds) bits.push(`@ ${L.ticketOdds}`);
    if (L.stake) bits.push(`stake R$${L.stake}`);
    lines.push(bits.join(" "));
  }
  if (L.liveNote) lines.push(`Note: ${L.liveNote}`);
  return lines;
}

/** Live-only paste: thin matchup header + LIVE block (no draft, no bet list). */
function composeLiveOnly(state) {
  const { t1, t2, blue, red } = sideNames(state);
  const map = state.map || "?";
  const league = state.league || "?";
  const lines = [
    `${t1} vs ${t2} | ${league} | Map ${map} | Blue ${blue} / Red ${red}`,
    "",
    ...composeLiveLines(state),
  ];
  return lines.join("\n");
}

function composeFromState(state) {
  const { t1, t2, blue, red } = sideNames(state);
  const map = state.map || "?";
  const league = state.league || "?";
  const series = (state.series || "").trim();
  const blueChamps = state.blue || [];
  const redChamps = state.red || [];

  const lines = [];
  lines.push(`${t1} vs ${t2} | ${league} | Map ${map}${series ? ` | ${series}` : ""}`);
  lines.push(`BLUE ${blue}: ${blueChamps.length ? blueChamps.join(", ") : "(no draft)"}`);
  lines.push(`RED ${red}: ${redChamps.length ? redChamps.join(", ") : "(no draft)"}`);

  if (state.includeLive) {
    lines.push("");
    lines.push(...composeLiveLines(state));
  }

  if (state.includeBets !== false && state.bets && state.bets.length) {
    lines.push("");
    lines.push("Bets:");
    for (const b of state.bets) {
      const bits = [`- ${b.selection || b.market}`];
      if (b.market && b.selection && !String(b.selection).toLowerCase().includes(String(b.market).toLowerCase().slice(0, 8))) {
        bits[0] = `- [${b.market}] ${b.selection}`;
      }
      if (b.odds) bits.push(`@ ${b.odds}`);
      if (b.stake) bits.push(`stake R$${b.stake}`);
      lines.push(bits.join(" "));
    }
  }

  return lines.join("\n");
}

function readFormState() {
  const fd = new FormData($("#form"));
  const blue = ROLES.map((_, i) => (fd.get(`blue${i}`) || "").trim()).filter(Boolean);
  const red = ROLES.map((_, i) => (fd.get(`red${i}`) || "").trim()).filter(Boolean);
  const bets = $$(".bet-row").map((row) => ({
    market: row.querySelector(".bet-market").value,
    selection: row.querySelector(".bet-sel").value.trim(),
    odds: row.querySelector(".bet-odds").value,
    stake: row.querySelector(".bet-stake").value,
  })).filter((b) => b.selection || b.odds);

  return {
    team1: resolveTeam(fd.get("team1")),
    team2: resolveTeam(fd.get("team2")),
    league: fd.get("league"),
    map: fd.get("map") || "1",
    blueTeam: fd.get("blueTeam") || "team1",
    series: (fd.get("series") || "").trim(),
    blue,
    red,
    includeLive: fd.get("includeLive") === "on",
    includeBets: fd.get("includeBets") === "on",
    live: {
      clock: fd.get("clock"),
      killsBlue: fd.get("killsBlue"),
      killsRed: fd.get("killsRed"),
      goldBlue: fd.get("goldBlue"),
      goldRed: fd.get("goldRed"),
      towersBlue: fd.get("towersBlue"),
      towersRed: fd.get("towersRed"),
      dragonsBlue: fd.get("dragonsBlue"),
      dragonsRed: fd.get("dragonsRed"),
      grubsBlue: fd.get("grubsBlue"),
      grubsRed: fd.get("grubsRed"),
      baron: fd.get("baron"),
      herald: fd.get("herald"),
      liveNote: fd.get("liveNote"),
    },
    bets,
    warnings: [],
  };
}

function applyStateToForm(state) {
  SUPPRESS_FORM = true;
  const form = $("#form");
  form.elements.namedItem("team1").value = state.team1 || "";
  form.elements.namedItem("team2").value = state.team2 || "";
  if (state.league) form.elements.namedItem("league").value = state.league;
  form.elements.namedItem("map").value = state.map || "1";
  form.elements.namedItem("blueTeam").value = state.blueTeam || "team1";
  form.elements.namedItem("series").value = state.series || "";
  form.elements.namedItem("includeLive").checked = !!state.includeLive;
  form.elements.namedItem("includeBets").checked = state.includeBets !== false;

  ROLES.forEach((_, i) => {
    const b = form.elements.namedItem(`blue${i}`);
    const r = form.elements.namedItem(`red${i}`);
    if (b) b.value = state.blue[i] || "";
    if (r) r.value = state.red[i] || "";
  });

  const L = state.live || {};
  for (const k of [
    "clock","killsBlue","killsRed","goldBlue","goldRed","towersBlue","towersRed",
    "dragonsBlue","dragonsRed","grubsBlue","grubsRed","baron","herald","liveNote",
  ]) {
    const el = form.elements.namedItem(k);
    if (el) el.value = L[k] ?? "";
  }

  $("#betRows").innerHTML = "";
  const bets = state.bets && state.bets.length ? state.bets : [{}];
  bets.forEach(addBetRow);
  $(".live-fields").dataset.enabled = state.includeLive ? "true" : "false";
  const { blue, red } = sideNames(state);
  $("#blueLabel").textContent = `Blue · ${blue}`;
  $("#redLabel").textContent = `Red · ${red}`;
  SUPPRESS_FORM = false;
}

function renderChips(state) {
  const chips = $("#chips");
  chips.innerHTML = "";
  const bits = [];
  if (state.team1 && state.team2) bits.push(`${state.team1} vs ${state.team2}`);
  if (state.league) bits.push(state.league);
  if (state.map) bits.push(`Map ${state.map}`);
  if (state.blue.length) bits.push(`Blue ${state.blue.length}/5`);
  if (state.red.length) bits.push(`Red ${state.red.length}/5`);
  if (state.bets.length) bits.push(`${state.bets.length} bet${state.bets.length > 1 ? "s" : ""}`);
  if (state.includeLive) bits.push("Live");
  if (state.red.includes("Locke") || state.blue.includes("Locke")) bits.push("Locke ✓");
  for (const w of state.warnings || []) bits.push(`⚠ ${w}`);
  for (const b of bits) {
    const span = document.createElement("span");
    span.className = "chip" + (b.startsWith("⚠") ? " warn" : "");
    span.textContent = b;
    chips.appendChild(span);
  }
}

function numOr(v, d = 0) {
  if (v === undefined || v === null || v === "") return d;
  const n = +v;
  return Number.isFinite(n) ? n : d;
}

function pickTicketSide(state) {
  const { blue, red } = sideNames(state);
  const L = state.live || {};
  const winners = (state.bets || []).filter(
    (b) => !b.market || /winner/i.test(b.market) || !/kills|inhib|blood|race/i.test(b.market || "")
  );
  for (const b of winners) {
    const sel = resolveTeam(b.selection || "");
    if (!sel) continue;
    if (sel.toLowerCase() === blue.toLowerCase()) {
      return {
        side: "blue",
        name: blue,
        odds: b.odds || L.ticketOdds,
        stake: b.stake || L.stake,
      };
    }
    if (sel.toLowerCase() === red.toLowerCase()) {
      return {
        side: "red",
        name: red,
        odds: b.odds || L.ticketOdds,
        stake: b.stake || L.stake,
      };
    }
  }
  return {
    side: "blue",
    name: blue,
    odds: L.ticketOdds || null,
    stake: L.stake || null,
  };
}

function bookImpliedPPre(state, ticket) {
  /** No-vig (or single-price) prior from Winner odds in dump/pad. */
  const { blue, red } = sideNames(state);
  const prices = {};
  for (const b of state.bets || []) {
    if (b.market && !/winner/i.test(b.market) && /kills|inhib|blood|race/i.test(b.market)) continue;
    const sel = resolveTeam(b.selection || "");
    const odds = +b.odds;
    if (!sel || !Number.isFinite(odds) || odds <= 1) continue;
    prices[sel.toLowerCase()] = odds;
  }
  if (ticket.odds && ticket.name) {
    const o = +ticket.odds;
    if (Number.isFinite(o) && o > 1) prices[ticket.name.toLowerCase()] = o;
  }
  const ob = prices[blue.toLowerCase()];
  const or_ = prices[red.toLowerCase()];
  if (ob && or_) {
    const ib = 1 / ob;
    const ir = 1 / or_;
    const pBlue = ib / (ib + ir);
    return {
      pBlue,
      pTicket: ticket.side === "blue" ? pBlue : 1 - pBlue,
      source: "book",
    };
  }
  if (ticket.odds && +ticket.odds > 1) {
    const pTicket = 1 / +ticket.odds;
    const pBlue = ticket.side === "blue" ? pTicket : 1 - pTicket;
    return { pBlue, pTicket, source: "book-1way" };
  }
  return null;
}

function resolvePPre(state, ticket) {
  const L = state.live || {};
  let pBlue = null;
  let source = "50%";
  if (L.pPre !== undefined && L.pPre !== "") {
    const p = +L.pPre;
    if (Number.isFinite(p) && p > 0 && p < 1) {
      const sideName = L.pPreSide ? resolveTeam(L.pPreSide) : null;
      const { blue, red } = sideNames(state);
      if (sideName && sideName.toLowerCase() === red.toLowerCase()) pBlue = 1 - p;
      else pBlue = p;
      source = "typed";
    }
  }
  if (pBlue == null) {
    const book = bookImpliedPPre(state, ticket);
    if (book) {
      pBlue = book.pBlue;
      source = book.source;
    }
  }
  if (pBlue == null) {
    const cached = localStorage.getItem("lol-slip-composer-ppre");
    if (cached) {
      try {
        const c = JSON.parse(cached);
        if (c && c.blue != null) {
          pBlue = +c.blue;
          source = "cached";
        }
      } catch { /* ignore */ }
    }
  }
  if (pBlue == null) pBlue = 0.5;
  const pTicket = ticket.side === "blue" ? pBlue : 1 - pBlue;
  return { pBlue, pTicket, source };
}

function liveGlanceInputs(state) {
  const L = state.live || {};
  if (!state.includeLive) return null;
  const minute = window.LiveWin?.clockToMinute(L.clock);
  const hasKills = L.killsBlue !== "" && L.killsBlue != null && L.killsRed !== "" && L.killsRed != null;
  const hasGold = L.goldBlue !== "" && L.goldBlue != null && L.goldRed !== "" && L.goldRed != null;
  if (minute == null || (!hasKills && !hasGold)) return null;

  const ticket = pickTicketSide(state);
  const { pBlue, pTicket, source: pPreSource } = resolvePPre(state, ticket);

  const kb = numOr(L.killsBlue);
  const kr = numOr(L.killsRed);
  const gb = numOr(L.goldBlue) * 1000;
  const gr = numOr(L.goldRed) * 1000;
  const db = numOr(L.dragonsBlue);
  const dr = numOr(L.dragonsRed);
  const vb = L.grubsBlue !== "" && L.grubsBlue != null ? numOr(L.grubsBlue) : null;
  const vr = L.grubsRed !== "" && L.grubsRed != null ? numOr(L.grubsRed) : null;
  const tb = numOr(L.towersBlue);
  const tr = numOr(L.towersRed);

  // Score from ticket side (this team − opp)
  const fromBlue = ticket.side === "blue";
  const opts = {
    p_pre: pTicket,
    minute,
    kill_diff: fromBlue ? kb - kr : kr - kb,
    gold_diff: fromBlue ? gb - gr : gr - gb,
    dragons: fromBlue ? db : dr,
    opp_dragons: fromBlue ? dr : db,
    towers: fromBlue ? tb : tr,
    opp_towers: fromBlue ? tr : tb,
    void_grubs_blue: vb,
    void_grubs_red: vr,
    // When scoring red, flip grub sides for net
    draft_edge: L.draftEdge !== undefined && L.draftEdge !== "" ? +L.draftEdge : null,
  };
  if (!fromBlue && (vb != null || vr != null)) {
    opts.void_grubs_blue = vr;
    opts.void_grubs_red = vb;
  }
  if (ticket.stake && ticket.odds) {
    opts.stake = +ticket.stake;
    opts.odds = +ticket.odds;
    if (L.cashout) opts.cashout = +L.cashout;
  } else if (L.stake && ticket.odds) {
    opts.stake = +L.stake;
    opts.odds = +ticket.odds;
    if (L.cashout) opts.cashout = +L.cashout;
  }
  return { opts, ticket, pBlue, minute, L, pPreSource };
}

function fmtPp(v) {
  const n = +v;
  const s = n > 0 ? `+${n.toFixed(1)}` : n.toFixed(1);
  return `${s}pp`;
}

function channelChipLabel(ch, L, ticket) {
  const fromBlue = ticket.side === "blue";
  const kb = numOr(L.killsBlue);
  const kr = numOr(L.killsRed);
  const gb = numOr(L.goldBlue);
  const gr = numOr(L.goldRed);
  const db = numOr(L.dragonsBlue);
  const dr = numOr(L.dragonsRed);
  const vb = numOr(L.grubsBlue);
  const vr = numOr(L.grubsRed);
  const tb = numOr(L.towersBlue);
  const tr = numOr(L.towersRed);
  if (ch === "gold") {
    const a = fromBlue ? gb : gr;
    const b = fromBlue ? gr : gb;
    const lead = (a - b).toFixed(1);
    return `gold ${a}k–${b}k (${lead > 0 ? "+" : ""}${lead}k)`;
  }
  if (ch === "kills") {
    const a = fromBlue ? kb : kr;
    const b = fromBlue ? kr : kb;
    return `kills ${a}–${b}`;
  }
  if (ch === "dragons") {
    const a = fromBlue ? db : dr;
    const b = fromBlue ? dr : db;
    return `drakes ${a}–${b}`;
  }
  if (ch === "void_grubs") {
    const a = fromBlue ? vb : vr;
    const b = fromBlue ? vr : vb;
    return `grubs ${a}–${b}`;
  }
  if (ch === "towers") {
    const a = fromBlue ? tb : tr;
    const b = fromBlue ? tr : tb;
    return `towers ${a}–${b}`;
  }
  return ch;
}

function renderGlance(state) {
  const card = $("#glanceCard");
  const strip = $("#glanceStrip");
  const chips = $("#glanceChips");
  const meta = $("#glanceMeta");
  const phaseEl = $("#glancePhase");
  if (!card || !window.LiveWin) return;

  const inputs = liveGlanceInputs(state);
  if (!inputs) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const { opts, ticket, L, pPreSource } = inputs;
  const br = LiveWin.objectiveDeltaPpBreakdown(opts);
  phaseEl.textContent = `${br.phase || "—"} · ${opts.minute.toFixed(1)}'`;

  const pct = (100 * br.p_win).toFixed(0);
  const oppPct = (100 * (1 - br.p_win)).toFixed(0);
  const { blue, red } = sideNames(state);
  const oppName = ticket.side === "blue" ? red : blue;
  const kickPct = (100 * opts.p_pre).toFixed(0);

  let html = `
    <div class="glance-pct">${pct}%<span>${ticket.name}</span></div>
    <div class="glance-fair">fair ${br.fair_odds.toFixed(2)} · ${oppName} ${oppPct}% / ${br.fair_odds_opp.toFixed(2)}</div>
    <div class="glance-delta">vs kickoff ${kickPct}% (${pPreSource || "?"}) ${fmtPp(br.delta_vs_pre_pp)}</div>
  `;
  if (br.ticket) {
    html += `<div class="glance-fair">fair cashout ≈ R$${br.ticket.fair_cashout.toFixed(2)}</div>`;
  }
  if (br.cashout) {
    const v = br.cashout.verdict;
    html += `<div class="glance-verdict ${v === "HOLD" ? "hold" : "cashout"}">${v}</div>`;
  }
  strip.innerHTML = html;

  chips.innerHTML = "";
  for (const t of br.top || []) {
    const el = document.createElement("span");
    el.className = "glance-chip " + (t.delta_pp < 0 ? "neg" : "pos");
    el.textContent = `${channelChipLabel(t.channel, L, ticket)} ${fmtPp(t.delta_pp)}`;
    chips.appendChild(el);
  }
  if (br.grubs_research) {
    const el = document.createElement("span");
    el.className = "glance-chip research";
    el.title = br.grubs_research.note;
    el.textContent = `grubs research contest +${br.grubs_research.delta_pp}pp (not in WR)`;
    chips.appendChild(el);
  }
  if (L.baron || L.herald) {
    const el = document.createElement("span");
    el.className = "glance-chip research";
    el.textContent = "Baron/Herald not in WR model";
    chips.appendChild(el);
  }

  const bits = [`method ${br.method}`];
  if (opts.draft_edge == null) bits.push("softcap (add draft_edge for OE matrix)");
  if (br.cashout) bits.push(br.cashout.reason);
  meta.textContent = bits.join(" · ");

  // Cache blue-side pre for next dump
  try {
    const pBlue = ticket.side === "blue" ? opts.p_pre : 1 - opts.p_pre;
    localStorage.setItem("lol-slip-composer-ppre", JSON.stringify({ blue: pBlue }));
  } catch { /* ignore */ }
}

function livePadFields() {
  return $$("#livePad [data-live]");
}

function readLivePad() {
  const L = {};
  for (const el of livePadFields()) {
    const key = el.dataset.live;
    const v = (el.value || "").trim();
    if (key === "pPrePct") {
      if (v !== "" && Number.isFinite(+v)) L.pPre = String(+v / 100);
      continue;
    }
    if (key === "ticketOdds") {
      if (v !== "") L.ticketOdds = v;
      continue;
    }
    if (v !== "") L[key] = v;
  }
  return L;
}

function livePadHasSignal(L) {
  if (!L) return false;
  return Object.keys(L).some((k) => L[k] !== undefined && L[k] !== "");
}

function applyLivePad(live) {
  const L = live || {};
  SUPPRESS_LIVE_PAD = true;
  for (const el of livePadFields()) {
    const key = el.dataset.live;
    if (key === "pPrePct") {
      el.value = L.pPre !== undefined && L.pPre !== "" ? String(Math.round(+L.pPre * 100)) : "";
    } else if (key === "ticketOdds") {
      el.value = L.ticketOdds || "";
    } else {
      el.value = L[key] ?? "";
    }
  }
  SUPPRESS_LIVE_PAD = false;
}

function mergePadIntoState(state) {
  const pad = readLivePad();
  if (!livePadHasSignal(pad)) return state;
  const live = { ...(state.live || {}) };
  for (const [k, v] of Object.entries(pad)) {
    if (v !== undefined && v !== "") live[k] = v;
  }
  state.live = live;
  state.includeLive = true;
  if (pad.stake || pad.ticketOdds) {
    if (!state.bets) state.bets = [];
    if (!state.bets.length) {
      const { blue } = sideNames(state);
      state.bets.push({
        market: "Winner",
        selection: blue,
        odds: pad.ticketOdds || "",
        stake: pad.stake || "",
      });
    } else {
      if (pad.stake) state.bets[0].stake = pad.stake;
      if (pad.ticketOdds) state.bets[0].odds = pad.ticketOdds;
    }
  }
  return state;
}

function buildStateForPaste() {
  /** Always merge Live pad into dump/form — used by preview + Copy. */
  let state;
  if (MODE === "form") state = readFormState();
  else {
    const dump = $("#dump").value;
    state = dump.trim() ? parseDump(dump) : readFormState();
  }
  const pad = readLivePad();
  if (livePadHasSignal(pad)) {
    state = mergePadIntoState(state);
  }
  return state;
}

function currentState() {
  return buildStateForPaste();
}

function render() {
  const state = currentState();
  if (MODE === "nl" && $("#dump").value.trim()) {
    applyStateToForm(state);
  }
  // Sync pad from dump only when user isn't mid-edit on pad
  if (!LIVE_PAD_DIRTY && !SUPPRESS_LIVE_PAD) {
    applyLivePad(state.live);
  }
  const { blue, red } = sideNames(state);
  $("#blueLabel").textContent = `Blue · ${blue}`;
  $("#redLabel").textContent = `Red · ${red}`;
  $(".live-fields").dataset.enabled = state.includeLive ? "true" : "false";
  $("#preview").textContent = composeFromState(state);
  renderChips(state);
  renderGlance(state);
  $("#modePill").textContent = MODE === "nl" ? "natural language" : "fine-tune";
  save(state);
}

function save(state) {
  const s = state || currentState();
  localStorage.setItem("lol-slip-composer-v2", JSON.stringify({
    dump: $("#dump").value,
    mode: MODE,
    state: s,
  }));
}

function buildRoleInputs(containerId, prefix) {
  const root = $(containerId);
  root.innerHTML = "";
  ROLES.forEach((role, i) => {
    const row = document.createElement("div");
    row.className = "role-row";
    row.innerHTML = `
      <span>${role}</span>
      <input name="${prefix}${i}" list="champs" placeholder="${role}" />
    `;
    root.appendChild(row);
  });
}

function addBetRow(data = {}) {
  const row = document.createElement("div");
  row.className = "bet-row";
  const opts = MARKET_PRESETS.map(
    (m) => `<option value="${m}" ${data.market === m ? "selected" : ""}>${m}</option>`
  ).join("");
  row.innerHTML = `
    <label>Market<select class="bet-market">${opts}</select></label>
    <label>Selection<input class="bet-sel" placeholder="G2 / Under 34.5" value="${data.selection || ""}" /></label>
    <label>Odds<input class="bet-odds" type="number" step="0.01" min="1.01" placeholder="1.87" value="${data.odds || ""}" /></label>
    <label>Stake<input class="bet-stake" type="number" step="0.01" min="0" placeholder="10" value="${data.stake || ""}" /></label>
    <button type="button" class="icon-btn" title="Remove">×</button>
  `;
  row.querySelector(".icon-btn").addEventListener("click", () => {
    row.remove();
    MODE = "form";
    render();
  });
  $$("input, select", row).forEach((el) => {
    el.addEventListener("input", () => {
      if (SUPPRESS_FORM) return;
      MODE = "form";
      render();
    });
  });
  $("#betRows").appendChild(row);
}

async function copyPaste() {
  const state = buildStateForPaste();
  const text = composeFromState(state);
  $("#preview").textContent = text;
  try {
    await navigator.clipboard.writeText(text);
    const hasLive = !!(state.includeLive && state.live && livePadHasSignal(state.live));
    $("#status").textContent = hasLive
      ? "Copied Dump + Live — paste into Cursor. (Use Copy live for LIVE-only.)"
      : "Copied Dump only — Live pad empty. Fill Live pad, then Copy live or Copy for Cursor.";
    $("#status").className = "status ok";
  } catch {
    $("#status").textContent = "Clipboard blocked — select the paste block and copy manually.";
    $("#status").className = "status err";
  }
}

async function copyLiveOnly() {
  const pad = readLivePad();
  if (!livePadHasSignal(pad)) {
    $("#status").textContent = "Live pad empty — fill clock/kills/gold first.";
    $("#status").className = "status err";
    return;
  }
  const state = buildStateForPaste();
  const text = composeLiveOnly(state);
  $("#preview").textContent = text;
  try {
    await navigator.clipboard.writeText(text);
    $("#status").textContent = "Copied LIVE only (no draft) — paste into Cursor.";
    $("#status").className = "status ok";
  } catch {
    $("#status").textContent = "Clipboard blocked — select the paste block and copy manually.";
    $("#status").className = "status err";
  }
}

function clearAll() {
  localStorage.removeItem("lol-slip-composer-v2");
  localStorage.removeItem("lol-slip-composer");
  $("#dump").value = "";
  $("#form").reset();
  $("#betRows").innerHTML = "";
  addBetRow();
  LIVE_PAD_DIRTY = false;
  applyLivePad({});
  MODE = "nl";
  render();
  $("#status").textContent = "Cleared.";
  $("#status").className = "status";
  $("#dump").focus();
}

function clearLivePad() {
  LIVE_PAD_DIRTY = false;
  applyLivePad({});
  // Strip live lines from being re-applied: mark pad empty so dump-only live can return
  render();
  $("#status").textContent = "Live pad cleared — Dump live lines still apply if present.";
  $("#status").className = "status";
}

function loadExample() {
  $("#dump").value = `MKOI vs G2 · LEC map 2
Blue: Gnar, Xin Zhao, Orianna, Syndra, Tahm Kench
Red: Rumble, Vi, Yone, Lucian, Leona
G2 @ 1.59 · MKOI @ 2.30`;
  LIVE_PAD_DIRTY = true;
  applyLivePad({
    clock: "18:00",
    killsBlue: "4",
    killsRed: "8",
    goldBlue: "31.8",
    goldRed: "36.3",
    towersBlue: "0",
    towersRed: "1",
    dragonsBlue: "0",
    dragonsRed: "3",
    grubsBlue: "1",
    grubsRed: "2",
    pPre: "",
    stake: "1.81",
    ticketOdds: "5.70",
    cashout: "1.40",
  });
  MODE = "nl";
  render();
  $("#status").textContent = "Example: Kickoff % blank → book. Copy merges Dump + Live pad.";
  $("#status").className = "status ok";
  const clockEl = $('#livePad [data-live="clock"]');
  if (clockEl) clockEl.focus();
}

async function loadTeamData() {
  try {
    const res = await fetch("teams.json", { cache: "no-store" });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    TEAM_DATA = {
      teams: data.teams || [],
      aliases: { ...TEAM_DATA.aliases, ...(data.aliases || {}) },
      leagues: data.leagues || TEAM_DATA.leagues,
    };
  } catch (e) {
    console.warn("teams.json missing", e);
    TEAM_DATA.teams = [
      "T1", "Gen.G", "Hanwha Life Esports", "Dplus Kia", "Movistar KOI",
      "Karmine Corp", "G2 Esports", "Fnatic", "Team Vitality", "Cloud9",
    ];
  }

  const list = $("#teams");
  list.innerHTML = "";
  const priority = ["Movistar KOI", "G2 Esports", "Karmine Corp", "Dplus Kia", "T1", "Gen.G"];
  const rest = TEAM_DATA.teams.filter((t) => !priority.includes(t)).sort((a, b) => a.localeCompare(b));
  [...priority.filter((t) => TEAM_DATA.teams.includes(t)), ...rest].forEach((t) => {
    const o = document.createElement("option");
    o.value = t;
    list.appendChild(o);
  });
  Object.keys(TEAM_DATA.aliases).forEach((alias) => {
    const o = document.createElement("option");
    o.value = alias;
    list.appendChild(o);
  });

  const leagueSel = $("#form").elements.namedItem("league");
  leagueSel.innerHTML = "";
  TEAM_DATA.leagues.forEach((lg) => {
    const o = document.createElement("option");
    o.value = lg;
    o.textContent = lg;
    leagueSel.appendChild(o);
  });
  leagueSel.value = TEAM_DATA.leagues.includes("LEC") ? "LEC" : TEAM_DATA.leagues[0];
}

function init() {
  buildChampIndex();
  const list = $("#champs");
  CHAMPS.forEach((c) => {
    const o = document.createElement("option");
    o.value = c;
    list.appendChild(o);
  });

  buildRoleInputs("#blueRoles", "blue");
  buildRoleInputs("#redRoles", "red");

  loadTeamData().then(async () => {
    if (window.LiveWin) {
      try {
        await LiveWin.loadCoefs(".");
      } catch (e) {
        console.warn("live coefs missing", e);
      }
    }
    const saved = localStorage.getItem("lol-slip-composer-v2");
    if (saved) {
      try {
        const data = JSON.parse(saved);
        if (data.dump) $("#dump").value = data.dump;
        MODE = data.mode || "nl";
        if (data.state && !data.dump) applyStateToForm(data.state);
      } catch { /* ignore */ }
    }
    if (!$("#betRows").children.length) addBetRow();

    $("#dump").addEventListener("input", () => {
      MODE = "nl";
      LIVE_PAD_DIRTY = false; // dump rewrite may refresh live fields
      render();
    });
    $("#form").addEventListener("input", () => {
      if (SUPPRESS_FORM) return;
      MODE = "form";
      // Keep dump in sync as optional note — don't wipe it
      render();
    });
    $("#fineTune").addEventListener("toggle", () => {
      if ($("#fineTune").open) MODE = "form";
    });
    $("#addBet").addEventListener("click", () => {
      MODE = "form";
      addBetRow();
      render();
    });
    $("#copy").addEventListener("click", copyPaste);
    const copyLiveBtn = $("#copyLive");
    if (copyLiveBtn) copyLiveBtn.addEventListener("click", copyLiveOnly);
    $("#loadExample").addEventListener("click", loadExample);
    $("#clear").addEventListener("click", clearAll);
    const livePadClear = $("#livePadClear");
    if (livePadClear) livePadClear.addEventListener("click", clearLivePad);
    for (const el of livePadFields()) {
      el.addEventListener("input", () => {
        if (SUPPRESS_LIVE_PAD) return;
        LIVE_PAD_DIRTY = true;
        render();
      });
    }

    // Cmd/Ctrl+Enter → copy
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        copyPaste();
      }
    });

    render();
    if (!$("#dump").value.trim()) $("#dump").focus();
  });
}

init();
