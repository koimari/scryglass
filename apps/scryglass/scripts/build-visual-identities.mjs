import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const teamOutputPath = path.join(appRoot, "src", "data", "teamVisualIdentities.json");
const playerOutputPath = path.join(appRoot, "src", "data", "playerVisualIdentities.json");
const reportPath = path.join(appRoot, "visual-identity-report.json");
const cacheDir = process.env.SCRYGLASS_IDENTITY_CACHE?.trim() || "";
const blobRoot = "https://97gks2fobqkgppwx.public.blob.vercel-storage.com";
const cargoExport = "https://lol.fandom.com/wiki/Special:CargoExport";
const mediaWikiApi = "https://lol.fandom.com/api.php";

const TEAM_ALIASES = new Map([
  ["løs", "los"],
  ["ground zero", "ground zero gaming"],
  ["giantx", "giantx"],
]);

function text(value) {
  return value === null || value === undefined ? "" : String(value).trim();
}

function normalized(value) {
  return text(value)
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replaceAll("ø", "o")
    .replaceAll("Ø", "O")
    .replaceAll("ł", "l")
    .replaceAll("Ł", "L")
    .replaceAll("đ", "d")
    .replaceAll("Đ", "D")
    .replaceAll("ß", "ss")
    .toLowerCase()
    .replaceAll("&", " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

const PLAYER_IMAGE_OVERRIDES = new Map([
  ["Van1", "Fukuoka SoftBank HAWKS gaming", "QTD_Van_2025_Split_3.png", "Van1"],
  ["1jw", "Inferno Esports", "IMP_LJW_2023_Split_2.png", "1jw"],
  ["Moopz", "Inferno Esports", "IE_Moopz_LCP_WC_2025.png", "Moopz (Marc Jazztine Barrion)"],
  ["HOYA", "Ninjas in Pyjamas", "GRF_Hoya_2020_Split_2.png", "HOYA"],
  ["Violet", "Saving OCE", "MKZ_Violet.png", "Violet (Lim Doo-sung)"],
  ["Krimson", "Inferno Esports", "IE_Krimson_LCP_WC_2025.png", "Krimson (Karl Justin Guevarra)"],
  ["Castle", "LYON", "KT.C_Castle_2021_Split_1.png", "Castle (Cho Hyeon-seong)"],
  ["M G", "LØS", "LOS MG 2025 Split 1.png", "M G (Lee Ji-hoon)"],
  ["Minji", "Ground Zero Gaming", "JT Minji 2024 Split 2.png", "Minji"],
].map(([player, team, file, page]) => [
  `${normalized(player)}|${normalized(team)}`,
  { file, page },
]));

function broadTeamKey(value) {
  return normalized(value)
    .replace(/\b20\d{2}\b/g, " ")
    .replace(/\b(?:esports?|e sports?|gaming|team|club)\b/g, " ")
    .replace(/\b(?:american|brazilian|chinese|korean|polish|portuguese|french|latin american)\b/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function playerHandle(value) {
  return text(value).replace(/\s+\([^)]*\)$/, "");
}

function wikiPageUrl(page) {
  return `https://lol.fandom.com/wiki/${encodeURIComponent(text(page).replaceAll(" ", "_"))}`;
}

async function fetchJson(url, attempts = 3) {
  let latestError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { "user-agent": "Scryglass visual identity builder/1.0" },
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return await response.json();
    } catch (error) {
      latestError = error;
      if (attempt < attempts) await new Promise((resolve) => setTimeout(resolve, attempt * 1_000));
    }
  }
  throw latestError;
}

function cargoUrl({ tables, fields, joinOn, where, offset = 0 }) {
  const url = new URL(cargoExport);
  url.searchParams.set("tables", tables);
  url.searchParams.set("fields", fields);
  url.searchParams.set("format", "json");
  url.searchParams.set("limit", "5000");
  url.searchParams.set("offset", String(offset));
  if (joinOn) url.searchParams.set("join_on", joinOn);
  if (where) url.searchParams.set("where", where);
  return url.toString();
}

async function cachedJson(fileName, url) {
  if (cacheDir) {
    try {
      return JSON.parse(await fs.readFile(path.join(cacheDir, fileName), "utf8"));
    } catch {
      // Fetch below when the optional local cache does not contain this page.
    }
  }
  const payload = await fetchJson(url);
  if (cacheDir) {
    await fs.mkdir(cacheDir, { recursive: true });
    await fs.writeFile(path.join(cacheDir, fileName), `${JSON.stringify(payload)}\n`, "utf8");
  }
  return payload;
}

function originalPngUrl(url) {
  return text(url).replace(/\?.*$/, "?format=original");
}

async function resolveWikiImageUrls(fileNames) {
  const files = [...new Set(fileNames.map(text).filter((file) => /\.png$/i.test(file)))];
  const resolved = new Map();
  for (let index = 0; index < files.length; index += 25) {
    const chunk = files.slice(index, index + 25);
    const url = new URL(mediaWikiApi);
    url.searchParams.set("action", "query");
    url.searchParams.set("prop", "imageinfo");
    url.searchParams.set("iiprop", "url|mime");
    url.searchParams.set("titles", chunk.map((file) => `File:${file}`).join("|"));
    url.searchParams.set("format", "json");
    url.searchParams.set("origin", "*");
    const signature = createHash("md5").update(chunk.join("\n")).digest("hex");
    const payload = await cachedJson(`image-info-${signature}.json`, url.toString());
    for (const page of Object.values(payload.query?.pages ?? {})) {
      const file = text(page.title).replace(/^File:/i, "");
      const info = page.imageinfo?.[0];
      if (info?.mime === "image/png" && info.url) resolved.set(normalized(file), originalPngUrl(info.url));
    }
  }
  return resolved;
}

async function cargoPages(name, query) {
  const rows = [];
  for (let offset = 0; ; offset += 5000) {
    const page = await cachedJson(`${name}-${offset}.json`, cargoUrl({ ...query, offset }));
    rows.push(...page);
    if (page.length < 5000) break;
  }
  return rows;
}

async function currentPack() {
  const configured = process.env.SCRYGLASS_IDENTITY_PACK_BASE_URL?.trim();
  if (configured) return configured.replace(/\/$/, "");
  const manifest = await fetchJson(`${blobRoot}/packs/manifest.json`);
  if (!/^https:\/\//.test(manifest.base_url || "")) throw new Error("Public pack base URL is unavailable");
  return manifest.base_url.replace(/\/$/, "");
}

function recordDate(image) {
  const date = text(image.SortDate) || text(image.DateStart);
  if (date) return date;
  const year = text(image.FileName).match(/\b(20\d{2})\b/)?.[1];
  return year ? `${year}-01-01` : "0000-00-00";
}

function pickPlayerImage(candidates, currentTeam) {
  const teamKey = normalized(currentTeam);
  return [...candidates]
    .filter((image) => /\.png$/i.test(text(image.FileName)))
    .sort((a, b) => {
      const sameTeamA = normalized(a.Team) === teamKey ? 1 : 0;
      const sameTeamB = normalized(b.Team) === teamKey ? 1 : 0;
      if (sameTeamA !== sameTeamB) return sameTeamB - sameTeamA;
      const headshotA = normalized(a.ImageType) === "headshot" ? 1 : 0;
      const headshotB = normalized(b.ImageType) === "headshot" ? 1 : 0;
      if (headshotA !== headshotB) return headshotB - headshotA;
      return recordDate(b).localeCompare(recordDate(a));
    })[0] ?? null;
}

function uniqueBroadTeams(teams) {
  const index = new Map();
  for (const team of teams) {
    for (const value of [team.Name, team.OverviewPage, team.Short]) {
      const key = broadTeamKey(value);
      if (!key) continue;
      const rows = index.get(key) ?? [];
      if (!rows.includes(team)) rows.push(team);
      index.set(key, rows);
    }
  }
  return new Map([...index].filter(([, rows]) => rows.length === 1));
}

const packBase = await currentPack();
const [playerRecords, playerRatings, teamRecords, leaguePlayers, leagueTeams, teamNames, datedProfileImages, rawProfileImages] = await Promise.all([
  fetchJson(`${packBase}/features/player_records.json`),
  fetchJson(`${packBase}/features/player_ratings_snapshot.json`),
  fetchJson(`${packBase}/features/team_records.json`),
  cargoPages("players-with-country", {
    tables: "Players",
    fields: "Player,OverviewPage,Image,Team,Name,NationalityPrimary,Country",
  }),
  cargoPages("teams", {
    tables: "Teams",
    fields: "Name,OverviewPage,Short,Image,IsDisbanded",
  }),
  cargoPages("teamnames", {
    tables: "Teamnames",
    fields: "Link,Longname,Short,Medium,Inputs",
  }),
  cargoPages("profile-images", {
    tables: "PlayerImages=PI,Tournaments=T",
    fields: "PI.FileName,PI.Link,PI.Team,PI.ImageType,PI.IsProfileImage,PI.SortDate,T.DateStart",
    joinOn: "PI.Tournament=T.OverviewPage",
    where: "PI.IsProfileImage=1",
  }),
  cargoPages("player-images", {
    tables: "PlayerImages",
    fields: "FileName,Link,Team,Tournament,ImageType,IsProfileImage,SortDate",
    where: "IsProfileImage=1",
  }),
]);

const teamsByExactKey = new Map();
for (const team of leagueTeams) {
  for (const value of [team.Name, team.OverviewPage, team.Short]) {
    const key = normalized(value);
    if (!key) continue;
    const rows = teamsByExactKey.get(key) ?? [];
    if (!rows.includes(team)) rows.push(team);
    teamsByExactKey.set(key, rows);
  }
}
const teamsByBroadKey = uniqueBroadTeams(leagueTeams);
const teamLinksByAlias = new Map();
for (const row of teamNames) {
  const inputs = Array.isArray(row.Inputs) ? row.Inputs : [row.Inputs];
  for (const value of [row.Link, row.Longname, row.Short, row.Medium, ...inputs]) {
    const key = normalized(value);
    if (!key) continue;
    const links = teamLinksByAlias.get(key) ?? new Set();
    links.add(text(row.Link));
    teamLinksByAlias.set(key, links);
  }
}

const teamIdentityCandidates = {};
const unresolvedTeams = [];
for (const teamName of Object.keys(teamRecords)) {
  const alias = TEAM_ALIASES.get(teamName.toLowerCase()) ?? teamName;
  const exact = teamsByExactKey.get(normalized(alias)) ?? [];
  const aliasLinks = [...(teamLinksByAlias.get(normalized(alias)) ?? [])];
  const linked = aliasLinks.flatMap((link) => teamsByExactKey.get(normalized(link)) ?? []);
  const broad = teamsByBroadKey.get(broadTeamKey(alias)) ?? [];
  const candidates = exact.length ? exact : linked.length ? linked : broad;
  const matched = candidates
    .filter((team) => /\.png$/i.test(text(team.Image)))
    .sort((a, b) => Number(a.IsDisbanded ?? 0) - Number(b.IsDisbanded ?? 0))[0];
  if (!matched) {
    teamIdentityCandidates[normalized(teamName)] = {
      publishedName: teamName,
      source: wikiPageUrl(teamName),
      files: [`${teamName}logo square.png`, `${teamName}logo profile.png`],
    };
    continue;
  }
  teamIdentityCandidates[normalized(teamName)] = {
    publishedName: teamName,
    source: wikiPageUrl(matched.OverviewPage),
    files: [matched.Image],
  };
}

const playerRowsByHandle = new Map();
const playerRowsByTeam = new Map();
for (const player of leaguePlayers) {
  const keys = new Set([normalized(player.Player), normalized(playerHandle(player.Player))].filter(Boolean));
  for (const key of keys) {
    const rows = playerRowsByHandle.get(key) ?? [];
    rows.push(player);
    playerRowsByHandle.set(key, rows);
  }
  const teamKey = normalized(player.Team);
  if (teamKey) {
    const teamRows = playerRowsByTeam.get(teamKey) ?? [];
    teamRows.push(player);
    playerRowsByTeam.set(teamKey, teamRows);
  }
}

const imagesByLink = new Map();
const profileImagesByFile = new Map();
for (const image of [...rawProfileImages, ...datedProfileImages]) {
  if (Number(image.IsProfileImage ?? 1) !== 1) continue;
  const fileKey = normalized(image.FileName);
  const existing = profileImagesByFile.get(fileKey);
  if (!existing || recordDate(image) > recordDate(existing)) profileImagesByFile.set(fileKey, image);
}
for (const image of profileImagesByFile.values()) {
  const key = normalized(image.Link);
  if (!key) continue;
  const rows = imagesByLink.get(key) ?? [];
  rows.push(image);
  imagesByLink.set(key, rows);
}

const playerIdentityCandidates = {};
const unresolvedPlayers = [];
for (const [playerName, record] of Object.entries(playerRecords)) {
  const currentTeam = text(record.current_team);
  const playerKey = normalized(playerName);
  let playerRows = playerRowsByHandle.get(playerKey) ?? [];
  if (!playerRows.length && currentTeam) {
    const teamRows = playerRowsByTeam.get(normalized(currentTeam)) ?? [];
    const suffixRows = teamRows
      .filter((row) => {
        const handle = normalized(playerHandle(row.Player));
        return handle.length >= 3 && (playerKey.endsWith(handle) || handle.endsWith(playerKey));
      })
      .sort((a, b) => normalized(b.Player).length - normalized(a.Player).length);
    if (suffixRows.length) {
      const longest = normalized(suffixRows[0].Player).length;
      playerRows = suffixRows.filter((row) => normalized(row.Player).length === longest);
    }
  }
  if (!playerRows.length) {
    const suffixRows = leaguePlayers
      .filter((row) => {
        const handle = normalized(playerHandle(row.Player));
        return handle.length >= 4 && playerKey.endsWith(handle);
      })
      .sort((a, b) => normalized(playerHandle(b.Player)).length - normalized(playerHandle(a.Player)).length);
    if (suffixRows.length) {
      const longest = normalized(playerHandle(suffixRows[0].Player)).length;
      const longestRows = suffixRows.filter((row) => normalized(playerHandle(row.Player)).length === longest);
      const overviewPages = new Set(longestRows.map((row) => normalized(row.OverviewPage)));
      if (overviewPages.size === 1) playerRows = longestRows;
    }
  }
  const sameTeamRows = playerRows.filter((row) => (
    normalized(row.Team) === normalized(currentTeam)
    || (broadTeamKey(row.Team) && broadTeamKey(row.Team) === broadTeamKey(currentTeam))
  ));
  const distinctPages = new Set(playerRows.map((row) => normalized(row.OverviewPage)).filter(Boolean));
  const identityRows = sameTeamRows.length
    ? sameTeamRows
    : distinctPages.size <= 1
      ? playerRows
      : [];
  const candidateIdentityRows = identityRows.length ? identityRows : playerRows;
  const links = new Set(candidateIdentityRows.map((row) => normalized(row.OverviewPage)).filter(Boolean));
  if (!links.size) links.add(normalized(playerName));
  const allCandidates = [...links].flatMap((link) => imagesByLink.get(link) ?? []);
  const teamCandidates = allCandidates.filter((image) => (
    normalized(image.Team) === normalized(currentTeam)
    || (broadTeamKey(image.Team) && broadTeamKey(image.Team) === broadTeamKey(currentTeam))
  ));
  const candidates = identityRows.length || distinctPages.size <= 1 ? allCandidates : teamCandidates;
  let picked = pickPlayerImage(candidates, currentTeam);
  let directImage = "";
  if (!picked) {
    directImage = text(identityRows.find((row) => /\.png$/i.test(text(row.Image)))?.Image);
  }
  const override = PLAYER_IMAGE_OVERRIDES.get(`${normalized(playerName)}|${normalized(currentTeam)}`);
  if (!picked && !directImage && !override) {
    unresolvedPlayers.push(`${playerName}|${currentTeam}`);
    continue;
  }
  const sourcePage = override?.page || (picked
    ? candidateIdentityRows.find((row) => normalized(row.OverviewPage) === normalized(picked.Link))?.OverviewPage || picked.Link
    : identityRows[0]?.OverviewPage || playerName);
  const fileName = override?.file || picked?.FileName || directImage;
  playerIdentityCandidates[`${normalized(playerName)}|${normalized(currentTeam)}`] = {
    publishedName: `${playerName}|${currentTeam}`,
    source: wikiPageUrl(sourcePage),
    file: fileName,
  };
}

const imageUrls = await resolveWikiImageUrls([
  ...Object.values(teamIdentityCandidates).flatMap((identity) => identity.files),
  ...Object.values(playerIdentityCandidates).map((identity) => identity.file),
]);
const teamIdentities = {};
for (const [key, identity] of Object.entries(teamIdentityCandidates)) {
  const file = identity.files.find((candidate) => imageUrls.has(normalized(candidate)));
  const src = file ? imageUrls.get(normalized(file)) : null;
  if (!src) {
    unresolvedTeams.push(identity.publishedName);
    continue;
  }
  teamIdentities[key] = { src, source: identity.source, file };
}
const playerIdentities = {};
for (const [key, identity] of Object.entries(playerIdentityCandidates)) {
  const src = imageUrls.get(normalized(identity.file));
  if (!src) {
    unresolvedPlayers.push(identity.publishedName);
    continue;
  }
  playerIdentities[key] = { src, source: identity.source, file: identity.file };
}

const generatedAt = new Date().toISOString();
const source = "Leaguepedia Cargo exports";

function tierCoverage(entries, tierForName, isResolved) {
  const rows = {};
  for (const name of entries) {
    const tier = text(tierForName(name)) || "unknown";
    const item = rows[tier] ?? { published: 0, resolved: 0 };
    item.published += 1;
    if (isResolved(name)) item.resolved += 1;
    rows[tier] = item;
  }
  return Object.fromEntries(Object.entries(rows).sort(([a], [b]) => a.localeCompare(b)));
}

const ratedPlayerNames = playerRatings
  .filter((row) => Number(row.n_maps ?? 0) >= 5)
  .map((row) => text(row.player))
  .filter((name) => name && playerRecords[name]);
const teamCoverageByTier = tierCoverage(
  Object.keys(teamRecords),
  (name) => teamRecords[name]?.current_tier,
  (name) => Boolean(teamIdentities[normalized(name)]),
);
const playerCoverageByTier = tierCoverage(
  ratedPlayerNames,
  (name) => playerRecords[name]?.current_tier,
  (name) => Boolean(playerIdentities[`${normalized(name)}|${normalized(playerRecords[name]?.current_team)}`]),
);
const teamPayload = {
  generated_at: generatedAt,
  source,
  identities: teamIdentities,
};
const playerPayload = {
  generated_at: generatedAt,
  source,
  identities: playerIdentities,
};
const report = {
  generated_at: generatedAt,
  pack_base_url: packBase,
  teams: {
    published: Object.keys(teamRecords).length,
    resolved: Object.keys(teamIdentities).length,
    visual_fallbacks: Object.keys(teamRecords).length - Object.keys(teamIdentities).length,
    by_tier: teamCoverageByTier,
    unresolved: unresolvedTeams,
  },
  players: {
    published: Object.keys(playerRecords).length,
    resolved: Object.keys(playerIdentities).length,
    visual_fallbacks: Object.keys(playerRecords).length - Object.keys(playerIdentities).length,
    rated_by_tier: playerCoverageByTier,
    unresolved: unresolvedPlayers,
  },
};

await fs.mkdir(path.dirname(teamOutputPath), { recursive: true });
await fs.writeFile(teamOutputPath, `${JSON.stringify(teamPayload)}\n`, "utf8");
await fs.writeFile(playerOutputPath, `${JSON.stringify(playerPayload)}\n`, "utf8");
await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

console.log(JSON.stringify({
  teams: `${report.teams.resolved}/${report.teams.published}`,
  players: `${report.players.resolved}/${report.players.published}`,
  teams_output: path.relative(appRoot, teamOutputPath),
  players_output: path.relative(appRoot, playerOutputPath),
  report: path.relative(appRoot, reportPath),
}, null, 2));
