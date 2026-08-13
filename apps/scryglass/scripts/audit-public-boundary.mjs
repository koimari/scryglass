import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

const appRoot = process.cwd();
// PUBLIC_ASSET_ALLOWLIST_V1. A cross-language contract test keeps this set in
// step with the publisher, web runtime, and final database constraint.
const allowedAssetPaths = new Set([
  "features/ratings_snapshot.json",
  "features/player_ratings_snapshot.json",
  "features/team_records.json",
  "features/team_weekly_ranks.json",
  "features/player_records.json",
  "features/player_champion_records.json",
  "features/profile_records.json",
  "features/match_index.json",
  "features/match_records_2025.json",
  "features/match_records_2026.json",
  "features/player_weekly_ranks.json",
  "features/player_metadata.json",
  "features/schedule.json",
  "features/leaderboards.json",
  "features/draft_records.json",
  "features/match_records_2025_q1.json",
  "features/match_records_2025_q2.json",
  "features/match_records_2025_q3.json",
  "features/match_records_2025_q4.json",
  "features/match_records_2026_q1.json",
  "features/match_records_2026_q2.json",
  "features/match_records_2026_q3.json",
  "features/match_records_2026_q4.json",
  "rankings/tierlists.json",
  "rankings/tierlists-latest.json",
]);
const forbiddenPaths = [
  "src/app/articles",
  "src/app/browse",
  "src/app/grubs",
  "src/app/live",
  "src/app/reproduce",
  "src/app/sandbox",
  "src/app/api/draft-sandbox",
  "src/app/api/draft-wr",
  "src/app/api/v2/draft",
  "src/app/api/v2/tierlist",
  "src/app/api/data-upload/route.ts",
  "src/components/DraftSandbox.tsx",
  "src/components/DraftWrPanel.tsx",
  "src/components/HeadToHead.tsx",
  "src/components/LiveBoard.tsx",
  "src/components/MatchLoader.tsx",
  "src/lib/draftScore.ts",
  "src/lib/draftTerminalScore.ts",
  "src/lib/duck.ts",
  "src/lib/live.ts",
  "src/lib/publicDraftGate.ts",
  "src/lib/tierlistServer.ts",
  "scripts/publish-data.mjs",
  "data/draft",
  "api/cron",
  "public/v2/tierlists",
  "public/live",
  "public/packs/manifest.json",
  "public/rankings/tierlists.json",
];
const requiredDynamicPaths = [
  "src/app/packs/manifest.json/route.ts",
  "src/app/rankings/tierlists.json/route.ts",
  "src/app/rankings/tierlists-latest.json/route.ts",
];
const forbiddenArtifactText = [
  "BLOB_READ_WRITE_TOKEN",
  "GRID_API_KEY",
  "SCRYGLASS_SUPABASE_SERVICE_ROLE_KEY",
  "service_role_key",
  "data/lol/warehouse/raw_grid",
  "data/lol/v2/",
  "private source receipts",
  '"training_rows"',
  '"raw_sha256"',
  "public.blob.vercel-storage.com",
];
const forbiddenPublicDataText = [
  "oracle_elixir_api",
  "public_datalisk_api",
  '"last_run_id"',
  '"worker_commit"',
  '"service_role"',
];
const retiredPublicationText = [
  "@vercel/blob",
  "BLOB_READ_WRITE_TOKEN",
  "LIVE_BLOB_BASE_URL",
  "SCRYGLASS_TIERLIST_BLOB_BASE_URL",
];
const textExtensions = new Set([".css", ".html", ".js", ".json", ".map", ".mjs", ".rsc", ".txt"]);

async function exists(relative) {
  try {
    await stat(path.join(appRoot, relative));
    return true;
  } catch {
    return false;
  }
}

async function filesUnder(relative) {
  const root = path.join(appRoot, relative);
  if (!(await exists(relative))) return [];
  const files = [];
  async function visit(current) {
    for (const entry of await readdir(current, { withFileTypes: true })) {
      const target = path.join(current, entry.name);
      if (entry.isDirectory()) await visit(target);
      else if (entry.isFile()) files.push(target);
    }
  }
  await visit(root);
  return files;
}

async function findText(root, values, failures) {
  for (const file of await filesUnder(root)) {
    if (!textExtensions.has(path.extname(file))) continue;
    const content = await readFile(file, "utf8").catch(() => "");
    for (const value of values) {
      if (content.includes(value)) failures.push(`${value} appears in ${path.relative(appRoot, file)}`);
    }
  }
}

const failures = [];
for (const relative of forbiddenPaths) {
  if (await exists(relative)) failures.push(`retired public path exists: ${relative}`);
}
for (const relative of requiredDynamicPaths) {
  if (!(await exists(relative))) failures.push(`active-release compatibility route is missing: ${relative}`);
}

const publicPackFiles = await filesUnder("public/packs");
if (publicPackFiles.length) {
  failures.push(`checked-in public pack assets remain: ${publicPackFiles.map((file) => path.relative(appRoot, file)).join(", ")}`);
}
await findText("public", [...forbiddenArtifactText, ...forbiddenPublicDataText], failures);

// A clean production build has BUILD_ID. Ignore `.next/dev`, which can contain
// an older local server and is not a deployable artifact.
if (await exists(".next/BUILD_ID")) {
  await findText(".next/static", forbiddenArtifactText, failures);
  await findText(".next/server/app", forbiddenArtifactText, failures);
}

for (const relative of ["package.json", "package-lock.json"]) {
  const content = await readFile(path.join(appRoot, relative), "utf8");
  for (const value of retiredPublicationText) {
    if (content.includes(value)) failures.push(`retired publication dependency appears in ${relative}: ${value}`);
  }
}

for (const relative of ["next.config.ts", "src/proxy.ts", "src/lib/pack.ts", "src/lib/serverPack.ts"]) {
  const content = await readFile(path.join(appRoot, relative), "utf8");
  if (content.includes("public.blob.vercel-storage.com")) {
    failures.push(`retired public Blob origin appears in ${relative}`);
  }
}

const bundledManifestPath = "src/lib/bundledPackManifest.json";
if (await exists(bundledManifestPath)) {
  failures.push(`stale build-only release manifest exists: ${bundledManifestPath}`);
}

const draftRoute = await readFile(path.join(appRoot, "src/app/api/chat/query_drafts/route.ts"), "utf8").catch(() => "");
if (!draftRoute.match(/authority:\s*["']unavailable["']/) || draftRoute.includes("draft_records")) {
  failures.push("draft chat route does not fail closed");
}

if (failures.length) {
  process.stderr.write(`${[...new Set(failures)].join("\n")}\n`);
  process.exit(1);
}

process.stdout.write("Public release boundary: clean\n");
