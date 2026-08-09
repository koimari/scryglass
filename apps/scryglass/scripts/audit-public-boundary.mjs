import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

const appRoot = process.cwd();
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
  "data/draft",
  "api/cron",
  "public/v2/tierlists",
];
const forbiddenText = [
  "draft_score",
  "draft-wr",
  "draft_recommendation",
  "draft_composition",
  "composition_runtime",
  "Draft Score",
];
const forbiddenPublicPath = /(?:draft|composition|blade_chest|tierlists_csv)/i;
const excludedPublicRatingText = ["los ratones"];
const invalidPublicLeagueText = ["oracle_elixir_api", "public_datalisk_api"];
const forbiddenTierText = [
  "artifact_sha256",
  "claim_ceiling",
  "development_only",
  "publication_eligible",
  "raw_sha256",
  "training",
];
const allowedPackFile = /^public\/packs\/(?:manifest\.json|[^/]+\/features\/(?:ratings_snapshot|player_ratings_snapshot|team_records|team_weekly_ranks|player_records|player_weekly_ranks|player_metadata)\.json)$/;
const textExtensions = new Set([
  ".css",
  ".html",
  ".js",
  ".json",
  ".map",
  ".md",
  ".mjs",
  ".rsc",
  ".tex",
  ".txt",
]);

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
    const entries = await readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const target = path.join(current, entry.name);
      if (entry.isDirectory()) await visit(target);
      else if (entry.isFile()) files.push(target);
    }
  }
  await visit(root);
  return files;
}

const failures = [];
for (const relative of forbiddenPaths) {
  if (await exists(relative)) failures.push(`removed public path exists: ${relative}`);
}

for (const file of await filesUnder("public")) {
  const relative = path.relative(appRoot, file);
  if (forbiddenPublicPath.test(relative)) {
    failures.push(`private model path is public: ${relative}`);
  }
  if (relative.startsWith("public/packs/") && !allowedPackFile.test(relative)) {
    failures.push(`non-rating pack file is public: ${relative}`);
  }
  if (relative.startsWith("public/packs/") && path.extname(file) === ".json") {
    const content = await readFile(file, "utf8").catch(() => "");
    for (const value of excludedPublicRatingText) {
      if (content.toLowerCase().includes(value)) {
        failures.push(`excluded team appears in ${relative}`);
      }
    }
    for (const value of invalidPublicLeagueText) {
      if (content.toLowerCase().includes(value)) {
        failures.push(`transport label appears as public data in ${relative}`);
      }
    }
  }
}

for (const root of ["public", ".next"]) {
  for (const file of await filesUnder(root)) {
    if (!textExtensions.has(path.extname(file))) continue;
    const content = await readFile(file, "utf8").catch(() => "");
    for (const value of forbiddenText) {
      if (content.includes(value)) {
        failures.push(`${value} appears in ${path.relative(appRoot, file)}`);
      }
    }
  }
}

const tierFile = "public/rankings/tierlists.json";
if (!(await exists(tierFile))) {
  failures.push(`missing static tier-list file: ${tierFile}`);
} else {
  const content = await readFile(path.join(appRoot, tierFile), "utf8");
  for (const value of forbiddenTierText) {
    if (content.includes(value)) failures.push(`${value} appears in ${tierFile}`);
  }
}

if (failures.length) {
  process.stderr.write(`${failures.join("\n")}\n`);
  process.exit(1);
}

process.stdout.write("Public rankings boundary: clean\n");
