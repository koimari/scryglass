import { cp, mkdir, readFile, rm } from "node:fs/promises";
import path from "node:path";

const appRoot = process.cwd();
const repoRoot = path.resolve(appRoot, "../..");
const bundleRoot = path.join(appRoot, ".tier-worker");

await rm(bundleRoot, { recursive: true, force: true });

async function copy(relative) {
  const source = path.join(repoRoot, relative);
  const destination = path.join(bundleRoot, relative);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true });
}

for (const relative of [
  "lol_kills",
  "data/lol/v2/champions",
  "data/lol/v2/tierlists",
  "data/lol/features",
  "data/lol/models",
  "output/pdf",
]) {
  await copy(relative);
}

const latestPath = path.join(repoRoot, "apps/scryglass/public/packs/latest.json");
const latest = JSON.parse(await readFile(latestPath, "utf8"));
if (typeof latest.pack_id !== "string" || !/^[A-Za-z0-9._-]+$/.test(latest.pack_id)) {
  throw new Error("The committed public pack id is invalid");
}

for (const relative of [
  "apps/scryglass/public/packs/latest.json",
  "apps/scryglass/public/packs/manifest.json",
  `apps/scryglass/public/packs/${latest.pack_id}`,
]) {
  await copy(relative);
}
