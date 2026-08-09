import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import os from "node:os";
import path from "node:path";

const execFileAsync = promisify(execFile);
const appRoot = process.cwd();
const repoRoot = path.resolve(appRoot, "../..");
const bundleRoot = path.join(appRoot, ".tier-worker");

async function copy(relative) {
  const source = path.join(repoRoot, relative);
  const destination = path.join(bundleRoot, relative);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true });
}

function privateBlobUrl(bundlePath, token) {
  if (!bundlePath || !token) return null;
  if (!/^[-A-Za-z0-9._/]+$/.test(bundlePath) || bundlePath.startsWith("/")) {
    throw new Error("SCRYGLASS_TIER_WORKER_BUNDLE_PATH is invalid");
  }
  const prefix = "vercel_blob_rw_";
  if (!token.startsWith(prefix)) {
    throw new Error("TIER_WORKER_READ_WRITE_TOKEN is invalid");
  }
  const storeId = token.slice(prefix.length).split("_", 1)[0].toLowerCase();
  if (!storeId) throw new Error("TIER_WORKER_READ_WRITE_TOKEN has no store id");
  const encodedPath = bundlePath
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
  return `https://${storeId}.private.blob.vercel-storage.com/${encodedPath}`;
}

async function unpackRemoteBundle() {
  const bundlePath = process.env.SCRYGLASS_TIER_WORKER_BUNDLE_PATH;
  const token = process.env.TIER_WORKER_READ_WRITE_TOKEN;
  const url = privateBlobUrl(bundlePath, token);
  if (!url) return false;

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error(
      `Tier worker bundle download failed: HTTP ${response.status}`,
    );
  }
  const archive = path.join(
    os.tmpdir(),
    `scryglass-tier-worker-${process.pid}.tar.gz`,
  );
  await writeFile(archive, Buffer.from(await response.arrayBuffer()));
  await mkdir(bundleRoot, { recursive: true });
  await execFileAsync("tar", ["-xzf", archive, "-C", bundleRoot]);
  await rm(archive, { force: true });
  return true;
}

await rm(bundleRoot, { recursive: true, force: true });

const remoteBundle = await unpackRemoteBundle();
for (const relative of [
  "lol_kills",
  "data/lol/v2/champions",
  "data/lol/v2/tierlists",
]) {
  await copy(relative);
}

if (remoteBundle) {
  process.stdout.write("[tier-worker] unpacked private Vercel Blob bundle\n");
  process.exit(0);
}

for (const relative of [
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
