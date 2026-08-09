import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import os from "node:os";
import path from "node:path";

const execFileAsync = promisify(execFile);
const appRoot = process.cwd();
const repoRoot = path.resolve(appRoot, "../..");
const bundleRoot = path.join(appRoot, ".tier-worker");
const remoteBundleRoot = path.join(appRoot, ".tier-worker-remote");

async function copyFrom(sourceRoot, relative, { required = true } = {}) {
  const source = path.join(sourceRoot, relative);
  const destination = path.join(bundleRoot, relative);
  await mkdir(path.dirname(destination), { recursive: true });
  try {
    await cp(source, destination, { recursive: true });
  } catch (error) {
    if (!required && error?.code === "ENOENT") return false;
    throw error;
  }
  return true;
}

async function copy(relative, options) {
  return copyFrom(repoRoot, relative, options);
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

function blobUploadUrl(bundlePath) {
  if (!/^[-A-Za-z0-9._/]+$/.test(bundlePath) || bundlePath.startsWith("/")) {
    throw new Error("SCRYGLASS_TIER_WORKER_RUNTIME_BUNDLE_PATH is invalid");
  }
  const encodedPath = bundlePath
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
  return `https://blob.vercel-storage.com/${encodedPath}`;
}

async function publishRuntimeBundle() {
  const token = process.env.TIER_WORKER_READ_WRITE_TOKEN;
  if (!token || !process.env.VERCEL) return;
  const bundlePath =
    process.env.SCRYGLASS_TIER_WORKER_RUNTIME_BUNDLE_PATH ||
    "tier-worker/tier-worker-runtime-current.tar.gz";
  const url = blobUploadUrl(bundlePath);
  const storeId = token.slice("vercel_blob_rw_".length).split("_", 1)[0].toLowerCase();
  const archive = path.join(
    os.tmpdir(),
    `scryglass-tier-worker-runtime-${process.pid}.tar.gz`,
  );
  try {
    await execFileAsync("tar", ["-czf", archive, "-C", bundleRoot, "."]);
    const response = await fetch(url, {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${token}`,
        "X-API-Version": "7",
        "x-vercel-blob-store-id": storeId,
        "x-vercel-blob-access": "private",
        "x-add-random-suffix": "0",
        "x-allow-overwrite": "1",
        "Content-Type": "application/gzip",
        "x-content-type": "application/gzip",
      },
      body: await readFile(archive),
    });
    if (!response.ok) {
      throw new Error(
        `Tier worker runtime bundle upload failed: HTTP ${response.status}`,
      );
    }
    process.stdout.write(`[tier-worker] published private runtime bundle ${bundlePath}\n`);
  } finally {
    await rm(archive, { force: true });
  }
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
  const archive = path.join(os.tmpdir(), `scryglass-tier-worker-${process.pid}.tar.gz`);
  await writeFile(archive, Buffer.from(await response.arrayBuffer()));
  await rm(remoteBundleRoot, { recursive: true, force: true });
  await mkdir(remoteBundleRoot, { recursive: true });
  await execFileAsync("tar", ["-xzf", archive, "-C", remoteBundleRoot]);
  await rm(archive, { force: true });
  return remoteBundleRoot;
}

async function copyBaselinePack(sourceRoot) {
  const latestPath = path.join(sourceRoot, "apps/scryglass/public/packs/latest.json");
  const latest = JSON.parse(await readFile(latestPath, "utf8"));
  if (typeof latest.pack_id !== "string" || !/^[A-Za-z0-9._-]+$/.test(latest.pack_id)) {
    throw new Error("The tier worker pack id is invalid");
  }
  const packRoot = `apps/scryglass/public/packs/${latest.pack_id}`;
  for (const relative of [
    "apps/scryglass/public/packs/latest.json",
    "apps/scryglass/public/packs/manifest.json",
    `${packRoot}/manifest.json`,
    `${packRoot}/player_games`,
    `${packRoot}/team_games`,
    `${packRoot}/models`,
    `${packRoot}/studies`,
  ]) {
    await copyFrom(sourceRoot, relative);
  }
}

await rm(bundleRoot, { recursive: true, force: true });
await mkdir(bundleRoot, { recursive: true });

const remoteBundle = await unpackRemoteBundle();
for (const relative of [
  "lol_kills",
  "data/lol/v2/champions",
  "data/lol/v2/models/draft-terminal",
]) {
  await copy(relative);
}

if (remoteBundle) {
  await copyBaselinePack(remoteBundle);
  process.stdout.write("[tier-worker] unpacked private Vercel Blob bundle\n");
} else {
  await copyBaselinePack(repoRoot);
}

try {
  await publishRuntimeBundle();
} finally {
  if (remoteBundle) {
    await rm(remoteBundleRoot, { recursive: true, force: true });
  }
}
