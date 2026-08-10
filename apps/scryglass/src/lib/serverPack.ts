import { promises as fs } from "node:fs";
import path from "node:path";
import type { PackManifest } from "./pack";

const packJsonCache = new Map<string, Promise<unknown>>();
const PACK_CACHE_SECONDS = 21_600;
export const PACK_MANIFEST_CACHE_TAG = "scryglass-pack-manifest";
const DEFAULT_BLOB_ROOT = "https://97gks2fobqkgppwx.public.blob.vercel-storage.com";

function localPackRoot(): string {
  const configured = process.env.SCRYGLASS_PACK_ROOT?.trim();
  return configured ? path.resolve(configured) : path.join(process.cwd(), "public", "packs");
}

function manifestUrl(): string {
  const configured = process.env.SCRYGLASS_PACK_MANIFEST_URL?.trim();
  if (configured) return configured;
  const blobRoot = process.env.LIVE_BLOB_BASE_URL?.trim() || DEFAULT_BLOB_ROOT;
  return `${blobRoot.replace(/\/$/, "")}/packs/manifest.json`;
}

function safeRelativePath(relativePath: string): string {
  const clean = relativePath.replace(/^\/+/, "");
  if (!clean || clean.split("/").includes("..")) {
    throw new Error("pack path is invalid");
  }
  return clean;
}

async function readLocalManifest(): Promise<PackManifest> {
  return JSON.parse(
    await fs.readFile(path.join(localPackRoot(), "manifest.json"), "utf8"),
  ) as PackManifest;
}

/** Read the current Blob pointer. The bundled pointer is an outage fallback. */
export async function readPackManifest(): Promise<PackManifest> {
  try {
    const response = await fetch(manifestUrl(), {
      next: { revalidate: PACK_CACHE_SECONDS, tags: [PACK_MANIFEST_CACHE_TAG] },
    });
    if (!response.ok) throw new Error(`pack manifest ${response.status}`);
    const manifest = (await response.json()) as PackManifest;
    if (!manifest.pack_id || !/^https:\/\//.test(manifest.base_url || "")) {
      throw new Error("pack manifest is malformed");
    }
    return manifest;
  } catch {
    return readLocalManifest();
  }
}

/** Load immutable pack JSON from Blob. Local files support development and outages. */
export async function readPackJson<T>(manifest: PackManifest, relativePath: string): Promise<T> {
  const clean = safeRelativePath(relativePath);
  const baseUrl = (manifest.base_url || "").replace(/\/$/, "");
  const remoteUrl = /^https:\/\//.test(baseUrl) ? `${baseUrl}/${clean}` : "";
  const localPath = path.join(localPackRoot(), manifest.pack_id, clean);
  const cacheKey = remoteUrl || localPath;
  let pending = packJsonCache.get(cacheKey);
  if (!pending) {
    pending = remoteUrl
      ? fetch(remoteUrl, { cache: "force-cache" }).then(async (response) => {
          if (!response.ok) throw new Error(`pack file ${response.status}`);
          return response.json() as Promise<unknown>;
        })
      : fs.readFile(localPath, "utf8").then((raw) => JSON.parse(raw) as unknown);
    packJsonCache.set(cacheKey, pending);
    pending.catch(() => packJsonCache.delete(cacheKey));
  }
  return pending as Promise<T>;
}
