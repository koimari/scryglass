import { promises as fs } from "node:fs";
import https from "node:https";
import path from "node:path";
import type { PackManifest } from "./pack";

const packJsonCache = new Map<string, Promise<unknown>>();
const PACK_CACHE_SECONDS = 21_600;
export const PACK_MANIFEST_CACHE_TAG = "scryglass-pack-manifest";
const DEFAULT_BLOB_ROOT = "https://97gks2fobqkgppwx.public.blob.vercel-storage.com";
const MAX_STORAGE_ASSET_BYTES = 50 * 1024 * 1024;

type SupabaseConfig = {
  url: string;
  publishableKey: string;
};

function supabaseConfig(): SupabaseConfig | null {
  const url = (
    process.env.SCRYGLASS_SUPABASE_URL
    || process.env.NEXT_PUBLIC_SUPABASE_URL
    || ""
  ).trim().replace(/\/$/, "");
  const publishableKey = (
    process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY
    || process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY
    || ""
  ).trim();
  if (!url && !publishableKey) return null;
  if (!/^https:\/\/[a-z0-9]+\.supabase\.co$/.test(url) || !publishableKey.startsWith("sb_publishable_")) {
    throw new Error("Supabase public data configuration is incomplete");
  }
  return { url, publishableKey };
}

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

function readStorageJson<T>(url: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = https.get(url, { timeout: 60_000 }, (response) => {
      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`Supabase Storage asset ${response.statusCode ?? "unavailable"}`));
        return;
      }
      const chunks: Buffer[] = [];
      let bytes = 0;
      response.on("data", (chunk: Buffer) => {
        bytes += chunk.length;
        if (bytes > MAX_STORAGE_ASSET_BYTES) {
          response.destroy(new Error("Supabase Storage asset is too large"));
          return;
        }
        chunks.push(chunk);
      });
      response.on("end", () => {
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")) as T);
        } catch {
          reject(new Error("Supabase Storage asset is invalid JSON"));
        }
      });
      response.on("error", reject);
    });
    request.on("timeout", () => request.destroy(new Error("Supabase Storage request timed out")));
    request.on("error", reject);
  });
}

async function readSupabaseManifest(cache: "no-store" | "cached"): Promise<PackManifest> {
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const response = await fetch(
    `${config.url}/rest/v1/scryglass_public_releases?status=eq.active&select=manifest&limit=1`,
    {
      headers: { apikey: config.publishableKey },
      ...(cache === "no-store"
        ? { cache: "no-store" as const }
        : { next: { revalidate: PACK_CACHE_SECONDS, tags: [PACK_MANIFEST_CACHE_TAG] } }),
    },
  );
  if (!response.ok) throw new Error(`Supabase release ${response.status}`);
  const rows = (await response.json()) as Array<{ manifest?: PackManifest }>;
  const manifest = rows[0]?.manifest;
  if (!manifest?.pack_id || manifest.data_backend !== "supabase") {
    throw new Error("Supabase release is unavailable");
  }
  return manifest;
}

async function readBlobManifest(cache: "no-store" | "cached"): Promise<PackManifest> {
  const response = await fetch(manifestUrl(), {
    ...(cache === "no-store"
      ? { cache: "no-store" as const }
      : { next: { revalidate: PACK_CACHE_SECONDS, tags: [PACK_MANIFEST_CACHE_TAG] } }),
  });
  if (!response.ok) throw new Error(`pack manifest ${response.status}`);
  const manifest = (await response.json()) as PackManifest;
  if (!manifest.pack_id || !/^https:\/\//.test(manifest.base_url || "")) {
    throw new Error("pack manifest is malformed");
  }
  return manifest;
}

/** Read the remote pointer without falling back to the bundled outage copy. */
export async function readRemotePackManifest(): Promise<PackManifest> {
  return supabaseConfig()
    ? readSupabaseManifest("no-store")
    : readBlobManifest("no-store");
}

/** Read the current Blob pointer. The bundled pointer is an outage fallback. */
export async function readPackManifest(): Promise<PackManifest> {
  try {
    if (supabaseConfig()) return await readSupabaseManifest("cached");
    return await readBlobManifest("cached");
  } catch {
    try {
      return await readBlobManifest("cached");
    } catch {
      return readLocalManifest();
    }
  }
}

async function readSupabaseAsset<T>(releaseId: string, relativePath: string): Promise<T> {
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const release = encodeURIComponent(releaseId);
  const assetPath = encodeURIComponent(relativePath);
  const response = await fetch(
    `${config.url}/rest/v1/scryglass_public_assets?release_id=eq.${release}&path=eq.${assetPath}&select=body,storage_path&limit=1`,
    {
      headers: { apikey: config.publishableKey },
      cache: "force-cache",
    },
  );
  if (!response.ok) throw new Error(`Supabase public asset ${response.status}`);
  const rows = (await response.json()) as Array<{ body?: T | null; storage_path?: string | null }>;
  const row = rows[0];
  if (!row) throw new Error("Supabase public asset is missing");
  if (row.body !== null && row.body !== undefined) return row.body;
  if (!row.storage_path) throw new Error("Supabase public asset has no payload");
  const storagePath = row.storage_path
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
  return readStorageJson<T>(
    `${config.url}/storage/v1/object/public/scryglass-public/${storagePath}`,
  );
}

/** Load immutable pack JSON from Blob. Local files support development and outages. */
export async function readPackJson<T>(manifest: PackManifest, relativePath: string): Promise<T> {
  const clean = safeRelativePath(relativePath);
  const baseUrl = (manifest.base_url || "").replace(/\/$/, "");
  const remoteUrl = /^https:\/\//.test(baseUrl) ? `${baseUrl}/${clean}` : "";
  const localPath = path.join(localPackRoot(), manifest.pack_id, clean);
  const supabase = manifest.data_backend === "supabase" && supabaseConfig();
  const cacheKey = supabase
    ? `supabase:${manifest.pack_id}:${clean}`
    : remoteUrl || localPath;
  let pending = packJsonCache.get(cacheKey);
  if (!pending) {
    pending = supabase
      ? readSupabaseAsset(manifest.pack_id, clean)
      : remoteUrl
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

export async function readPublicTierList<T>(): Promise<T> {
  if (supabaseConfig()) {
    const manifest = await readSupabaseManifest("no-store");
    return readSupabaseAsset<T>(manifest.pack_id, "rankings/tierlists.json");
  }
  const configured = process.env.SCRYGLASS_TIERLIST_DISPLAY_URL?.trim();
  const blobRoot = process.env.LIVE_BLOB_BASE_URL?.trim() || DEFAULT_BLOB_ROOT;
  const url = configured || `${blobRoot.replace(/\/$/, "")}/rankings/tierlists.json`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`tier list ${response.status}`);
  return response.json() as Promise<T>;
}

/** Send the large tier artifact from storage instead of proxying it through Vercel. */
export async function publicTierListDownloadUrl(): Promise<string> {
  return publicTierListViewDownloadUrl("full");
}

export async function publicTierListViewDownloadUrl(view: "full" | "latest"): Promise<string> {
  const relativePath = view === "latest"
    ? "rankings/tierlists-latest.json"
    : "rankings/tierlists.json";
  const config = supabaseConfig();
  if (config) {
    const manifest = await readSupabaseManifest("no-store");
    const release = encodeURIComponent(manifest.pack_id);
    const assetPath = encodeURIComponent(relativePath);
    const response = await fetch(
      `${config.url}/rest/v1/scryglass_public_assets?release_id=eq.${release}&path=eq.${assetPath}&select=storage_path&limit=1`,
      { headers: { apikey: config.publishableKey }, cache: "no-store" },
    );
    if (!response.ok) throw new Error(`Supabase public tier asset ${response.status}`);
    const rows = (await response.json()) as Array<{ storage_path?: string | null }>;
    const stored = rows[0]?.storage_path;
    if (!stored) throw new Error("Supabase public tier asset has no storage path");
    const storagePath = stored
      .split("/")
      .map((part) => encodeURIComponent(part))
      .join("/");
    return `${config.url}/storage/v1/object/public/scryglass-public/${storagePath}`;
  }
  const configured = process.env.SCRYGLASS_TIERLIST_DISPLAY_URL?.trim();
  const blobRoot = process.env.LIVE_BLOB_BASE_URL?.trim() || DEFAULT_BLOB_ROOT;
  const url = configured
    ? view === "latest"
      ? configured.replace(/tierlists\.json$/, "tierlists-latest.json")
      : configured
    : `${blobRoot.replace(/\/$/, "")}/${relativePath}`;
  if (!/^https:\/\//.test(url)) throw new Error("tier-list download URL is invalid");
  return url;
}
