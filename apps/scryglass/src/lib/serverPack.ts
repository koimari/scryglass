import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import type { PackManifest } from "./pack";
import { hasPromotedDraftAuthority } from "./pack";

const packJsonCache = new Map<string, Promise<unknown>>();
const PACK_CACHE_SECONDS = 21_600;
export const PACK_MANIFEST_CACHE_TAG = "scryglass-pack-manifest";
export const MAX_STORAGE_ASSET_BYTES = 120 * 1024 * 1024;
export const PUBLIC_ASSET_CONTENT_TYPE = "application/json";

export const PUBLIC_ASSET_PATHS = new Set([
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
  "rankings/tierlists.json",
  "rankings/tierlists-latest.json",
]);

const RELEASE_ID = /^v\d{4}\.\d{2}\.\d{2}\.\d{6}$/;
const SHA256 = /^[a-f0-9]{64}$/;

type SupabaseConfig = {
  url: string;
  publishableKey: string;
};

export function supabaseConfig(): SupabaseConfig | null {
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

function bundledDataAllowed(): boolean {
  return process.env.NODE_ENV !== "production"
    || process.env.NEXT_PHASE === "phase-production-build";
}

export function safeRelativePath(relativePath: string): string {
  const clean = relativePath.replace(/^\/+/, "");
  if (
    !clean
    || clean.length > 160
    || clean.split("/").some((part) => !part || part === "." || part === "..")
    || !PUBLIC_ASSET_PATHS.has(clean)
  ) {
    throw new Error("pack path is invalid");
  }
  return clean;
}

function assetAuthorized(manifest: PackManifest, relativePath: string): boolean {
  return relativePath !== "features/draft_records.json" || hasPromotedDraftAuthority(manifest);
}

async function readLocalManifest(): Promise<PackManifest> {
  // Minimal secretless build fallback. Runtime production never uses it.
  return JSON.parse(
    await fs.readFile(path.join(process.cwd(), "src", "lib", "bundledPackManifest.json"), "utf8"),
  ) as PackManifest;
}

async function readStorageJson<T>(
  url: string,
  expected?: { bytes: number; sha256: string },
): Promise<T> {
  const response = await fetch(url, {
    headers: { Accept: PUBLIC_ASSET_CONTENT_TYPE },
    // Next's data cache rejects entries over 2 MB. Large immutable files use
    // the CDN response cache and the per-process promise cache instead.
    cache: expected && expected.bytes <= 1_900_000 ? "force-cache" : "no-store",
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new Error(`Public pack asset ${response.status}`);
  const contentType = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  if (contentType !== PUBLIC_ASSET_CONTENT_TYPE) {
    throw new Error("Public pack asset has an invalid content type");
  }
  const raw = new Uint8Array(await response.arrayBuffer());
  if (raw.byteLength > MAX_STORAGE_ASSET_BYTES) {
    throw new Error("Public pack asset is too large");
  }
  if (expected) {
    const digest = createHash("sha256").update(raw).digest("hex");
    if (raw.byteLength !== expected.bytes || digest !== expected.sha256) {
      throw new Error("Public pack asset integrity check failed");
    }
  }
  try {
    return JSON.parse(new TextDecoder().decode(raw)) as T;
  } catch {
    throw new Error("Public pack asset is invalid JSON");
  }
}

function blobPackBase(manifest: PackManifest): string | null {
  const base = manifest.base_url?.trim().replace(/\/$/, "");
  if (!base) return null;
  try {
    const url = new URL(base);
    return url.protocol === "https:"
      && url.port === ""
      && url.hostname.endsWith(".public.blob.vercel-storage.com")
      && !url.username
      && !url.password
      && !url.search
      && !url.hash
      ? url.toString().replace(/\/$/, "")
      : null;
  } catch {
    return null;
  }
}

type ManifestWithRelease = PackManifest & {
  release?: {
    release_id?: string;
    artifact_hashes?: Record<string, string>;
  };
};

export function validatePublicManifest(
  candidate: PackManifest,
  rowReleaseId = candidate.pack_id,
): asserts candidate is ManifestWithRelease {
  const manifest = candidate as ManifestWithRelease;
  if (
    !RELEASE_ID.test(rowReleaseId)
    || manifest.pack_id !== rowReleaseId
    || manifest.data_backend !== "supabase"
    || manifest.release?.release_id !== rowReleaseId
    || !manifest.release.artifact_hashes
    || manifest.total_files !== manifest.files.length
  ) {
    throw new Error("The public manifest release binding is invalid");
  }
  const paths = new Set<string>();
  let totalBytes = 0;
  for (const file of manifest.files) {
    const relativePath = file.relative ?? file.path;
    if (
      file.path !== relativePath
      || !PUBLIC_ASSET_PATHS.has(relativePath)
      || paths.has(relativePath)
      || !Number.isSafeInteger(file.bytes)
      || file.bytes < 0
      || file.bytes > MAX_STORAGE_ASSET_BYTES
      || !SHA256.test(file.sha256)
      || manifest.release.artifact_hashes[relativePath] !== file.sha256
    ) {
      throw new Error("The public manifest file inventory is invalid");
    }
    paths.add(relativePath);
    totalBytes += file.bytes;
  }
  if (
    manifest.total_bytes !== totalBytes
    || Object.keys(manifest.release.artifact_hashes).length !== paths.size
    || Object.keys(manifest.release.artifact_hashes).some((file) => !paths.has(file))
  ) {
    throw new Error("The public manifest digest inventory is invalid");
  }
}

type PublicAssetRow = {
  body?: unknown | null;
  storage_path?: string | null;
  bytes?: number;
  sha256?: string;
  content_type?: string;
};

export type ActivePublicAsset = {
  releaseId: string;
  path: string;
  bytes: number;
  sha256: string;
  contentType: typeof PUBLIC_ASSET_CONTENT_TYPE;
  storagePath: string | null;
  body: unknown | null;
};

function manifestAsset(
  manifest: ManifestWithRelease,
  relativePath: string,
): { bytes: number; sha256: string } | null {
  if (!assetAuthorized(manifest, relativePath)) return null;
  const file = manifest.files.find((candidate) => (
    candidate.relative === relativePath || candidate.path === relativePath
  ));
  if (
    !file
    || !Number.isSafeInteger(file.bytes)
    || file.bytes < 0
    || file.bytes > MAX_STORAGE_ASSET_BYTES
    || !SHA256.test(file.sha256)
  ) {
    return null;
  }
  const bound = manifest.release?.artifact_hashes?.[relativePath];
  if (bound !== undefined && bound !== file.sha256) return null;
  return { bytes: file.bytes, sha256: file.sha256 };
}

async function supabaseJson<T>(
  url: string,
  config: SupabaseConfig,
  cache: RequestCache = "no-store",
  extraHeaders: HeadersInit = {},
): Promise<T> {
  const response = await fetch(url, {
    headers: {
      Accept: PUBLIC_ASSET_CONTENT_TYPE,
      apikey: config.publishableKey,
      ...Object.fromEntries(new Headers(extraHeaders)),
    },
    cache,
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) throw new Error(`Supabase public data ${response.status}`);
  return response.json() as Promise<T>;
}

export async function readActivePublicAsset(
  releaseId: string,
  relativePath: string,
): Promise<ActivePublicAsset | null> {
  const clean = safeRelativePath(relativePath);
  if (!RELEASE_ID.test(releaseId)) return null;
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const release = encodeURIComponent(releaseId);
  const assetPath = encodeURIComponent(clean);
  const releases = await supabaseJson<Array<{
    release_id?: string;
    status?: string;
    manifest?: ManifestWithRelease;
  }>>(
    `${config.url}/rest/v1/scryglass_public_releases?release_id=eq.${release}&status=eq.active&select=release_id,status,manifest&limit=1`,
    config,
  );
  const active = releases[0];
  if (active?.manifest) {
    try {
      validatePublicManifest(active.manifest, releaseId);
    } catch {
      return null;
    }
  }
  const expected = active?.manifest ? manifestAsset(active.manifest, clean) : null;
  if (
    active?.release_id !== releaseId
    || active.status !== "active"
    || active.manifest?.pack_id !== releaseId
    || !expected
  ) {
    return null;
  }
  const assets = await supabaseJson<PublicAssetRow[]>(
    `${config.url}/rest/v1/scryglass_public_assets?release_id=eq.${release}&path=eq.${assetPath}&select=body,storage_path,bytes,sha256,content_type&limit=1`,
    config,
  );
  const row = assets[0];
  const bytes = Number(row?.bytes);
  const sha256 = row?.sha256 ?? "";
  const contentType = row?.content_type?.split(";", 1)[0]?.trim().toLowerCase();
  const storagePath = row?.storage_path ?? null;
  if (
    !row
    || !Number.isSafeInteger(bytes)
    || bytes !== expected.bytes
    || sha256 !== expected.sha256
    || !SHA256.test(sha256)
    || contentType !== PUBLIC_ASSET_CONTENT_TYPE
    || (storagePath !== null && storagePath !== `${releaseId}/${clean}`)
    || (storagePath === null && (row.body === undefined || row.body === null))
  ) {
    return null;
  }
  return {
    releaseId,
    path: clean,
    bytes,
    sha256,
    contentType: PUBLIC_ASSET_CONTENT_TYPE,
    storagePath,
    body: row.body ?? null,
  };
}

function privateStorageHeaders(config: SupabaseConfig): HeadersInit {
  return {
    Accept: PUBLIC_ASSET_CONTENT_TYPE,
    apikey: config.publishableKey,
    Authorization: `Bearer ${config.publishableKey}`,
  };
}

function encodedStoragePath(storagePath: string): string {
  return storagePath.split("/").map((part) => encodeURIComponent(part)).join("/");
}

function customObjectMetadata(payload: unknown): Record<string, unknown> {
  if (!payload || typeof payload !== "object") return {};
  const record = payload as Record<string, unknown>;
  const metadata = record.metadata;
  if (metadata && typeof metadata === "object") return metadata as Record<string, unknown>;
  if (typeof metadata === "string") {
    try {
      const parsed = JSON.parse(metadata) as unknown;
      return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
    } catch {
      return {};
    }
  }
  return {};
}

export async function fetchVerifiedStorageAsset(
  asset: ActivePublicAsset,
): Promise<Response> {
  if (!asset.storagePath) throw new Error("Public asset is not storage-backed");
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const stored = encodedStoragePath(asset.storagePath);
  const headers = privateStorageHeaders(config);
  const info = await supabaseJson<Record<string, unknown>>(
    `${config.url}/storage/v1/object/info/authenticated/scryglass-public/${stored}`,
    config,
    "no-store",
    headers,
  );
  const custom = customObjectMetadata(info);
  if (
    Number(info.size) !== asset.bytes
    || String(info.mimetype ?? info.content_type ?? "").split(";", 1)[0]?.toLowerCase() !== asset.contentType
    || Number(custom.bytes) !== asset.bytes
    || String(custom.sha256 ?? "") !== asset.sha256
    || String(custom.content_type ?? "").split(";", 1)[0]?.toLowerCase() !== asset.contentType
  ) {
    throw new Error("Public Storage metadata does not match the active release");
  }
  const response = await fetch(
    `${config.url}/storage/v1/object/authenticated/scryglass-public/${stored}`,
    {
      headers,
      cache: "force-cache",
      signal: AbortSignal.timeout(60_000),
    },
  );
  if (!response.ok || !response.body) throw new Error(`Public Storage asset ${response.status}`);
  const responseType = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  const responseBytes = Number(response.headers.get("content-length"));
  if (
    responseType !== asset.contentType
    || (Number.isFinite(responseBytes) && responseBytes !== asset.bytes)
  ) {
    throw new Error("Public Storage response metadata is invalid");
  }
  return response;
}

export async function readVerifiedAssetBytes(asset: ActivePublicAsset): Promise<Uint8Array> {
  if (!asset.storagePath) {
    const raw = new TextEncoder().encode(JSON.stringify(asset.body));
    const digest = createHash("sha256").update(raw).digest("hex");
    if (raw.byteLength !== asset.bytes || digest !== asset.sha256) {
      throw new Error("Inline public asset integrity check failed");
    }
    return raw;
  }
  const response = await fetchVerifiedStorageAsset(asset);
  const raw = new Uint8Array(await response.arrayBuffer());
  const digest = createHash("sha256").update(raw).digest("hex");
  if (raw.byteLength !== asset.bytes || digest !== asset.sha256) {
    throw new Error("Public Storage asset integrity check failed");
  }
  return raw;
}

async function readSupabaseManifest(cache: "no-store" | "cached"): Promise<PackManifest> {
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const response = await fetch(
    `${config.url}/rest/v1/scryglass_public_releases?status=eq.active&select=release_id,manifest&limit=1`,
    {
      headers: { apikey: config.publishableKey },
      ...(cache === "no-store"
        ? { cache: "no-store" as const }
        : { next: { revalidate: PACK_CACHE_SECONDS, tags: [PACK_MANIFEST_CACHE_TAG] } }),
    },
  );
  if (!response.ok) throw new Error(`Supabase release ${response.status}`);
  const rows = (await response.json()) as Array<{ release_id?: string; manifest?: ManifestWithRelease }>;
  const row = rows[0];
  const manifest = row?.manifest;
  if (!manifest?.pack_id || !row.release_id) {
    throw new Error("Supabase release is unavailable");
  }
  validatePublicManifest(manifest, row.release_id);
  return manifest;
}

export type PublicRefreshHealth = {
  status: "ok" | "partial" | "error";
  refresh_status: "idle" | "running" | "failed" | "stale";
  checked_at: string;
  last_success_at: string | null;
  source_as_of: string | null;
  active_release_id: string | null;
  last_run_id: string | null;
  worker_commit: string | null;
  stale: boolean;
};

export async function readPublicRefreshHealth(): Promise<PublicRefreshHealth | null> {
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const response = await fetch(
    `${config.url}/rest/v1/scryglass_public_health?health_id=eq.public-refresh&select=status,refresh_status,checked_at,last_success_at,source_as_of,active_release_id,last_run_id,worker_commit,stale&limit=1`,
    {
      headers: { apikey: config.publishableKey },
      cache: "no-store",
    },
  );
  if (!response.ok) throw new Error(`Supabase public health ${response.status}`);
  const rows = (await response.json()) as PublicRefreshHealth[];
  return rows[0] ?? null;
}

/** Read the active Supabase release without a bundled fallback. */
export async function readRemotePackManifest(): Promise<PackManifest> {
  return readSupabaseManifest("no-store");
}

export function publicPackManifest(manifest: PackManifest) {
  validatePublicManifest(manifest);
  const files = manifest.files.flatMap((file) => {
    const relativePath = file.relative ?? file.path;
    if (!PUBLIC_ASSET_PATHS.has(relativePath)) return [];
    if (!assetAuthorized(manifest, relativePath)) return [];
    if (
      !Number.isSafeInteger(file.bytes)
      || file.bytes < 0
      || file.bytes > MAX_STORAGE_ASSET_BYTES
      || !SHA256.test(file.sha256)
    ) return [];
    return [{
      path: relativePath,
      bytes: file.bytes,
      sha256: file.sha256,
      url: `/api/assets/${encodeURIComponent(manifest.pack_id)}/${encodeURIComponent(relativePath)}`,
    }];
  });
  return {
    schema_version: manifest.schema_version,
    release_id: manifest.pack_id,
    pack_id: manifest.pack_id,
    created_utc: manifest.created_utc,
    data_backend: "supabase" as const,
    source_as_of: manifest.ratings?.source_as_of ?? null,
    total_files: files.length,
    total_bytes: files.reduce((total, file) => total + file.bytes, 0),
    files,
  };
}

/** Use bundled data only during builds and local development. */
export async function readPackManifest(): Promise<PackManifest> {
  try {
    return await readSupabaseManifest("cached");
  } catch (error) {
    if (bundledDataAllowed()) return readLocalManifest();
    throw error;
  }
}

function publicOrigin(): string {
  if (process.env.NODE_ENV === "production") {
    const previewHost = process.env.VERCEL_ENV === "preview"
      && /^[a-z0-9-]+\.vercel\.app$/.test(process.env.VERCEL_URL ?? "")
      ? process.env.VERCEL_URL
      : null;
    const raw = previewHost
      ? `https://${previewHost}`
      : (process.env.SCRYGLASS_PUBLISH_ORIGIN || "https://scryglass.xyz").trim();
    const url = new URL(raw);
    if (
      url.protocol !== "https:"
      || (previewHost ? url.hostname !== previewHost : url.hostname !== "scryglass.xyz")
      || url.port
      || url.pathname !== "/"
      || url.search
      || url.hash
      || url.username
      || url.password
    ) {
      throw new Error("The public origin is invalid");
    }
    return url.origin;
  }
  const raw = (process.env.SCRYGLASS_PUBLISH_ORIGIN || "http://127.0.0.1:3000").trim();
  const url = new URL(raw);
  if (
    url.protocol !== "http:"
    || !["127.0.0.1", "localhost"].includes(url.hostname)
    || url.pathname !== "/"
    || url.search
    || url.hash
    || url.username
    || url.password
  ) {
    throw new Error("The local public origin is invalid");
  }
  return url.origin;
}

async function readSupabaseAsset<T>(manifest: PackManifest, relativePath: string): Promise<T> {
  const releaseId = manifest.pack_id;
  const expected = manifestAsset(manifest, relativePath);
  if (!expected) throw new Error("The active manifest does not contain the public asset");
  if (process.env.NEXT_PHASE === "phase-production-build") {
    const asset = await readActivePublicAsset(releaseId, relativePath);
    if (!asset) throw new Error("Supabase public asset is unavailable");
    const raw = await readVerifiedAssetBytes(asset);
    return JSON.parse(new TextDecoder().decode(raw)) as T;
  }
  // Runtime: every asset (storage or DB-row) is served through the Vercel CDN
  // proxy so Supabase egress is one fetch per release per cache window.
  return readStorageJson<T>(
    `${publicOrigin()}/api/assets/${encodeURIComponent(releaseId)}/${encodeURIComponent(relativePath)}`,
    expected,
  );
}

/** Load one immutable release asset from Supabase or bundled build data. */
export async function readPackJson<T>(manifest: PackManifest, relativePath: string): Promise<T> {
  const clean = safeRelativePath(relativePath);
  const localPath = path.join(localPackRoot(), manifest.pack_id, clean);
  const supabase = manifest.data_backend === "supabase" && supabaseConfig();
  const blobBase = supabase ? null : blobPackBase(manifest);
  const cacheKey = supabase
    ? `supabase:${manifest.pack_id}:${clean}`
    : blobBase
      ? `blob:${blobBase}:${clean}`
      : localPath;
  let pending = packJsonCache.get(cacheKey);
  if (!pending) {
    pending = supabase
      ? readSupabaseAsset(manifest, clean)
      : blobBase
        ? readStorageJson<unknown>(`${blobBase}/${clean}`, manifestAsset(manifest, clean) ?? undefined)
        : fs.readFile(localPath, "utf8").then((raw) => JSON.parse(raw) as unknown);
    packJsonCache.set(cacheKey, pending);
    pending.catch(() => packJsonCache.delete(cacheKey));
  }
  return pending as Promise<T>;
}

export async function readPublicTierList<T>(): Promise<T> {
  const manifest = await readSupabaseManifest("no-store");
  return readSupabaseAsset<T>(manifest, "rankings/tierlists.json");
}

/** Send the large tier artifact from storage instead of proxying it through Vercel. */
export async function publicTierListDownloadUrl(): Promise<string> {
  return publicTierListViewDownloadUrl("full");
}

export async function publicTierListViewDownloadUrl(view: "full" | "latest"): Promise<string> {
  const relativePath = view === "latest"
    ? "rankings/tierlists-latest.json"
    : "rankings/tierlists.json";
  const manifest = await readSupabaseManifest("no-store");
  const asset = await readActivePublicAsset(manifest.pack_id, relativePath);
  if (!asset) throw new Error("Supabase public tier asset is unavailable");
  return `/api/assets/${encodeURIComponent(manifest.pack_id)}/${encodeURIComponent(relativePath)}`;
}
