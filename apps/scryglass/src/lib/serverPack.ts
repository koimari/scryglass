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
  "features/match_records_2025_q1.json",
  "features/match_records_2025_q2.json",
  "features/match_records_2025_q3.json",
  "features/match_records_2025_q4.json",
  "features/match_records_2026.json",
  "features/match_records_2026_q1.json",
  "features/match_records_2026_q2.json",
  "features/match_records_2026_q3.json",
  "features/match_records_2026_q4.json",
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

const E2E_LOCAL_PACK_FLAG = "SCRYGLASS_E2E_LOCAL_PACK";
const VERCEL_RUNTIME_KEYS = ["VERCEL", "VERCEL_ENV", "VERCEL_URL", "VERCEL_REGION"] as const;

/** Resolve the generated browser fixture only for an explicit, non-Vercel E2E run. */
export function e2eLocalPackRoot(
  env: Readonly<Record<string, string | undefined>> = process.env,
  cwd = process.cwd(),
): string | null {
  if (env[E2E_LOCAL_PACK_FLAG] !== "1") return null;
  if (VERCEL_RUNTIME_KEYS.some((key) => Boolean(env[key]?.trim()))) {
    throw new Error("The local E2E pack is disabled on Vercel");
  }
  return path.join(path.resolve(cwd), "output", "playwright", "e2e-pack");
}

function localPackRoot(): string {
  const e2eRoot = e2eLocalPackRoot();
  if (!e2eRoot) throw new Error("Local public packs are available only to the E2E fixture");
  return e2eRoot;
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
  const manifestPath = path.join(localPackRoot(), "manifest.json");
  return JSON.parse(
    await fs.readFile(manifestPath, "utf8"),
  ) as PackManifest;
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
  storagePath: string;
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

function boundedSignal(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

async function supabaseJson<T>(
  url: string,
  config: SupabaseConfig,
  cache: RequestCache = "no-store",
  extraHeaders: HeadersInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(url, {
    headers: {
      Accept: PUBLIC_ASSET_CONTENT_TYPE,
      apikey: config.publishableKey,
      ...Object.fromEntries(new Headers(extraHeaders)),
    },
    cache,
    signal: boundedSignal(signal, 10_000),
  });
  if (!response.ok) throw new Error(`Supabase public data ${response.status}`);
  return response.json() as Promise<T>;
}

export async function readActivePublicAsset(
  releaseId: string,
  relativePath: string,
  signal?: AbortSignal,
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
    `${config.url}/rest/v1/rpc/get_scryglass_active_release?p_release_id=${release}`,
    config,
    "no-store",
    {},
    signal,
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
    `${config.url}/rest/v1/rpc/get_scryglass_active_asset?p_release_id=${release}&p_path=${assetPath}`,
    config,
    "no-store",
    {},
    signal,
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
    || storagePath !== `${releaseId}/${clean}`
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
  };
}

function privateStorageHeaders(config: SupabaseConfig): HeadersInit {
  return {
    Accept: PUBLIC_ASSET_CONTENT_TYPE,
    apikey: config.publishableKey,
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
  signal?: AbortSignal,
): Promise<Response> {
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const stored = encodedStoragePath(asset.storagePath);
  const headers = privateStorageHeaders(config);
  const info = await supabaseJson<Record<string, unknown>>(
    `${config.url}/storage/v1/object/info/authenticated/scryglass-public/${stored}`,
    config,
    "no-store",
    headers,
    signal,
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
      signal: boundedSignal(signal, 60_000),
    },
  );
  if (!response.ok || !response.body) throw new Error(`Public Storage asset ${response.status}`);
  const responseType = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  const responseLength = response.headers.get("content-length");
  const responseBytes = responseLength && /^\d+$/.test(responseLength)
    ? Number(responseLength)
    : null;
  if (
    responseType !== asset.contentType
    || (
      responseBytes !== null
      && (!Number.isSafeInteger(responseBytes) || responseBytes !== asset.bytes)
    )
  ) {
    throw new Error("Public Storage response metadata is invalid");
  }
  return response;
}

export async function readVerifiedAssetBytes(
  asset: ActivePublicAsset,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const response = await fetchVerifiedStorageAsset(asset, signal);
  const raw = new Uint8Array(await response.arrayBuffer());
  const digest = createHash("sha256").update(raw).digest("hex");
  if (raw.byteLength !== asset.bytes || digest !== asset.sha256) {
    throw new Error("Public Storage asset integrity check failed");
  }
  return raw;
}

async function readSupabaseManifest(
  cache: "no-store" | "cached",
  signal?: AbortSignal,
): Promise<PackManifest> {
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const response = await fetch(
    `${config.url}/rest/v1/rpc/get_scryglass_active_release`,
    {
      headers: { apikey: config.publishableKey },
      ...(cache === "no-store"
        ? { cache: "no-store" as const }
        : { next: { revalidate: PACK_CACHE_SECONDS, tags: [PACK_MANIFEST_CACHE_TAG] } }),
      signal: boundedSignal(signal, 10_000),
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
  if (
    manifest.query_api?.schema_version !== "scryglass:query-api:v1"
    || manifest.query_api.status !== "available"
  ) {
    throw new Error("The active release has no bounded public query API");
  }
  return manifest;
}

export type PublicRefreshHealth = {
  status: "ok" | "partial" | "error";
  refresh_status: "idle" | "running" | "failed" | "stale";
  checked_at: string;
  last_success_at: string | null;
  source_as_of: string | null;
  active_release_id: string | null;
  stale: boolean;
};

export type PrivateRefreshHealth = PublicRefreshHealth & {
  last_run_id: string | null;
  worker_commit: string | null;
};

export async function readPublicRefreshHealth(): Promise<PublicRefreshHealth | null> {
  const config = supabaseConfig();
  if (!config) throw new Error("Supabase public data is not configured");
  const response = await fetch(
    `${config.url}/rest/v1/rpc/get_scryglass_public_health`,
    {
      headers: { apikey: config.publishableKey },
      cache: "no-store",
    },
  );
  if (!response.ok) throw new Error(`Supabase public health ${response.status}`);
  const rows = (await response.json()) as PublicRefreshHealth[];
  return rows[0] ?? null;
}

export async function readPrivateRefreshHealth(): Promise<PrivateRefreshHealth | null> {
  const config = supabaseConfig();
  const token = (process.env.SCRYGLASS_DIAGNOSTIC_TOKEN || "").trim();
  if (!config || token.length < 32 || token.length > 512) {
    throw new Error("Supabase diagnostics are not configured");
  }
  const response = await fetch(
    `${config.url}/rest/v1/rpc/get_scryglass_private_health`,
    {
      method: "POST",
      headers: {
        apikey: config.publishableKey,
        "content-type": "application/json",
      },
      body: JSON.stringify({ p_token: token }),
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    },
  );
  if (!response.ok) throw new Error(`Supabase diagnostic health ${response.status}`);
  const rows = (await response.json()) as PrivateRefreshHealth[];
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

/** Read the active remote release, except for the explicit local E2E fixture. */
export async function readPackManifest(signal?: AbortSignal): Promise<PackManifest> {
  if (e2eLocalPackRoot()) return readLocalManifest();
  return readSupabaseManifest("cached", signal);
}

async function readSupabaseAsset<T>(
  manifest: PackManifest,
  relativePath: string,
  signal?: AbortSignal,
): Promise<T> {
  const releaseId = manifest.pack_id;
  const expected = manifestAsset(manifest, relativePath);
  if (!expected) throw new Error("The active manifest does not contain the public asset");
  const asset = await readActivePublicAsset(releaseId, relativePath, signal);
  if (asset && asset.bytes === expected.bytes && asset.sha256 === expected.sha256) {
    const raw = await readVerifiedAssetBytes(asset, signal);
    return JSON.parse(new TextDecoder().decode(raw)) as T;
  }
  throw new Error("Supabase public Storage asset is unavailable");
}

/** Load one immutable release asset from private Storage or the E2E fixture. */
export async function readPackJson<T>(
  manifest: PackManifest,
  relativePath: string,
  signal?: AbortSignal,
): Promise<T> {
  const clean = safeRelativePath(relativePath);
  const supabase = manifest.data_backend === "supabase" && supabaseConfig();
  const localPath = supabase
    ? null
    : path.join(
        /* turbopackIgnore: true */ localPackRoot(),
        manifest.pack_id,
        clean,
      );
  const cacheKey = supabase ? `supabase:${manifest.pack_id}:${clean}` : localPath!;
  if (signal) {
    return supabase
      ? readSupabaseAsset<T>(manifest, clean, signal)
      : fs.readFile(/* turbopackIgnore: true */ localPath!, "utf8").then((raw) => JSON.parse(raw) as T);
  }
  let pending = packJsonCache.get(cacheKey);
  if (!pending) {
    pending = supabase
      ? readSupabaseAsset(manifest, clean)
      : fs.readFile(/* turbopackIgnore: true */ localPath!, "utf8").then((raw) => JSON.parse(raw) as unknown);
    packJsonCache.set(cacheKey, pending);
    pending.catch(() => packJsonCache.delete(cacheKey));
  }
  return pending as Promise<T>;
}

export async function readPublicTierList<T>(): Promise<T> {
  if (e2eLocalPackRoot()) {
    const manifest = await readLocalManifest();
    return readPackJson<T>(manifest, "rankings/tierlists.json");
  }
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
