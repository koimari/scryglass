import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { packUrl, type PackManifest } from "./pack";

export type PackRetrievalSource = "remote" | "bundled";

export type PackDegradedReason =
  | "remote_unavailable"
  | "remote_http_error"
  | "remote_invalid";

export type PackClock = {
  value: string | null;
  field: "created_utc" | "data_as_of";
  status: "available" | "not_declared";
};

export type PackArtifactIds = {
  data_pack_id: string;
  model_pack_id: string | null;
  strength_snapshot_sha256: string | null;
  calibration_sha256: string | null;
  one_clock_verified: boolean;
  status: "verified_same_bundle" | "not_declared" | "mismatch";
};

export type PackSourceProvenance = {
  sources: Array<{ source: "oe" | "grid" | string; rows: number | null }>;
  attribution: string;
  canonicalization: string;
  overlap_precedence: string;
  overlap_precedence_status: "declared" | "not_declared";
};

/**
 * A manifest and its retrieval metadata are one value. Existing pack consumers
 * can continue to use the PackManifest fields, while public/API surfaces can
 * distinguish the immutable bundle, publication clock, observation clock, and
 * degraded fallback state.
 */
export type ResolvedPackManifest = PackManifest & {
  source: PackRetrievalSource;
  degraded: boolean;
  degraded_reason: PackDegradedReason | null;
  clocks: {
    publication: PackClock;
    data_through: PackClock;
  };
  artifact_ids: PackArtifactIds;
  source_provenance: PackSourceProvenance;
};

export type PackServerOptions = {
  packRoot?: string;
  fetchImpl?: typeof fetch;
  now?: () => number;
  configuredManifestUrl?: string | null;
};

type UnknownRecord = Record<string, unknown>;

type ReadSession = {
  source: PackRetrievalSource;
  successfulRemoteReads: number;
  tail: Promise<void>;
};

const readSessions = new WeakMap<object, ReadSession>();

export class PackServiceError extends Error {
  readonly code:
    | "PACK_MANIFEST_UNAVAILABLE"
    | "PACK_MANIFEST_INVALID"
    | "PACK_FILE_UNAVAILABLE"
    | "PACK_BUNDLE_INTEGRITY";

  constructor(
    code: PackServiceError["code"],
    message: string,
    options?: { cause?: unknown },
  ) {
    super(message, options);
    this.name = "PackServiceError";
    this.code = code;
  }
}

function defaultPackRoot(): string {
  return path.join(process.cwd(), "public", "packs");
}

function asRecord(value: unknown): UnknownRecord | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null;
}

function parseManifest(raw: string, origin: string): PackManifest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new PackServiceError(
      "PACK_MANIFEST_INVALID",
      `Pack manifest is invalid (${origin}).`,
      { cause: error },
    );
  }

  const record = asRecord(parsed);
  if (
    !record ||
    typeof record.pack_id !== "string" ||
    !record.pack_id.trim() ||
    typeof record.created_utc !== "string" ||
    !Number.isFinite(Date.parse(record.created_utc)) ||
    (record.data_as_of != null &&
      (typeof record.data_as_of !== "string" ||
        !Number.isFinite(Date.parse(record.data_as_of)))) ||
    !Array.isArray(record.files)
  ) {
    throw new PackServiceError(
      "PACK_MANIFEST_INVALID",
      `Pack manifest is invalid (${origin}).`,
    );
  }
  return parsed as PackManifest;
}

function normalizeRelativePath(relativePath: string): string {
  const normalized = relativePath.replaceAll("\\", "/").replace(/^\/+/, "");
  if (
    !normalized ||
    normalized === "." ||
    normalized.startsWith("../") ||
    normalized.includes("/../") ||
    path.posix.isAbsolute(normalized)
  ) {
    throw new PackServiceError(
      "PACK_FILE_UNAVAILABLE",
      "Requested pack file path is invalid.",
    );
  }
  return normalized;
}

function sourceCounts(manifest: PackManifest): Map<string, number | null> {
  const manifestRecord = manifest as PackManifest & UnknownRecord;
  const sourceSummary = asRecord(manifestRecord.source_summary);
  const summarySources = asRecord(sourceSummary?.sources);
  const canonical = asRecord(summarySources?.canonical_map_inclusion);
  const detail = asRecord(summarySources?.map_detail_enrichment);
  const counts = new Map<string, number | null>();

  if (canonical) {
    const oe = asRecord(canonical.oe);
    const oeMaps = oe?.maps;
    if (typeof oeMaps === "number" && Number.isFinite(oeMaps)) {
      counts.set("oe", oeMaps);
    }
    const gridGap = asRecord(canonical.grid_gap_fill);
    const gridGapMaps = gridGap?.maps;
    const gridDetailPresent =
      detail != null &&
      Object.entries(detail).some(([source, block]) => {
        const maps = asRecord(block)?.maps;
        return (
          source.startsWith("grid_") &&
          typeof maps === "number" &&
          Number.isFinite(maps) &&
          maps > 0
        );
      });
    if (
      gridDetailPresent ||
      (typeof gridGapMaps === "number" &&
        Number.isFinite(gridGapMaps) &&
        gridGapMaps > 0)
    ) {
      // GRID may be canonical gap fill, detail enrichment, or both. Those
      // grains are not additive, so an aggregate row count would mislead.
      counts.set(
        "grid",
        gridDetailPresent ? null : (gridGapMaps as number),
      );
    }
    if (counts.size > 0) return counts;
  }

  const ingest = asRecord(manifestRecord.ingest);
  const refreshMeta = asRecord(ingest?.refresh_meta);
  const rawCounts = asRecord(refreshMeta?.source_counts);

  if (rawCounts) {
    for (const [rawSource, rawCount] of Object.entries(rawCounts)) {
      const source = rawSource.trim().toLowerCase();
      if (!source) continue;
      counts.set(
        source,
        typeof rawCount === "number" && Number.isFinite(rawCount)
          ? rawCount
          : null,
      );
    }
  }

  if (counts.size === 0) {
    const attribution = manifest.attribution?.toLowerCase() ?? "";
    if (attribution.includes("oracle") || attribution.includes("elixir")) {
      counts.set("oe", null);
    }
    if (attribution.includes("grid")) counts.set("grid", null);
  }
  return counts;
}

function declaredOverlapPrecedence(manifest: PackManifest): string | null {
  const manifestRecord = manifest as PackManifest & UnknownRecord;
  const sourceSummary = asRecord(manifestRecord.source_summary);
  const canonicalization = asRecord(sourceSummary?.canonicalization);
  const ingest = asRecord(manifestRecord.ingest);
  const refreshMeta = asRecord(ingest?.refresh_meta);
  const value =
    canonicalization?.overlap_precedence ??
    manifestRecord.overlap_precedence ??
    ingest?.overlap_precedence ??
    refreshMeta?.overlap_precedence;
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function buildSourceProvenance(manifest: PackManifest): PackSourceProvenance {
  const manifestRecord = manifest as PackManifest & UnknownRecord;
  const sourceSummary = asRecord(manifestRecord.source_summary);
  const summaryCanonicalization = asRecord(
    sourceSummary?.canonicalization,
  );
  const counts = sourceCounts(manifest);
  const sources = [...counts.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([source, rows]) => ({ source, rows }));
  const sourceNames = sources.map(({ source }) => {
    if (source === "oe") return "Oracle's Elixir (OE)";
    if (source === "grid") return "GRID";
    return source.toUpperCase();
  });
  const countText = sources
    .map(({ source, rows }) => `${source.toUpperCase()} ${rows ?? "count not declared"}`)
    .join("; ");
  const overlapPrecedence = declaredOverlapPrecedence(manifest);

  const declaredAttribution =
    typeof sourceSummary?.attribution === "string" &&
    sourceSummary.attribution.trim()
      ? sourceSummary.attribution.trim()
      : typeof manifest.attribution === "string" &&
          manifest.attribution.trim() &&
          sourceSummary
        ? manifest.attribution.trim()
        : null;
  const attribution =
    declaredAttribution ??
    (sourceNames.length > 0
      ? `Pro-match rows in this immutable pack are attributed to ${sourceNames.join(
          " and ",
        )}${countText ? ` (${countText})` : ""}. Ratings and model artifacts are Scryglass outputs.`
      : "Source attribution is not declared by this immutable pack; row-level provenance must be inspected before reuse.");

  const canonicalField =
    typeof summaryCanonicalization?.canonical_inclusion_field === "string"
      ? summaryCanonicalization.canonical_inclusion_field
      : null;
  const detailField =
    typeof summaryCanonicalization?.detail_enrichment_field === "string"
      ? summaryCanonicalization.detail_enrichment_field
      : null;

  return {
    sources,
    attribution,
    canonicalization:
      canonicalField && detailField
        ? `Canonical inclusion is declared by ${canonicalField}; optional detail enrichment is separately declared by ${detailField}.`
        : "Rows retain source provenance and are published through the canonical map, team, and player grains declared by this immutable pack.",
    overlap_precedence:
      overlapPrecedence ??
      "OE/GRID overlap precedence is not declared by this immutable manifest; overlap-dependent claims are not verified.",
    overlap_precedence_status: overlapPrecedence ? "declared" : "not_declared",
  };
}

function explicitModelPackId(manifest: PackManifest): string | null {
  const record = manifest as PackManifest & UnknownRecord;
  const models = asRecord(record.models);
  const candidates = [
    record.model_pack_id,
    record.model_bundle_id,
    models?.pack_id,
    models?.bundle_id,
  ];
  const value = candidates.find(
    (candidate) => typeof candidate === "string" && candidate.trim(),
  );
  return typeof value === "string" ? value.trim() : null;
}

function fileSha(manifest: PackManifest, relativePaths: string[]): string | null {
  const candidates = new Set(relativePaths);
  const file = manifest.files.find((entry) =>
    candidates.has(entry.relative ?? entry.path),
  );
  return file?.sha256 || null;
}

function buildArtifactIds(manifest: PackManifest): PackArtifactIds {
  const modelPackId = explicitModelPackId(manifest);
  const strengthSnapshot = fileSha(manifest, [
    "features/ratings_snapshot.json",
    "features/ratings_snapshot.parquet",
  ]);
  const calibration = fileSha(manifest, [
    "models/elo_wr_calibration.json",
    "models/draft_wr_calibration.json",
  ]);
  const status =
    modelPackId == null
      ? "not_declared"
      : modelPackId === manifest.pack_id
        ? "verified_same_bundle"
        : "mismatch";

  return {
    data_pack_id: manifest.pack_id,
    model_pack_id: modelPackId,
    strength_snapshot_sha256: strengthSnapshot,
    calibration_sha256: calibration,
    one_clock_verified: status === "verified_same_bundle",
    status,
  };
}

function resolveManifest(
  manifest: PackManifest,
  source: PackRetrievalSource,
  degraded: boolean,
  degradedReason: PackDegradedReason | null,
): ResolvedPackManifest {
  const sourceProvenance = buildSourceProvenance(manifest);
  const dataThrough = manifest.data_as_of ?? null;
  return {
    ...manifest,
    attribution: sourceProvenance.attribution,
    source,
    degraded,
    degraded_reason: degradedReason,
    clocks: {
      publication: {
        value: manifest.created_utc,
        field: "created_utc",
        status: "available",
      },
      data_through: {
        value: dataThrough,
        field: "data_as_of",
        status: dataThrough ? "available" : "not_declared",
      },
    },
    artifact_ids: buildArtifactIds(manifest),
    source_provenance: sourceProvenance,
  };
}

async function readPointerManifest(packRoot: string): Promise<PackManifest | null> {
  try {
    return parseManifest(
      await fs.readFile(path.join(packRoot, "manifest.json"), "utf8"),
      "bundled pointer",
    );
  } catch {
    return null;
  }
}

async function readBundledManifest(packRoot: string): Promise<PackManifest> {
  let entries;
  try {
    entries = await fs.readdir(packRoot, { withFileTypes: true });
  } catch (error) {
    throw new PackServiceError(
      "PACK_MANIFEST_UNAVAILABLE",
      "No bundled pack manifest is available.",
      { cause: error },
    );
  }

  const candidates: PackManifest[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const manifestPath = path.join(packRoot, entry.name, "manifest.json");
    try {
      const manifest = parseManifest(
        await fs.readFile(manifestPath, "utf8"),
        `bundled pack ${entry.name}`,
      );
      if (manifest.pack_id !== entry.name) continue;
      candidates.push(manifest);
    } catch {
      // One malformed historical bundle must not hide a valid immutable bundle.
    }
  }

  candidates.sort((a, b) => {
    const byPublished = Date.parse(b.created_utc) - Date.parse(a.created_utc);
    return byPublished || b.pack_id.localeCompare(a.pack_id);
  });
  const manifest = candidates[0];
  if (!manifest) {
    throw new PackServiceError(
      "PACK_MANIFEST_UNAVAILABLE",
      "No valid bundled pack manifest is available.",
    );
  }
  return manifest;
}

function liveManifestUrl(
  pointer: PackManifest | null,
  configuredManifestUrl: string | null | undefined,
): string | null {
  if (configuredManifestUrl === null) return null;
  const configured =
    configuredManifestUrl === undefined
      ? process.env.SCRYGLASS_PACK_MANIFEST_URL?.trim()
      : configuredManifestUrl?.trim();
  if (configured) return configured;
  if (!pointer?.base_url?.startsWith("http")) return null;
  const marker = "/packs/";
  const markerAt = pointer.base_url.indexOf(marker);
  if (markerAt < 0) return null;
  return `${pointer.base_url.slice(0, markerAt)}${marker}manifest.json`;
}

async function fetchRemoteManifest(
  remoteUrl: string,
  fetchImpl: typeof fetch,
  now: () => number,
): Promise<
  | { ok: true; manifest: PackManifest }
  | { ok: false; reason: PackDegradedReason }
> {
  try {
    const bucket = Math.floor(now() / 60_000);
    const separator = remoteUrl.includes("?") ? "&" : "?";
    const response = await fetchImpl(`${remoteUrl}${separator}v=${bucket}`, {
      cache: "no-store",
    });
    if (!response.ok) return { ok: false, reason: "remote_http_error" };
    try {
      const manifest = parseManifest(await response.text(), "remote pointer");
      if (!manifest.base_url?.startsWith("http")) {
        return { ok: false, reason: "remote_invalid" };
      }
      return {
        ok: true,
        manifest,
      };
    } catch {
      return { ok: false, reason: "remote_invalid" };
    }
  } catch {
    return { ok: false, reason: "remote_unavailable" };
  }
}

/**
 * Resolve the live immutable pack pointer, with a real bundled pack as fallback.
 * The top-level public/packs/manifest.json is only a mutable pointer; it is
 * never treated as proof that matching local files exist.
 */
export async function readPackManifest(
  options: PackServerOptions = {},
): Promise<ResolvedPackManifest> {
  const packRoot = options.packRoot ?? defaultPackRoot();
  const bundled = await readBundledManifest(packRoot);
  const pointer = await readPointerManifest(packRoot);
  const remoteUrl = liveManifestUrl(pointer, options.configuredManifestUrl);

  if (!remoteUrl) {
    const resolved = resolveManifest(bundled, "bundled", false, null);
    readSessions.set(resolved, {
      source: "bundled",
      successfulRemoteReads: 0,
      tail: Promise.resolve(),
    });
    return resolved;
  }

  const remote = await fetchRemoteManifest(
    remoteUrl,
    options.fetchImpl ?? fetch,
    options.now ?? Date.now,
  );
  if (remote.ok) {
    const resolved = resolveManifest(remote.manifest, "remote", false, null);
    readSessions.set(resolved, {
      source: "remote",
      successfulRemoteReads: 0,
      tail: Promise.resolve(),
    });
    return resolved;
  }

  const resolved = resolveManifest(bundled, "bundled", true, remote.reason);
  readSessions.set(resolved, {
    source: "bundled",
    successfulRemoteReads: 0,
    tail: Promise.resolve(),
  });
  return resolved;
}

function declaredFileSha(
  manifest: PackManifest,
  relativePath: string,
  expectedSha256?: string | null,
): string {
  const file = manifest.files.find(
    (entry) => (entry.relative ?? entry.path) === relativePath,
  );
  const declared = file?.sha256?.trim().toLowerCase() ?? "";
  const expected = expectedSha256?.trim().toLowerCase() ?? null;
  if (!file || !/^[a-f0-9]{64}$/.test(declared)) {
    throw new PackServiceError(
      "PACK_BUNDLE_INTEGRITY",
      `Pack does not declare a valid SHA-256 for ${relativePath}.`,
    );
  }
  if (expected && (!/^[a-f0-9]{64}$/.test(expected) || expected !== declared)) {
    throw new PackServiceError(
      "PACK_BUNDLE_INTEGRITY",
      `Pack and runtime hashes do not reconcile for ${relativePath}.`,
    );
  }
  return declared;
}

function verifyFileSha(
  bytes: Uint8Array,
  declaredSha256: string,
  relativePath: string,
): void {
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== declaredSha256) {
    throw new PackServiceError(
      "PACK_BUNDLE_INTEGRITY",
      `Pack file hash mismatch: ${relativePath}.`,
    );
  }
}

async function readBundledJson<T>(
  manifest: PackManifest,
  relativePath: string,
  packRoot: string,
  expectedSha256?: string | null,
): Promise<T> {
  const declaredSha256 = declaredFileSha(
    manifest,
    relativePath,
    expectedSha256,
  );
  let bytes: Buffer;
  try {
    bytes = await fs.readFile(
      path.join(packRoot, manifest.pack_id, ...relativePath.split("/")),
    );
  } catch (error) {
    throw new PackServiceError(
      "PACK_FILE_UNAVAILABLE",
      `Bundled pack file is unavailable: ${relativePath}.`,
      { cause: error },
    );
  }
  verifyFileSha(bytes, declaredSha256, relativePath);
  try {
    return JSON.parse(bytes.toString("utf8")) as T;
  } catch (error) {
    throw new PackServiceError(
      "PACK_BUNDLE_INTEGRITY",
      `Bundled pack JSON is invalid: ${relativePath}.`,
      { cause: error },
    );
  }
}

function replaceManifestInPlace(
  target: ResolvedPackManifest,
  replacement: ResolvedPackManifest,
): void {
  for (const key of Object.keys(target)) {
    delete (target as unknown as UnknownRecord)[key];
  }
  Object.assign(target, replacement);
}

/**
 * Load JSON from one immutable bundle.
 *
 * If the first remote file read fails, the manifest and requested file switch
 * together to the latest valid bundled pack. If any remote file has already
 * been consumed, switching would create a mixed-pack response, so the read
 * fails closed instead.
 */
export async function readPackJson<T>(
  manifest: PackManifest,
  relativePath: string,
  options: Pick<PackServerOptions, "packRoot" | "fetchImpl"> & {
    expectedSha256?: string | null;
  } = {},
): Promise<T> {
  const safeRelativePath = normalizeRelativePath(relativePath);
  const packRoot = options.packRoot ?? defaultPackRoot();
  const resolved = manifest as ResolvedPackManifest;
  const session =
    readSessions.get(manifest) ??
    ({
      source: resolved.source === "remote" ? "remote" : "bundled",
      successfulRemoteReads: 0,
      tail: Promise.resolve(),
    } satisfies ReadSession);
  readSessions.set(manifest, session);

  const waitForPrevious = session.tail;
  let release: () => void = () => {};
  session.tail = new Promise<void>((resolve) => {
    release = resolve;
  });
  await waitForPrevious;

  try {
    if (session.source === "remote" && manifest.base_url?.startsWith("http")) {
      let fallbackReason: PackDegradedReason = "remote_unavailable";
      try {
        const declaredSha256 = declaredFileSha(
          manifest,
          safeRelativePath,
          options.expectedSha256,
        );
        const response = await (options.fetchImpl ?? fetch)(
          packUrl(manifest, safeRelativePath),
          { cache: "no-store" },
        );
        if (response.ok) {
          try {
            const bytes = new Uint8Array(await response.arrayBuffer());
            verifyFileSha(bytes, declaredSha256, safeRelativePath);
            const value = JSON.parse(
              Buffer.from(bytes).toString("utf8"),
            ) as T;
            session.successfulRemoteReads += 1;
            return value;
          } catch {
            fallbackReason = "remote_invalid";
          }
        } else {
          fallbackReason = "remote_http_error";
        }
      } catch {
        // The atomic fallback/fail-closed decision is made below.
      }

      if (session.successfulRemoteReads > 0) {
        throw new PackServiceError(
          "PACK_BUNDLE_INTEGRITY",
          "A remote pack file failed after this request had already consumed remote pack data.",
        );
      }

      const bundledManifest = await readBundledManifest(packRoot);
      const bundled = resolveManifest(
        bundledManifest,
        "bundled",
        true,
        fallbackReason,
      );
      const value = await readBundledJson<T>(
        bundled,
        safeRelativePath,
        packRoot,
        options.expectedSha256,
      );
      replaceManifestInPlace(resolved, bundled);
      session.source = "bundled";
      return value;
    }

    const bundledManifest =
      resolved.source === "bundled" && manifest.pack_id
        ? manifest
        : resolveManifest(
            await readBundledManifest(packRoot),
            "bundled",
            true,
            "remote_unavailable",
          );
    return await readBundledJson<T>(
      bundledManifest,
      safeRelativePath,
      packRoot,
      options.expectedSha256,
    );
  } finally {
    release();
  }
}
