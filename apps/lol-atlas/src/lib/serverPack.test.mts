import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import type { PackManifest } from "./pack.ts";
import {
  PackServiceError,
  readPackJson,
  readPackManifest,
} from "./serverPack.ts";

const FILE_A = "features/a.json";
const FILE_B = "features/b.json";
const LOCAL_FILES = {
  [FILE_A]: { bundle: "local", file: "a" },
  [FILE_B]: { bundle: "local", file: "b" },
};
const REMOTE_FILES = {
  [FILE_A]: { bundle: "remote" },
  [FILE_B]: { bundle: "remote", file: "b" },
};

function jsonBytes(value: unknown): Buffer {
  return Buffer.from(JSON.stringify(value));
}

function jsonSha(value: unknown): string {
  return createHash("sha256").update(jsonBytes(value)).digest("hex");
}

function manifest(
  packId: string,
  createdUtc: string,
  options: {
    baseUrl?: string | null;
    dataAsOf?: string | null;
    files?: string[];
    sourceCounts?: Record<string, number>;
    sourceSummary?: PackManifest["source_summary"];
    modelPackId?: string;
    fileBodies?: Record<string, unknown>;
  } = {},
): PackManifest {
  return {
    pack_id: packId,
    schema_version: "test",
    created_utc: createdUtc,
    data_as_of: options.dataAsOf,
    filters: { years: [2026], leagues: "test" },
    attribution: "fixture",
    excluded: [],
    base_url: options.baseUrl ?? null,
    total_bytes: 0,
    total_files: (options.files ?? [FILE_A, FILE_B]).length,
    files: (options.files ?? [FILE_A, FILE_B]).map((relative) => {
      const body = options.fileBodies?.[relative] ?? LOCAL_FILES[relative as keyof typeof LOCAL_FILES];
      return {
        path: relative,
        relative,
        rows: 1,
        cols: 1,
        bytes: body == null ? 0 : jsonBytes(body).byteLength,
        sha256: body == null ? "0".repeat(64) : jsonSha(body),
      };
    }),
    ingest: {
      refresh_meta: {
        source_counts: options.sourceCounts ?? { oe: 2 },
      },
    },
    ...(options.sourceSummary
      ? { source_summary: options.sourceSummary }
      : {}),
    ...(options.modelPackId
      ? { model_pack_id: options.modelPackId }
      : {}),
  } as PackManifest;
}

async function writeBundle(
  root: string,
  value: PackManifest,
  files: Record<string, unknown>,
): Promise<void> {
  const bundleRoot = path.join(root, value.pack_id);
  await mkdir(bundleRoot, { recursive: true });
  await writeFile(
    path.join(bundleRoot, "manifest.json"),
    JSON.stringify(value),
  );
  for (const [relative, body] of Object.entries(files)) {
    const target = path.join(bundleRoot, ...relative.split("/"));
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, JSON.stringify(body));
  }
}

async function fixture(): Promise<{
  root: string;
  local: PackManifest;
  remote: PackManifest;
  cleanup: () => Promise<void>;
}> {
  const root = await mkdtemp(path.join(os.tmpdir(), "scryglass-pack-"));
  const local = manifest("v-local", "2026-07-26T10:00:00Z", {
    dataAsOf: "2026-07-25T22:00:00Z",
    sourceCounts: { oe: 20 },
  });
  const remote = manifest("v-remote", "2026-07-27T10:00:00Z", {
    baseUrl: "https://blob.example/packs/v-remote",
    dataAsOf: "2026-07-27T08:00:00Z",
    sourceCounts: { oe: 30, grid: 4 },
    fileBodies: REMOTE_FILES,
  });
  await writeBundle(root, local, LOCAL_FILES);
  await writeFile(
    path.join(root, "manifest.json"),
    JSON.stringify(
      manifest("v-pointer-without-local-directory", "2026-07-27T11:00:00Z", {
        baseUrl: "https://blob.example/packs/v-pointer-without-local-directory",
      }),
    ),
  );
  return {
    root,
    local,
    remote,
    cleanup: () => rm(root, { recursive: true, force: true }),
  };
}

test("remote failure selects a real immutable bundled manifest, not the mutable pointer ID", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      fetchImpl: async () => {
        throw new Error("private network detail");
      },
      now: () => 0,
    });
    assert.equal(resolved.pack_id, f.local.pack_id);
    assert.equal(resolved.source, "bundled");
    assert.equal(resolved.degraded, true);
    assert.equal(resolved.degraded_reason, "remote_unavailable");
  } finally {
    await f.cleanup();
  }
});

test("publication and data-through clocks remain distinct and missing data_as_of is not invented", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      fetchImpl: async () =>
        new Response(JSON.stringify(f.remote), { status: 200 }),
      now: () => 0,
    });
    assert.equal(
      resolved.clocks.publication.value,
      "2026-07-27T10:00:00Z",
    );
    assert.equal(
      resolved.clocks.data_through.value,
      "2026-07-27T08:00:00Z",
    );
    assert.notEqual(
      resolved.clocks.publication.value,
      resolved.clocks.data_through.value,
    );

    const withoutDataClock = manifest(
      "v-no-data-clock",
      "2026-07-27T12:00:00Z",
      { baseUrl: "https://blob.example/packs/v-no-data-clock" },
    );
    const missing = await readPackManifest({
      packRoot: f.root,
      configuredManifestUrl: "https://manifest.example/current.json",
      fetchImpl: async () =>
        new Response(JSON.stringify(withoutDataClock), { status: 200 }),
      now: () => 0,
    });
    assert.equal(missing.clocks.data_through.value, null);
    assert.equal(missing.clocks.data_through.status, "not_declared");
  } finally {
    await f.cleanup();
  }
});

test("mixed-source attribution is derived from source counts and model clock is not claimed without an ID", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      fetchImpl: async () =>
        new Response(JSON.stringify(f.remote), { status: 200 }),
      now: () => 0,
    });
    assert.deepEqual(
      resolved.source_provenance.sources.map(({ source, rows }) => [
        source,
        rows,
      ]),
      [
        ["grid", 4],
        ["oe", 30],
      ],
    );
    assert.match(resolved.attribution, /Oracle's Elixir/);
    assert.match(resolved.attribution, /GRID/);
    assert.equal(
      resolved.source_provenance.overlap_precedence_status,
      "not_declared",
    );
    assert.equal(resolved.artifact_ids.model_pack_id, null);
    assert.equal(resolved.artifact_ids.one_clock_verified, false);
    assert.equal(resolved.artifact_ids.status, "not_declared");
  } finally {
    await f.cleanup();
  }
});

test("schema-2 provenance keeps canonical inclusion separate from GRID detail enrichment", async () => {
  const f = await fixture();
  try {
    const schemaTwo = manifest(
      "v-schema-two",
      "2026-07-27T12:30:00Z",
      {
        baseUrl: "https://blob.example/packs/v-schema-two",
        sourceCounts: { oe: 999, grid: 999 },
        sourceSummary: {
          schema_version: 2,
          sources: {
            canonical_map_inclusion: {
              oe: { rows: 120, maps: 120 },
              grid_gap_fill: { rows: 3, maps: 3 },
            },
            map_detail_enrichment: {
              grid_events: { rows: 9, maps: 9 },
            },
          },
          canonicalization: {
            overlap_precedence:
              "oracle_elixir_then_verified_grid_gap_fill",
            canonical_inclusion_field: "canonical_map_source",
            detail_enrichment_field: "map_detail_source",
          },
          attribution:
            "Oracle's Elixir provides canonical results; GRID provides verified gap fill and optional detail enrichment.",
        },
      },
    );
    const resolved = await readPackManifest({
      packRoot: f.root,
      fetchImpl: async () =>
        new Response(JSON.stringify(schemaTwo), { status: 200 }),
      now: () => 0,
    });

    assert.deepEqual(resolved.source_provenance.sources, [
      { source: "grid", rows: null },
      { source: "oe", rows: 120 },
    ]);
    assert.equal(
      resolved.source_provenance.attribution,
      schemaTwo.source_summary?.attribution,
    );
    assert.equal(
      resolved.source_provenance.overlap_precedence,
      "oracle_elixir_then_verified_grid_gap_fill",
    );
    assert.equal(
      resolved.source_provenance.canonicalization,
      "Canonical inclusion is declared by canonical_map_source; optional detail enrichment is separately declared by map_detail_source.",
    );
  } finally {
    await f.cleanup();
  }
});

test("first remote file failure atomically switches manifest and file to the bundled pack", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      fetchImpl: async () =>
        new Response(JSON.stringify(f.remote), { status: 200 }),
      now: () => 0,
    });
    const value = await readPackJson<{ bundle: string }>(resolved, FILE_A, {
      packRoot: f.root,
      fetchImpl: async () => new Response("unavailable", { status: 503 }),
    });

    assert.equal(value.bundle, "local");
    assert.equal(resolved.pack_id, f.local.pack_id);
    assert.equal(resolved.source, "bundled");
    assert.equal(resolved.degraded, true);
    assert.equal(resolved.degraded_reason, "remote_http_error");
  } finally {
    await f.cleanup();
  }
});

test("a later remote file failure fails closed instead of mixing bundles", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      fetchImpl: async () =>
        new Response(JSON.stringify(f.remote), { status: 200 }),
      now: () => 0,
    });
    let reads = 0;
    const fetchImpl: typeof fetch = async () => {
      reads += 1;
      return reads === 1
        ? new Response(JSON.stringify(REMOTE_FILES[FILE_A]), { status: 200 })
        : new Response("unavailable", { status: 503 });
    };
    assert.deepEqual(
      await readPackJson(resolved, FILE_A, { packRoot: f.root, fetchImpl }),
      { bundle: "remote" },
    );
    await assert.rejects(
      readPackJson(resolved, FILE_B, { packRoot: f.root, fetchImpl }),
      (error: unknown) =>
        error instanceof PackServiceError &&
        error.code === "PACK_BUNDLE_INTEGRITY",
    );
    assert.equal(resolved.pack_id, f.remote.pack_id);
  } finally {
    await f.cleanup();
  }
});

test("concurrent first reads serialize so a fallback cannot race a remote success", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      fetchImpl: async () =>
        new Response(JSON.stringify(f.remote), { status: 200 }),
      now: () => 0,
    });
    let remoteFileReads = 0;
    const fetchImpl: typeof fetch = async () => {
      remoteFileReads += 1;
      return new Response("unavailable", { status: 503 });
    };
    const [a, b] = await Promise.all([
      readPackJson<{ bundle: string; file: string }>(resolved, FILE_A, {
        packRoot: f.root,
        fetchImpl,
      }),
      readPackJson<{ bundle: string; file: string }>(resolved, FILE_B, {
        packRoot: f.root,
        fetchImpl,
      }),
    ]);
    assert.deepEqual([a, b], [
      { bundle: "local", file: "a" },
      { bundle: "local", file: "b" },
    ]);
    assert.equal(remoteFileReads, 1);
    assert.equal(resolved.pack_id, f.local.pack_id);
    assert.equal(resolved.source, "bundled");
  } finally {
    await f.cleanup();
  }
});

test("pack JSON paths cannot escape the immutable bundle root", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      configuredManifestUrl: null,
    });
    await assert.rejects(
      readPackJson(resolved, "../manifest.json", { packRoot: f.root }),
      (error: unknown) =>
        error instanceof PackServiceError &&
        error.code === "PACK_FILE_UNAVAILABLE",
    );
  } finally {
    await f.cleanup();
  }
});

test("bundled JSON bytes must match the manifest hash", async () => {
  const f = await fixture();
  try {
    await writeFile(
      path.join(f.root, f.local.pack_id, ...FILE_A.split("/")),
      JSON.stringify({ bundle: "tampered" }),
    );
    const resolved = await readPackManifest({
      packRoot: f.root,
      configuredManifestUrl: null,
    });
    await assert.rejects(
      readPackJson(resolved, FILE_A, { packRoot: f.root }),
      (error: unknown) =>
        error instanceof PackServiceError &&
        error.code === "PACK_BUNDLE_INTEGRITY",
    );
  } finally {
    await f.cleanup();
  }
});

test("a first remote hash mismatch atomically falls back to verified bundled bytes", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      fetchImpl: async () =>
        new Response(JSON.stringify(f.remote), { status: 200 }),
      now: () => 0,
    });
    const value = await readPackJson<{ bundle: string }>(resolved, FILE_A, {
      packRoot: f.root,
      fetchImpl: async () =>
        new Response(JSON.stringify({ bundle: "tampered" }), { status: 200 }),
    });
    assert.equal(value.bundle, "local");
    assert.equal(resolved.source, "bundled");
    assert.equal(resolved.degraded_reason, "remote_invalid");
  } finally {
    await f.cleanup();
  }
});

test("runtime and manifest hashes must reconcile for composition-style reads", async () => {
  const f = await fixture();
  try {
    const resolved = await readPackManifest({
      packRoot: f.root,
      configuredManifestUrl: null,
    });
    await assert.rejects(
      readPackJson(resolved, FILE_A, {
        packRoot: f.root,
        expectedSha256: "f".repeat(64),
      }),
      (error: unknown) =>
        error instanceof PackServiceError &&
        error.code === "PACK_BUNDLE_INTEGRITY",
    );
    assert.deepEqual(
      await readPackJson(resolved, FILE_A, {
        packRoot: f.root,
        expectedSha256: f.local.files[0].sha256,
      }),
      LOCAL_FILES[FILE_A],
    );
  } finally {
    await f.cleanup();
  }
});
