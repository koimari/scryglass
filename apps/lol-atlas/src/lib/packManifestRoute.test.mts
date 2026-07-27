import assert from "node:assert/strict";
import test from "node:test";
import { packManifestResponse } from "../app/api/pack-manifest/route.ts";
import {
  PackServiceError,
  type ResolvedPackManifest,
} from "./serverPack.ts";

test("pack manifest route preserves typed provenance and no-store semantics", async () => {
  const fixture = {
    pack_id: "v-test",
    schema_version: "test",
    created_utc: "2026-07-27T10:00:00Z",
    data_as_of: "2026-07-27T08:00:00Z",
    filters: { years: [2026], leagues: "test" },
    attribution: "Oracle's Elixir and GRID",
    excluded: [],
    base_url: null,
    total_bytes: 0,
    total_files: 0,
    files: [],
    source: "bundled",
    degraded: true,
    degraded_reason: "remote_unavailable",
    clocks: {
      publication: {
        value: "2026-07-27T10:00:00Z",
        field: "created_utc",
        status: "available",
      },
      data_through: {
        value: "2026-07-27T08:00:00Z",
        field: "data_as_of",
        status: "available",
      },
    },
    artifact_ids: {
      data_pack_id: "v-test",
      model_pack_id: null,
      strength_snapshot_sha256: null,
      calibration_sha256: null,
      one_clock_verified: false,
      status: "not_declared",
    },
    source_provenance: {
      sources: [
        { source: "grid", rows: 2 },
        { source: "oe", rows: 8 },
      ],
      attribution: "Oracle's Elixir and GRID",
      canonicalization: "fixture",
      overlap_precedence: "fixture",
      overlap_precedence_status: "declared",
    },
  } satisfies ResolvedPackManifest;

  const response = await packManifestResponse(async () => fixture);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("cache-control"), "no-store, max-age=0");
  const payload = (await response.json()) as ResolvedPackManifest;
  assert.equal(payload.source, "bundled");
  assert.equal(payload.degraded, true);
  assert.equal(payload.clocks.publication.value, "2026-07-27T10:00:00Z");
  assert.equal(payload.clocks.data_through.value, "2026-07-27T08:00:00Z");
});

test("pack manifest route returns a stable typed service error without exception leakage", async () => {
  const response = await packManifestResponse(async () => {
    throw new PackServiceError(
      "PACK_MANIFEST_INVALID",
      "secret filesystem path /private/data and upstream body",
    );
  });
  assert.equal(response.status, 503);
  assert.equal(response.headers.get("cache-control"), "no-store, max-age=0");
  const text = await response.text();
  assert.doesNotMatch(text, /private|upstream body|secret filesystem/);
  assert.deepEqual(JSON.parse(text), {
    ok: false,
    error: {
      code: "PACK_MANIFEST_INVALID",
      message: "Pack manifest is temporarily unavailable.",
    },
  });
});

test("unexpected route failures use the stable public availability code", async () => {
  const response = await packManifestResponse(async () => {
    throw new Error("database password and stack");
  });
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), {
    ok: false,
    error: {
      code: "PACK_MANIFEST_UNAVAILABLE",
      message: "Pack manifest is temporarily unavailable.",
    },
  });
});
