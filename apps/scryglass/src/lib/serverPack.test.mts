import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import {
  fetchVerifiedStorageAsset,
  publicPackManifest,
  readActivePublicAsset,
  readVerifiedAssetBytes,
  safeRelativePath,
  validatePublicManifest,
} from "./serverPack";
import type { PackManifest } from "./pack";

const RELEASE_ID = "v2026.08.13.183000";
const PATH = "features/team_records.json";
const RAW = new TextEncoder().encode('{"teams":{}}');
const SHA256 = createHash("sha256").update(RAW).digest("hex");

function manifest(): PackManifest & {
  release: { release_id: string; artifact_hashes: Record<string, string> };
} {
  return {
    pack_id: RELEASE_ID,
    schema_version: "scryglass:public-pack:v1",
    created_utc: "2026-08-13T18:30:00Z",
    filters: { years: [2026], leagues: "all" },
    attribution: "Scryglass",
    excluded: [],
    base_url: null,
    data_backend: "supabase",
    release: {
      release_id: RELEASE_ID,
      artifact_hashes: { [PATH]: SHA256 },
    },
    total_bytes: RAW.byteLength,
    total_files: 1,
    files: [{ path: PATH, bytes: RAW.byteLength, rows: 0, cols: 0, sha256: SHA256 }],
  };
}

test("public asset paths use the canonical allowlist", () => {
  assert.equal(safeRelativePath(PATH), PATH);
  assert.throws(() => safeRelativePath("../secrets.json"), /invalid/);
  assert.throws(() => safeRelativePath("features/private_training.json"), /invalid/);
});

test("public manifest removes internal fields and binds same-origin asset URLs", () => {
  const result = publicPackManifest(manifest());
  assert.equal(result.release_id, RELEASE_ID);
  assert.equal(result.files[0]?.url, `/api/assets/${RELEASE_ID}/features%2Fteam_records.json`);
  assert.deepEqual(Object.keys(result.files[0] ?? {}).sort(), ["bytes", "path", "sha256", "url"]);
});

test("unpromoted draft records stay outside the public manifest", () => {
  const candidate = manifest();
  const draftPath = "features/draft_records.json";
  const draftSha = "d".repeat(64);
  candidate.files.push({ path: draftPath, bytes: 10, rows: 0, cols: 0, sha256: draftSha });
  candidate.total_files += 1;
  candidate.total_bytes += 10;
  candidate.release.artifact_hashes[draftPath] = draftSha;
  assert.equal(publicPackManifest(candidate).files.some((file) => file.path === draftPath), false);
});

test("public manifest rejects release and digest conflicts", () => {
  const wrongRelease = manifest();
  wrongRelease.release.release_id = "v2026.08.13.183001";
  assert.throws(() => validatePublicManifest(wrongRelease), /release binding/);

  const wrongDigest = manifest();
  wrongDigest.release.artifact_hashes[PATH] = "0".repeat(64);
  assert.throws(() => validatePublicManifest(wrongDigest), /file inventory/);
});

test("asset lookup requires the active manifest and matching database metadata", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  const requests: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    requests.push(url);
    if (url.includes("scryglass_public_releases")) {
      return Response.json([{ release_id: RELEASE_ID, status: "active", manifest: manifest() }]);
    }
    return Response.json([{
      body: null,
      storage_path: `${RELEASE_ID}/${PATH}`,
      bytes: RAW.byteLength,
      sha256: SHA256,
      content_type: "application/json",
    }]);
  }) as typeof fetch;
  try {
    const asset = await readActivePublicAsset(RELEASE_ID, PATH);
    assert.equal(asset?.storagePath, `${RELEASE_ID}/${PATH}`);
    assert.equal(requests.length, 2);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("legacy inline assets must reproduce the published byte digest", async () => {
  const body = { teams: {} };
  const asset = {
    releaseId: RELEASE_ID,
    path: PATH,
    bytes: RAW.byteLength,
    sha256: SHA256,
    contentType: "application/json" as const,
    storagePath: null,
    body,
  };
  assert.deepEqual(await readVerifiedAssetBytes(asset), RAW);
  await assert.rejects(
    () => readVerifiedAssetBytes({ ...asset, sha256: "0".repeat(64) }),
    /integrity/,
  );
});

test("private Storage reads require matching custom metadata and bearer auth", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  const asset = {
    releaseId: RELEASE_ID,
    path: PATH,
    bytes: RAW.byteLength,
    sha256: SHA256,
    contentType: "application/json" as const,
    storagePath: `${RELEASE_ID}/${PATH}`,
    body: null,
  };
  let calls = 0;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls += 1;
    const url = String(input);
    const headers = new Headers(init?.headers);
    assert.equal(headers.get("authorization"), "Bearer sb_publishable_test");
    assert.match(url, /\/object\/(?:info\/)?authenticated\/scryglass-public\//);
    if (url.includes("/object/info/")) {
      return Response.json({
        size: RAW.byteLength,
        mimetype: "application/json",
        metadata: {
          bytes: RAW.byteLength,
          sha256: SHA256,
          content_type: "application/json",
        },
      });
    }
    return new Response(RAW, {
      headers: {
        "content-length": String(RAW.byteLength),
        "content-type": "application/json",
      },
    });
  }) as typeof fetch;
  try {
    const response = await fetchVerifiedStorageAsset(asset);
    assert.equal(new TextDecoder().decode(await response.arrayBuffer()), '{"teams":{}}');
    assert.equal(calls, 2);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});
