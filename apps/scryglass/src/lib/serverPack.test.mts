import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import {
  e2eLocalPackRoot,
  fetchVerifiedStorageAsset,
  publicPackManifest,
  readActivePublicAsset,
  readActiveInlineAssetJson,
  readPackJson,
  readPrivateRefreshHealth,
  safeRelativePath,
  validatePublicManifest,
} from "./serverPack";
import type { PackManifest } from "./pack";
import { GET as getPublicAsset } from "../app/api/assets/[...path]/route";

const RELEASE_ID = "v2026.08.13.183000";
const PATH = "features/team_records.json";
const RAW = new TextEncoder().encode('{"teams":{}}');
const SHA256 = createHash("sha256").update(RAW).digest("hex");

test("local E2E data needs its exact flag and can never activate on Vercel", () => {
  assert.equal(e2eLocalPackRoot({}, "/workspace/app"), null);
  assert.equal(
    e2eLocalPackRoot({ SCRYGLASS_E2E_LOCAL_PACK: "1" }, "/workspace/app"),
    "/workspace/app/output/playwright/e2e-pack",
  );
  for (const key of ["VERCEL", "VERCEL_ENV", "VERCEL_URL", "VERCEL_REGION"]) {
    assert.throws(
      () => e2eLocalPackRoot({ SCRYGLASS_E2E_LOCAL_PACK: "1", [key]: "set" }, "/workspace/app"),
      /disabled on Vercel/,
    );
  }
});

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
  let returnMissingStorage = false;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    requests.push(url);
    if (url.includes("get_scryglass_active_release")) {
      return Response.json([{ release_id: RELEASE_ID, status: "active", manifest: manifest() }]);
    }
    return Response.json([{
      storage_path: returnMissingStorage ? null : `${RELEASE_ID}/${PATH}`,
      bytes: RAW.byteLength,
      sha256: SHA256,
      content_type: "application/json",
    }]);
  }) as typeof fetch;
  try {
    const asset = await readActivePublicAsset(RELEASE_ID, PATH);
    assert.equal(asset?.storagePath, `${RELEASE_ID}/${PATH}`);
    assert.equal(requests.length, 2);
    returnMissingStorage = true;
    assert.equal(await readActivePublicAsset(RELEASE_ID, PATH), null);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("temporary inline compatibility returns parsed active data but never asset bytes", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  let inlineCalls = 0;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("get_scryglass_active_release")) {
      return Response.json([{ release_id: RELEASE_ID, status: "active", manifest: manifest() }]);
    }
    if (url.includes("get_scryglass_active_asset")) return Response.json([]);
    if (url.includes("get_scryglass_active_inline_asset")) {
      inlineCalls += 1;
      return Response.json([{
        release_id: RELEASE_ID,
        path: PATH,
        body: { teams: {} },
        source_bytes: RAW.byteLength,
        source_sha256: SHA256,
        content_type: "application/json",
      }]);
    }
    throw new Error(`unexpected request: ${url}`);
  }) as typeof fetch;
  try {
    const inline = await readActiveInlineAssetJson<{ teams: Record<string, unknown> }>(RELEASE_ID, PATH);
    assert.deepEqual(inline?.body, { teams: {} });
    assert.equal(inline?.sourceBytes, RAW.byteLength);
    assert.equal(inline?.sourceSha256, SHA256);
    assert.equal(inlineCalls, 1);

    inlineCalls = 0;
    const response = await getPublicAsset(
      new Request(`https://scryglass.xyz/api/assets/${RELEASE_ID}/${PATH}`),
      { params: Promise.resolve({ path: [RELEASE_ID, PATH] }) },
    );
    assert.equal(response.status, 404);
    assert.equal(response.headers.get("etag"), null);
    assert.equal(inlineCalls, 0);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("temporary inline compatibility rejects superseded release rows", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  let inlineCalls = 0;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("get_scryglass_active_release")) {
      return Response.json([{ release_id: RELEASE_ID, status: "superseded", manifest: manifest() }]);
    }
    if (url.includes("get_scryglass_active_asset")) return Response.json([]);
    if (url.includes("get_scryglass_active_inline_asset")) {
      inlineCalls += 1;
      return Response.json([{
        release_id: RELEASE_ID,
        path: PATH,
        body: { teams: {} },
        source_bytes: RAW.byteLength,
        source_sha256: SHA256,
        content_type: "application/json",
      }]);
    }
    throw new Error(`unexpected request: ${url}`);
  }) as typeof fetch;
  try {
    assert.equal(await readActiveInlineAssetJson(RELEASE_ID, PATH), null);
    assert.equal(inlineCalls, 0);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("temporary inline compatibility rejects the wrong parsed root schema", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("get_scryglass_active_release")) {
      return Response.json([{ release_id: RELEASE_ID, status: "active", manifest: manifest() }]);
    }
    if (url.includes("get_scryglass_active_inline_asset")) {
      return Response.json([{
        release_id: RELEASE_ID,
        path: PATH,
        body: [],
        source_bytes: RAW.byteLength,
        source_sha256: SHA256,
        content_type: "application/json",
      }]);
    }
    throw new Error(`unexpected request: ${url}`);
  }) as typeof fetch;
  try {
    assert.equal(await readActiveInlineAssetJson(RELEASE_ID, PATH), null);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("temporary inline compatibility enforces the bounded RPC body budget", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("get_scryglass_active_release")) {
      return Response.json([{ release_id: RELEASE_ID, status: "active", manifest: manifest() }]);
    }
    if (url.includes("get_scryglass_active_inline_asset")) {
      return Response.json([{
        release_id: RELEASE_ID,
        path: PATH,
        body: { padding: "x".repeat(500 * 1024) },
        source_bytes: RAW.byteLength,
        source_sha256: SHA256,
        content_type: "application/json",
      }]);
    }
    throw new Error(`unexpected request: ${url}`);
  }) as typeof fetch;
  try {
    assert.equal(await readActiveInlineAssetJson(RELEASE_ID, PATH), null);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("private Storage reads require matching custom metadata and publishable-key auth", async () => {
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
  };
  let calls = 0;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls += 1;
    const url = String(input);
    const headers = new Headers(init?.headers);
    assert.equal(headers.get("apikey"), "sb_publishable_test");
    assert.equal(headers.get("authorization"), null);
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

test("Supabase asset reads never resolve a local pack path", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  const previousE2e = process.env.SCRYGLASS_E2E_LOCAL_PACK;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  delete process.env.SCRYGLASS_E2E_LOCAL_PACK;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("get_scryglass_active_release")) {
      return Response.json([{
        release_id: RELEASE_ID,
        status: "active",
        manifest: manifest(),
      }]);
    }
    if (url.includes("get_scryglass_active_asset")) {
      return Response.json([{
        storage_path: `${RELEASE_ID}/${PATH}`,
        bytes: RAW.byteLength,
        sha256: SHA256,
        content_type: "application/json",
      }]);
    }
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
    assert.deepEqual(await readPackJson(manifest(), PATH), { teams: {} });
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
    if (previousE2e === undefined) delete process.env.SCRYGLASS_E2E_LOCAL_PACK;
    else process.env.SCRYGLASS_E2E_LOCAL_PACK = previousE2e;
  }
});

test("private health sends its diagnostic token in a POST body", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  const previousToken = process.env.SCRYGLASS_DIAGNOSTIC_TOKEN;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  process.env.SCRYGLASS_DIAGNOSTIC_TOKEN = "d".repeat(64);
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    assert.equal(String(input), "https://abcdef.supabase.co/rest/v1/rpc/get_scryglass_private_health");
    assert.equal(init?.method, "POST");
    assert.equal(new Headers(init?.headers).get("content-type"), "application/json");
    assert.deepEqual(JSON.parse(String(init?.body)), { p_token: "d".repeat(64) });
    assert.doesNotMatch(String(input), /d{32}/);
    return Response.json([{
      status: "ok",
      refresh_status: "idle",
      checked_at: "2026-08-13T18:30:00Z",
      last_success_at: "2026-08-13T18:30:00Z",
      source_as_of: "2026-08-13T18:00:00Z",
      active_release_id: RELEASE_ID,
      last_run_id: "run-1",
      worker_commit: "a".repeat(40),
      stale: false,
    }]);
  }) as typeof fetch;

  try {
    assert.equal((await readPrivateRefreshHealth())?.last_run_id, "run-1");
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
    if (previousToken === undefined) delete process.env.SCRYGLASS_DIAGNOSTIC_TOKEN;
    else process.env.SCRYGLASS_DIAGNOSTIC_TOKEN = previousToken;
  }
});

test("asset route streams an asset above 4.5 MB without buffering it", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";

  const largeBytes = 5 * 1024 * 1024;
  const largeSha256 = "a".repeat(64);
  const largeManifest = manifest();
  largeManifest.total_bytes = largeBytes;
  largeManifest.files[0]!.bytes = largeBytes;
  largeManifest.files[0]!.sha256 = largeSha256;
  largeManifest.release.artifact_hashes[PATH] = largeSha256;

  let emittedBytes = 0;
  let arrayBufferCalls = 0;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("get_scryglass_active_release")) {
      return Response.json([{
        release_id: RELEASE_ID,
        status: "active",
        manifest: largeManifest,
      }]);
    }
    if (url.includes("get_scryglass_active_asset")) {
      return Response.json([{
        storage_path: `${RELEASE_ID}/${PATH}`,
        bytes: largeBytes,
        sha256: largeSha256,
        content_type: "application/json",
      }]);
    }
    if (url.includes("/object/info/")) {
      return Response.json({
        size: largeBytes,
        mimetype: "application/json",
        metadata: {
          bytes: largeBytes,
          sha256: largeSha256,
          content_type: "application/json",
        },
      });
    }

    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (emittedBytes === largeBytes) {
          controller.close();
          return;
        }
        const size = Math.min(64 * 1024, largeBytes - emittedBytes);
        emittedBytes += size;
        controller.enqueue(new Uint8Array(size));
      },
    });
    const response = new Response(body, {
      headers: {
        "content-length": String(largeBytes),
        "content-type": "application/json",
      },
    });
    Object.defineProperty(response, "arrayBuffer", {
      value: async () => {
        arrayBufferCalls += 1;
        throw new Error("the route must not buffer the Storage response");
      },
    });
    return response;
  }) as typeof fetch;

  try {
    const response = await getPublicAsset(
      new Request(`https://scryglass.xyz/api/assets/${RELEASE_ID}/${PATH}`),
      { params: Promise.resolve({ path: [RELEASE_ID, PATH] }) },
    );
    assert.equal(response.status, 200);
    assert.ok(response.body instanceof ReadableStream);
    assert.equal(response.headers.get("content-length"), String(largeBytes));
    assert.equal(response.headers.get("content-type"), "application/json");
    assert.equal(response.headers.get("etag"), `"${largeSha256}"`);
    assert.equal(arrayBufferCalls, 0);
    assert.ok(emittedBytes < largeBytes);

    const reader = response.body.getReader();
    let receivedBytes = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      receivedBytes += value.byteLength;
    }
    assert.equal(receivedBytes, largeBytes);
    assert.equal(arrayBufferCalls, 0);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("asset route fails before opening a Storage body when immutable metadata differs", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  let storageBodyRequests = 0;

  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("get_scryglass_active_release")) {
      return Response.json([{ release_id: RELEASE_ID, status: "active", manifest: manifest() }]);
    }
    if (url.includes("get_scryglass_active_asset")) {
      return Response.json([{
        storage_path: `${RELEASE_ID}/${PATH}`,
        bytes: RAW.byteLength,
        sha256: SHA256,
        content_type: "application/json",
      }]);
    }
    if (url.includes("/object/info/")) {
      return Response.json({
        size: RAW.byteLength + 1,
        mimetype: "application/json",
        metadata: {
          bytes: RAW.byteLength,
          sha256: SHA256,
          content_type: "application/json",
        },
      });
    }
    storageBodyRequests += 1;
    return new Response(RAW);
  }) as typeof fetch;

  try {
    const response = await getPublicAsset(
      new Request(`https://scryglass.xyz/api/assets/${RELEASE_ID}/${PATH}`),
      { params: Promise.resolve({ path: [RELEASE_ID, PATH] }) },
    );
    assert.equal(response.status, 502);
    assert.equal(response.headers.get("cache-control"), "private, no-store");
    assert.equal(storageBodyRequests, 0);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});
