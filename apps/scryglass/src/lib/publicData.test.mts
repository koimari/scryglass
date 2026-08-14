import assert from "node:assert/strict";
import test from "node:test";

import {
  boundedRowLimit,
  getMatchFacets,
  getMatches,
  getRatings,
  getTierFacets,
  getTierScope,
} from "./publicData";
import type { PackManifest } from "./pack";

const RELEASE_ID = "v2026.08.13.183000";

function queryManifest(): PackManifest {
  return {
    pack_id: RELEASE_ID,
    schema_version: "scryglass:public-pack:v1",
    created_utc: "2026-08-13T18:30:00Z",
    filters: { years: [2026], leagues: "all" },
    attribution: "Scryglass",
    excluded: [],
    base_url: null,
    data_backend: "supabase",
    query_api: { schema_version: "scryglass:query-api:v1", status: "available" },
    total_bytes: 0,
    total_files: 0,
    files: [],
  };
}

test("public query row limits stay inside the chat response budget", () => {
  assert.equal(boundedRowLimit(undefined, 20), 20);
  assert.equal(boundedRowLimit(1, 20), 1);
  assert.equal(boundedRowLimit(20, 20), 20);
  assert.equal(boundedRowLimit(50, 20), 20);
  assert.equal(boundedRowLimit(0, 20), 1);
});

test("ratings RPC uses publishable auth and caps exact-name comparisons at 20 rows", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    assert.equal(String(input), "https://abcdef.supabase.co/rest/v1/rpc/get_scryglass_ratings");
    const headers = new Headers(init?.headers);
    assert.equal(headers.get("apikey"), "sb_publishable_test");
    assert.equal(headers.get("authorization"), null);
    const body = JSON.parse(String(init?.body)) as { p_limit: number; p_names: string[] };
    assert.equal(body.p_limit, 20);
    assert.equal(body.p_names.length, 20);
    return Response.json({
      schema_version: "scryglass:query-api:v1",
      release_id: RELEASE_ID,
      rows: [],
      limit: 20,
      offset: 0,
      total: 0,
    });
  }) as typeof fetch;
  try {
    const result = await getRatings(queryManifest(), {
      kind: "players",
      names: Array.from({ length: 30 }, (_, index) => `Player ${index}`),
      limit: 500,
    });
    assert.equal(result.limit, 20);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("match RPCs preserve the bounded page, date window, and facet filters", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  const calls: Array<{ name: string; body: Record<string, unknown> }> = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const name = String(input).split("/").at(-1) ?? "";
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    calls.push({ name, body });
    if (name === "get_scryglass_match_facets") {
      return Response.json({
        schema_version: "scryglass:query-api:v1",
        release_id: RELEASE_ID,
        tiers: ["tier1"],
        years: [2026],
        months: ["2026-08"],
        teams: ["T1"],
        leagues: ["LCK"],
      });
    }
    return Response.json({
      schema_version: "scryglass:query-api:v1",
      release_id: RELEASE_ID,
      rows: [],
      limit: 20,
      offset: 40,
      total: 0,
      champion_images: { Galio: "https://cdn.communitydragon.org/latest/champion/3/square" },
    });
  }) as typeof fetch;
  try {
    const matches = await getMatches(queryManifest(), {
      leagues: ["LCK"],
      tiers: ["tier1"],
      team: "T1",
      years: [2026],
      from: "2026-08-01",
      to: "2026-09-01",
      limit: 100,
      offset: 40,
    });
    assert.equal(matches.champion_images.Galio, "https://cdn.communitydragon.org/latest/champion/3/square");
    const facets = await getMatchFacets(queryManifest(), {
      tiers: ["tier1"],
      years: [2026],
      from: "2026-08-01",
      to: "2026-09-01",
      team: "T1",
    });
    assert.deepEqual(facets.years, [2026]);
    assert.deepEqual(calls, [
      {
        name: "get_scryglass_matches",
        body: {
          p_leagues: ["LCK"],
          p_tiers: ["tier1"],
          p_team: "T1",
          p_champion: null,
          p_years: [2026],
          p_from: "2026-08-01",
          p_to: "2026-09-01",
          p_before: null,
          p_limit: 20,
          p_offset: 40,
        },
      },
      {
        name: "get_scryglass_match_facets",
        body: {
          p_tiers: ["tier1"],
          p_years: [2026],
          p_from: "2026-08-01",
          p_to: "2026-09-01",
          p_team: "T1",
        },
      },
    ]);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("tier RPCs load facets and one filtered scope", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  const calls: Array<{ name: string; body: Record<string, unknown> }> = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const name = String(input).split("/").at(-1) ?? "";
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    calls.push({ name, body });
    if (name === "get_scryglass_tier_facets") {
      return Response.json({
        schema_version: "scryglass:query-api:v1",
        release_id: RELEASE_ID,
        options: {
          patches: ["16.15"], roles: ["mid"], regions: ["LCK"],
          leagues: ["LCK"], tiers: ["tier1"], tier_buckets: ["A"],
        },
        scopes: [],
      });
    }
    return Response.json({
      schema_version: "scryglass:query-api:v1",
      release_id: RELEASE_ID,
      scope: null,
      rows: [],
      structural_similarity: null,
      champion_images: {},
    });
  }) as typeof fetch;
  try {
    const facets = await getTierFacets(queryManifest());
    const scope = await getTierScope(queryManifest(), {
      patch: "16.15",
      role: "mid",
      region: "LCK",
      league: "LCK",
      tier: "tier1",
      similarityLimit: 500,
    });
    assert.deepEqual(facets.options.regions, ["LCK"]);
    assert.deepEqual(scope.rows, []);
    assert.deepEqual(calls, [
      { name: "get_scryglass_tier_facets", body: {} },
      {
        name: "get_scryglass_tier_scope",
        body: {
          p_patch: "16.15",
          p_role: "mid",
          p_region: "LCK",
          p_league: "LCK",
          p_tier: "tier1",
          p_similarity_limit: 100,
        },
      },
    ]);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});
