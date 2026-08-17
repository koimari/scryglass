import assert from "node:assert/strict";
import test from "node:test";

import {
  boundedRowLimit,
  getMatchFacets,
  getMatches,
  getPlayerProfile,
  getRatings,
  getTierFacets,
  getTierScope,
  validatePromotedDraftScoreResult,
  validatePublicDraftResponse,
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

function descriptiveManifest(): PackManifest {
  return {
    ...queryManifest(),
    draft_authority: {
      schema_version: "scryglass:draft-authority:v1",
      status: "descriptive",
      authority: "descriptive",
      release_id: RELEASE_ID,
      model_version: "draft-descriptive-v1",
      artifact_sha256: "c".repeat(64),
      receipt_sha256: "a".repeat(64),
      issued_utc: "2026-08-13T18:31:17Z",
    },
  };
}

function promotedManifest(): PackManifest {
  return {
    ...queryManifest(),
    draft_authority: {
      schema_version: "scryglass:draft-authority:v1",
      status: "promoted",
      authority: "promoted",
      release_id: RELEASE_ID,
      model_version: "public-draft-score-v1",
      artifact_sha256: "d".repeat(64),
      receipt_sha256: "b".repeat(64),
      issued_utc: "2026-08-16T22:00:00Z",
    },
  };
}

test("public query row limits stay inside the chat response budget", () => {
  assert.equal(boundedRowLimit(undefined, 20), 20);
  assert.equal(boundedRowLimit(1, 20), 1);
  assert.equal(boundedRowLimit(20, 20), 20);
  assert.equal(boundedRowLimit(50, 20), 20);
  assert.equal(boundedRowLimit(50, 20, 100), 50);
  assert.equal(boundedRowLimit(500, 20, 100), 100);
  assert.equal(boundedRowLimit(0, 20), 1);
});

test("descriptive Draft responses accept model-unit fields and reject probability fields", () => {
  const manifest = descriptiveManifest();
  assert.doesNotThrow(() => validatePublicDraftResponse({
    authority: "descriptive",
    draft_metric: { draft_edge: 0.25, games: 12 },
  }, manifest));
  assert.throws(() => validatePublicDraftResponse({
    authority: "descriptive",
    draft_metric: { draft_win_share: 0.61 },
  }, manifest), /probability fields/);
  assert.throws(() => validatePublicDraftResponse({
    authority: "unavailable",
    draft_metric: { draft_edge: 0.25 },
  }, manifest), /unbound authority/);
});

test("promoted Draft responses allow probability and recommendation but never betting fields", () => {
  const manifest = promotedManifest();
  assert.doesNotThrow(() => validatePublicDraftResponse({
    authority: "promoted",
    match_win_probability: { Blue: 0.61, Red: 0.39 },
    controlled_draft_score: { edge_percentage_points: 2.4 },
    side_recommendation: "Blue",
  }, manifest));
  for (const field of ["odds", "fair_odds", "expected_value", "ev", "stake", "wager", "betting"]) {
    assert.throws(() => validatePublicDraftResponse({
      authority: "promoted",
      [field]: 1,
    }, manifest), /permanently forbidden betting field/);
  }
  assert.throws(() => validatePublicDraftResponse({
    authority: "descriptive",
    side_recommendation: "Blue",
  }, descriptiveManifest()), /recommendation without promoted authority/);
});

test("promoted Draft result keeps match probability and controlled draft evidence separate", () => {
  const manifest = promotedManifest();
  const result = {
    schema_version: "scryglass:public-draft-score-result:v1",
    authority: "promoted",
    release_id: RELEASE_ID,
    model_version: "public-draft-score-v1",
    receipt_sha256: "b".repeat(64),
    evidence_window: {
      start: "2025-01-01T00:00:00Z",
      end: "2026-08-16T00:00:00Z",
    },
    match_win_probability: { Blue: 0.61, Red: 0.39 },
    controlled_draft_score: {
      model_units: -0.18,
      edge_percentage_points: -1.9,
      stronger_draft: "Red",
      explanation: "Composition contribution with strength controls held fixed.",
    },
    side_recommendation: "Blue",
  };
  assert.doesNotThrow(() => validatePromotedDraftScoreResult(result, manifest));
  assert.throws(() => validatePromotedDraftScoreResult({
    ...result,
    match_win_probability: { Blue: 0.61, Red: 0.41 },
  }, manifest), /probabilities are invalid/);
  assert.throws(() => validatePromotedDraftScoreResult({
    ...result,
    side_recommendation: "Red",
  }, manifest), /recommendation conflicts/);
  assert.throws(() => validatePromotedDraftScoreResult({
    ...result,
    release_id: "v2026.08.16.999999",
  }, manifest), /not release-bound/);
  assert.throws(() => validatePromotedDraftScoreResult({
    ...result,
    controlled_draft_score: {
      ...result.controlled_draft_score,
      stronger_draft: "Blue",
    },
  }, manifest), /direction is inconsistent/);
  assert.throws(() => validatePromotedDraftScoreResult({
    ...result,
    receipt_sha256: "f".repeat(64),
  }, manifest), /not release-bound/);
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

test("player profiles reuse release-bound cache entries", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  let calls = 0;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    calls += 1;
    assert.equal(String(input), "https://abcdef.supabase.co/rest/v1/rpc/get_scryglass_player_profile");
    return Response.json({
      schema_version: "scryglass:query-api:v1",
      release_id: RELEASE_ID,
      row: null,
      team_row: null,
      champions: [],
      recent_games: [],
      champion_images: {},
      standing: null,
    });
  }) as typeof fetch;
  try {
    const first = await getPlayerProfile(queryManifest(), "Cache Probe");
    const second = await getPlayerProfile(queryManifest(), " cache probe ");
    assert.strictEqual(first, second);
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("player profile cache removes release-mismatched responses", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  let calls = 0;
  globalThis.fetch = (async () => {
    calls += 1;
    return Response.json({
      schema_version: "scryglass:query-api:v1",
      release_id: calls === 1 ? "v2026.08.13.183001" : RELEASE_ID,
      row: null,
      team_row: null,
      champions: [],
      recent_games: [],
      champion_images: {},
      standing: null,
    });
  }) as typeof fetch;
  try {
    await assert.rejects(
      getPlayerProfile(queryManifest(), "Rejected Cache Probe"),
      /different release/,
    );
    await getPlayerProfile(queryManifest(), "Rejected Cache Probe");
    assert.equal(calls, 2);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("player profile cache entries expire after the short TTL", async () => {
  const previousFetch = globalThis.fetch;
  const previousNow = Date.now;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  let now = 2_000_000;
  Date.now = () => now;
  let calls = 0;
  globalThis.fetch = (async () => {
    calls += 1;
    return Response.json({
      schema_version: "scryglass:query-api:v1",
      release_id: RELEASE_ID,
      row: null,
      team_row: null,
      champions: [],
      recent_games: [],
      champion_images: {},
      standing: null,
    });
  }) as typeof fetch;
  try {
    await getPlayerProfile(queryManifest(), "TTL Cache Probe");
    await getPlayerProfile(queryManifest(), "ttl cache probe");
    assert.equal(calls, 1);
    now += 30_001;
    await getPlayerProfile(queryManifest(), "TTL Cache Probe");
    assert.equal(calls, 2);
  } finally {
    Date.now = previousNow;
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("player profile cache evicts entries when serialized memory exceeds the bound", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  const largeImage = "x".repeat(270_000);
  let calls = 0;
  globalThis.fetch = (async () => {
    calls += 1;
    return Response.json({
      schema_version: "scryglass:query-api:v1",
      release_id: RELEASE_ID,
      row: null,
      team_row: null,
      champions: [],
      recent_games: [],
      champion_images: { CachePressure: largeImage },
      standing: null,
    });
  }) as typeof fetch;
  try {
    for (let index = 0; index < 33; index += 1) {
      await getPlayerProfile(queryManifest(), `Cache Pressure ${index}`);
    }
    await getPlayerProfile(queryManifest(), "Cache Pressure 0");
    assert.equal(calls, 34);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("ratings list can request the full bounded page", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body)) as { p_limit: number };
    assert.equal(body.p_limit, 100);
    return Response.json({
      schema_version: "scryglass:query-api:v1",
      release_id: RELEASE_ID,
      rows: [],
      limit: 100,
      offset: 0,
      total: 0,
    });
  }) as typeof fetch;
  try {
    const result = await getRatings(queryManifest(), {
      kind: "teams",
      tiers: ["tier1"],
      limit: 100,
    });
    assert.equal(result.limit, 100);
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
