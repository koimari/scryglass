import assert from "node:assert/strict";
import test from "node:test";
import {
  fallbackRoute,
  executeTool,
  routeQuestion,
  TOOLS,
  type ToolName,
} from "./supportChat";

const EXAMPLES: Array<{ question: string; tool: ToolName; args: Record<string, string> }> = [
  { question: "who is the player with most A grade games", tool: "query_players", args: { q: "who is the player with most A grade games" } },
  { question: "what is the jungler with a rating of 1643", tool: "leaderboards", args: { category: "rating", role: "jng" } },
  { question: "show me T1's recent matches", tool: "matches", args: { team: "T1", limit: "10" } },
  { question: "what is the best mid laner this patch", tool: "query_players", args: { q: "what is the best mid laner this patch" } },
  { question: "when does the next LEC game happen", tool: "schedule", args: { league: "LEC" } },
  { question: "how does the draft win share work", tool: "methodology", args: { topic: "draft" } },
  { question: "what is the top 1 team", tool: "leaderboards", args: { category: "teams", limit: "1" } },
  { question: "what is inspired's rating", tool: "query_players", args: { q: "what is inspired's rating" } },
  { question: "what is faker rating", tool: "query_players", args: { q: "what is faker rating" } },
  { question: "who has better rating, inspired or faker?", tool: "query_players", args: { q: "who has better rating, inspired or faker?" } },
  { question: "who is the best Galio player", tool: "query_players", args: { q: "who is the best Galio player" } },
  { question: "what is Inspired's worst champion?", tool: "query_players", args: { q: "what is Inspired's worst champion?" } },
  { question: "what is Faker's most median performance champion?", tool: "query_players", args: { q: "what is Faker's most median performance champion?" } },
  { question: "what is the median performance champion for Faker?", tool: "query_players", args: { q: "what is the median performance champion for Faker?" } },
  { question: "what is Faker's average performance champion?", tool: "query_players", args: { q: "what is Faker's average performance champion?" } },
  { question: "what is the worst champion in general?", tool: "query_champions", args: { q: "what is the worst champion in general?" } },
  { question: "which team has the best draft score", tool: "leaderboards", args: { category: "teams_draft", tier: "tier1", limit: "10" } },
  { question: "which team has the best draft", tool: "leaderboards", args: { category: "teams_draft", tier: "tier1", limit: "10" } },
  { question: "who has the best draft between KC and G2?", tool: "query_drafts", args: { q: "who has the best draft between KC and G2?" } },
  { question: "bottom 5 team draft scores", tool: "leaderboards", args: { category: "teams_draft", tier: "tier1", limit: "10" } },
  { question: "best Tier 1 LCK mid with at least 100 games", tool: "query_players", args: { q: "best Tier 1 LCK mid with at least 100 games" } },
  { question: "what is the best mid champion this patch", tool: "tier", args: { role: "mid" } },
];

test("fallback router handles example questions from every domain", () => {
  for (const example of EXAMPLES) {
    const result = fallbackRoute(example.question);
    assert.ok("call" in result, `${example.question} -> expected a tool call`);
    assert.equal(result.call.tool, example.tool, example.question);
    for (const [key, value] of Object.entries(example.args)) {
      assert.equal(result.call.args[key], value, `${example.question} arg ${key}`);
    }
  }
});

test("fallback router routes methodology by topic", () => {
  const rating = fallbackRoute("how is the rating computed?");
  assert.ok("call" in rating && rating.call.tool === "methodology");
  assert.equal(rating.call.args.topic, "ratings");

  const grades = fallbackRoute("what does an A grade mean?");
  assert.ok("call" in grades && grades.call.tool === "methodology");
  assert.equal(grades.call.args.topic, "grades");
});

test("fallback router routes team and match questions", () => {
  const team = fallbackRoute("tell me about team Fnatic");
  assert.ok("call" in team && team.call.tool === "team");
  assert.equal(team.call.args.name, "Fnatic");

  const matches = fallbackRoute("recent LCK games");
  assert.ok("call" in matches && matches.call.tool === "matches");
  assert.equal(matches.call.args.league, "LCK");
});

test("fallback router returns an explanation for off-topic input", () => {
  const result = fallbackRoute("what is the meaning of life?");
  assert.ok("explanation" in result);
});

test("the tool schema covers each supported domain with unique names", () => {
  const names = TOOLS.map((tool) => tool.name);
  assert.equal(new Set(names).size, names.length);
  // player_stats and team_stats serve the profile Stats sections, so chat
  // questions about per-game cs/min, gold share or damage share can reach the
  // published statistics instead of routing to the profile summary tools.
  assert.deepEqual(names, ["query_players", "query_champions", "query_drafts", "leaderboards", "player", "player_stats", "compare_players", "team", "team_stats", "matches", "tier", "schedule", "methodology", "navigation"]);
});

test("high-confidence ranking routes bypass model inference", async () => {
  assert.deepEqual(await routeQuestion("which team has the best draft score"), {
    call: { tool: "leaderboards", args: { category: "teams_draft", tier: "tier1", limit: "10" } },
  });
  assert.deepEqual(await routeQuestion("which team has the best draft"), {
    call: { tool: "leaderboards", args: { category: "teams_draft", tier: "tier1", limit: "10" } },
  });
  assert.deepEqual(await routeQuestion("who are the best players by draft"), {
    call: { tool: "leaderboards", args: { category: "players_draft", tier: "tier1", limit: "10" } },
  });
  assert.deepEqual(await routeQuestion("who has the best draft between KC and G2?"), {
    call: { tool: "query_drafts", args: { q: "who has the best draft between KC and G2?" } },
  });
  assert.deepEqual(await routeQuestion("what is Inspired's worst champion?"), {
    call: { tool: "query_players", args: { q: "what is Inspired's worst champion?" } },
  });
  assert.deepEqual(await routeQuestion("what is Faker's most median performance champion?"), {
    call: { tool: "query_players", args: { q: "what is Faker's most median performance champion?" } },
  });
  assert.deepEqual(await routeQuestion("what is the median performance champion for Faker?"), {
    call: { tool: "query_players", args: { q: "what is the median performance champion for Faker?" } },
  });
  assert.deepEqual(await routeQuestion("what is Faker's average performance champion?"), {
    call: { tool: "query_players", args: { q: "what is Faker's average performance champion?" } },
  });
  assert.deepEqual(await routeQuestion("what is the worst champion in general?"), {
    call: { tool: "query_champions", args: { q: "what is the worst champion in general?" } },
  });
});

test("routing rejects questions above the public request budget", async () => {
  const result = await routeQuestion("x".repeat(501));
  assert.deepEqual(result, { explanation: "Questions can contain up to 500 characters." });
});

test("executeTool builds the right endpoint and parses the response", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    assert.equal(url, "/api/chat/tier?role=mid");
    return new Response(JSON.stringify({ ok: true, data: { patch: "16.15", rows: [] } }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  try {
    const data = await executeTool({ tool: "tier", args: { role: "mid" } });
    assert.deepEqual(data, { patch: "16.15", rows: [] });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("executeTool preserves the original question for aggregate champion queries", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    assert.equal(url, "/api/chat/query_champions?q=what+is+the+worst+champion+in+general%3F");
    return new Response(JSON.stringify({ ok: true, data: { kind: "champion_query", rows: [] } }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  try {
    const data = await executeTool({ tool: "query_champions", args: { q: "what is the worst champion in general?" } });
    assert.deepEqual(data, { kind: "champion_query", rows: [] });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("executeTool surfaces a non-ok response as an error", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ ok: false, reason: "boom" }), { status: 404 })) as typeof fetch;
  try {
    await assert.rejects(() => executeTool({ tool: "player", args: { name: "Nobody" } }), /boom/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
