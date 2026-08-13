import assert from "node:assert/strict";
import test from "node:test";
import {
  fallbackRoute,
  executeTool,
  TOOLS,
  type ToolName,
} from "./supportChat.ts";

const EXAMPLES: Array<{ question: string; tool: ToolName; args: Record<string, string> }> = [
  { question: "who is the player with most A grade games", tool: "leaderboards", args: { category: "a_grades" } },
  { question: "what is the jungler with a rating of 1643", tool: "leaderboards", args: { category: "rating", role: "jng" } },
  { question: "show me T1's recent matches", tool: "matches", args: { team: "T1", limit: "10" } },
  { question: "what is the best mid laner this patch", tool: "leaderboards", args: { category: "rating", role: "mid", tier: "tier1", limit: "5" } },
  { question: "when does the next LEC game happen", tool: "schedule", args: { league: "LEC" } },
  { question: "how does the draft win share work", tool: "methodology", args: { topic: "draft" } },
  { question: "what is the top 1 team", tool: "leaderboards", args: { category: "teams", limit: "1" } },
  { question: "what is inspired's rating", tool: "player", args: { name: "inspired" } },
  { question: "what is faker rating", tool: "player", args: { name: "faker" } },
  { question: "who has better rating, inspired or faker?", tool: "compare_players", args: { player1: "inspired", player2: "faker" } },
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
  assert.deepEqual(names, ["leaderboards", "player", "compare_players", "team", "matches", "tier", "schedule", "methodology", "navigation"]);
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
