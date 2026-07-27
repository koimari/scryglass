import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  analyzeDraftSandbox,
  draftCatalog,
  draftScore,
  type DraftAction,
} from "./draftScore.ts";

const partialActions: DraftAction[] = [
  { side: "blue", champion: "Jarvan IV", role: "jng" },
  { side: "red", champion: "Ezreal", role: "bot" },
  { side: "red", champion: "Naafiri", role: "jng" },
  { side: "blue", champion: "Orianna", role: "mid" },
  { side: "blue", champion: "Jayce", role: "top" },
];

test("sandbox replays a partial sequence and ranks legal open-role responses", () => {
  const result = analyzeDraftSandbox({
    actions: partialActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
    limit: 20,
  });

  assert.equal(result.timeline.length, 5);
  assert.deepEqual(result.open_roles, ["top", "mid", "sup"]);
  assert.ok(result.current.projected_wr > 0 && result.current.projected_wr < 1);
  assert.ok(result.recommendations.length > 0);
  assert.ok(
    result.recommendations.every(
      (row) => row.role != null && result.open_roles.includes(row.role),
    ),
  );
  assert.ok(
    result.recommendations.every(
      (row) => !partialActions.some((action) => action.champion === row.champion),
    ),
  );
  assert.ok(
    result.recommendations.every(
      (row, index, rows) => index === 0 || rows[index - 1].projected_wr >= row.projected_wr,
    ),
  );
});

test("partial drafts apply a known same-role matchup term", () => {
  const withoutRoles = draftScore({
    blue: ["Orianna"],
    red: ["Akali"],
    league: "LCS",
  });
  const withRoles = draftScore({
    blue: ["Orianna"],
    red: ["Akali"],
    blue_roles: ["mid"],
    red_roles: ["mid"],
    league: "LCS",
  });

  assert.equal(withoutRoles.components.pair_logit, 0);
  assert.notEqual(withRoles.components.pair_logit, 0);
  assert.notEqual(withRoles.p_blue_draft, withoutRoles.p_blue_draft);
});

test("recommendations optimize the side whose turn it is", () => {
  const result = analyzeDraftSandbox({
    actions: [...partialActions, { side: "red", champion: "Rakan", role: "sup" }],
    perspective: "red",
    next_side: "blue",
    candidate_role: "open",
    league: "LCS",
  });

  assert.equal(result.perspective, "red");
  assert.equal(result.recommendation_side, "blue");
  assert.ok(result.recommendations[0].delta_pp > 0);
  assert.ok(result.recommendations[0].projected_wr > 1 - result.current.projected_wr);
});

test("mid-draft recommendations react to ally and enemy composition", () => {
  const base: DraftAction[] = [
    { side: "red", champion: "Renekton", role: "top" },
    { side: "red", champion: "Sejuani", role: "jng" },
    { side: "blue", champion: "Orianna", role: "mid" },
  ];
  const withJinx = analyzeDraftSandbox({
    actions: [...base, { side: "blue", champion: "Jinx", role: "bot" }],
    perspective: "blue",
    next_side: "blue",
    candidate_role: "sup",
    league: "LCS",
    limit: 10,
  });
  const withLucian = analyzeDraftSandbox({
    actions: [...base, { side: "blue", champion: "Lucian", role: "bot" }],
    perspective: "blue",
    next_side: "blue",
    candidate_role: "sup",
    league: "LCS",
    limit: 10,
  });

  assert.notDeepEqual(
    withJinx.recommendations.map((row) => row.champion),
    withLucian.recommendations.map((row) => row.champion),
  );
  assert.match(withJinx.current.score.note, /Partial full-composition counterfactual/);
});

test("sandbox catalog exposes the full composition pool and permits unseen legal roles", () => {
  const catalog = draftCatalog();
  assert.ok(catalog.length >= 160);
  const result = analyzeDraftSandbox({
    actions: [],
    perspective: "blue",
    next_side: "blue",
    candidate_role: "sup",
    league: "LCS",
    limit: 200,
  });
  const supportCandidates = new Map(result.recommendations.map((row) => [row.champion, row]));
  assert.equal(supportCandidates.size, catalog.length);
  assert.ok([...supportCandidates.values()].some((row) => row.evidence === "Unseen role"));
  assert.ok([...supportCandidates.values()].every((row) => row.role === "sup"));
});

test("excluded champions are removed from the legal recommendation set", () => {
  const baseline = analyzeDraftSandbox({
    actions: partialActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
  });
  const topChampion = baseline.recommendations[0].champion;
  const excluded = analyzeDraftSandbox({
    actions: partialActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    excluded: [topChampion],
    league: "LCS",
  });

  assert.ok(!excluded.recommendations.some((row) => row.champion === topChampion));
});

test("complete boards use the full-composition runtime and reconcile explanations", () => {
  const blue = ["Aatrox", "Sejuani", "Ahri", "Jinx", "Leona"];
  const red = ["Gnar", "Vi", "Orianna", "Xayah", "Nautilus"];
  const roles = ["top", "jng", "mid", "bot", "sup"];
  const result = draftScore({
    blue,
    red,
    blue_roles: roles,
    red_roles: roles,
    league: "LCK",
    patch: "16.13",
  });

  assert.match(result.note, /Full-composition draft model/);
  assert.equal(result.raw.source, "composition only; no roster/player strength");
  assert.equal(result.explanation?.reconciles, true);
  assert.equal(result.explanation?.champions.length, 10);
  const interval = result.uncertainty?.p_blue_95;
  assert.ok(interval);
  assert.ok(interval[0] >= 0);
  assert.ok(interval[1] <= 1);

  const swapped = draftScore({
    blue: red,
    red: blue,
    blue_roles: roles,
    red_roles: roles,
    league: "LCK",
    patch: "16.13",
  });
  assert.ok(Math.abs((result.components.composition_edge ?? 0) + (swapped.components.composition_edge ?? 0)) < 1e-5);
});

test("sandbox source contains no named public demo fixture", () => {
  const source = readFileSync(new URL("../components/DraftSandbox.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(source, /Example question|Load .*example/i);
  assert.match(source, /initialActions = \[\]/);
  assert.match(source, /sandbox-champion-grid/);
  assert.doesNotMatch(source, /list="sandbox-champions"/);
});
