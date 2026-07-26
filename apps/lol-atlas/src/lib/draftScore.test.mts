import assert from "node:assert/strict";
import test from "node:test";
import {
  analyzeDraftSandbox,
  draftScore,
  type DraftAction,
} from "./draftScore.ts";

const dayosActions: DraftAction[] = [
  { side: "blue", champion: "Jarvan IV", role: "jng" },
  { side: "red", champion: "Ezreal", role: "bot" },
  { side: "red", champion: "Naafiri", role: "jng" },
  { side: "blue", champion: "Orianna", role: "mid" },
  { side: "blue", champion: "Jayce", role: "top" },
];

test("sandbox replays the Dayos sequence and ranks legal open-role responses", () => {
  const result = analyzeDraftSandbox({
    actions: dayosActions,
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
      (row) => !dayosActions.some((action) => action.champion === row.champion),
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
    actions: [...dayosActions, { side: "red", champion: "Rakan", role: "sup" }],
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

test("excluded champions are removed from the legal recommendation set", () => {
  const baseline = analyzeDraftSandbox({
    actions: dayosActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
  });
  const topChampion = baseline.recommendations[0].champion;
  const excluded = analyzeDraftSandbox({
    actions: dayosActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    excluded: [topChampion],
    league: "LCS",
  });

  assert.ok(!excluded.recommendations.some((row) => row.champion === topChampion));
});
