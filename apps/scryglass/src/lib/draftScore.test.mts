import assert from "node:assert/strict";
import test from "node:test";
import {
  analyzeDraftSandbox,
  draftCatalog,
  draftScore,
  type DraftAction,
} from "./draftScore";

const sampleActions: DraftAction[] = [
  { side: "blue", champion: "Jarvan IV", role: "jng" },
  { side: "red", champion: "Ezreal", role: "bot" },
  { side: "red", champion: "Naafiri", role: "jng" },
  { side: "blue", champion: "Orianna", role: "mid" },
  { side: "blue", champion: "Jayce", role: "top" },
];

test("sandbox replays a partial sequence and ranks legal open-role responses", () => {
  const result = analyzeDraftSandbox({
    actions: sampleActions,
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
      (row) => !sampleActions.some((action) => action.champion === row.champion),
    ),
  );
  assert.ok(
    result.recommendations.every(
      (row, index, rows) => index === 0 || rows[index - 1].projected_wr >= row.projected_wr,
    ),
  );
});

test("same-role evidence obeys its time-validated serving gate", () => {
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
  if (withRoles.components.pair_logit === 0) {
    assert.equal(
      analyzeDraftSandbox({
        actions: [],
        perspective: "blue",
        next_side: "blue",
      }).model.interaction_gates.lane,
      0,
    );
  } else {
    assert.notEqual(withRoles.p_blue_draft, withoutRoles.p_blue_draft);
  }
});

test("recommendations optimize the side whose turn it is", () => {
  const result = analyzeDraftSandbox({
    actions: [...sampleActions, { side: "red", champion: "Rakan", role: "sup" }],
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
    actions: sampleActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
  });
  const topChampion = baseline.recommendations[0].champion;
  const excluded = analyzeDraftSandbox({
    actions: sampleActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    excluded: [topChampion],
    league: "LCS",
  });

  assert.ok(!excluded.recommendations.some((row) => row.champion === topChampion));
});

test("catalog includes the current champion roster and neutral-prior champions", () => {
  const catalog = draftCatalog();
  assert.equal(catalog.length, 173);
  const locke = catalog.find((champion) => champion.name === "Locke");
  assert.ok(locke);
  assert.equal(locke.modeled, false);
  assert.equal(locke.games, 0);
});

test("recommendations expose validated state-dependent interaction components", () => {
  const result = analyzeDraftSandbox({
    actions: sampleActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
    limit: 30,
  });

  assert.ok(
    result.recommendations.some(
      (row) =>
        Math.abs(row.components.synergy) > 0 ||
        Math.abs(row.components.counters) > 0 ||
        Math.abs(row.components.lane) > 0,
    ),
  );
  assert.ok(result.model.interaction_gates.synergy > 0);
});

test("recommendation order responds to a different allied and enemy draft state", () => {
  const first = analyzeDraftSandbox({
    actions: sampleActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
    limit: 10,
  });
  const second = analyzeDraftSandbox({
    actions: [
      { side: "blue", champion: "Sejuani", role: "jng" },
      { side: "red", champion: "Ezreal", role: "bot" },
      { side: "red", champion: "Naafiri", role: "jng" },
      { side: "blue", champion: "Azir", role: "mid" },
      { side: "blue", champion: "Gnar", role: "top" },
    ],
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
    limit: 10,
  });

  assert.notDeepEqual(
    first.recommendations.map((row) => `${row.champion}:${row.role}`),
    second.recommendations.map((row) => `${row.champion}:${row.role}`),
  );
});

test("recommendation evidence count is specific to the proposed role", () => {
  const catalog = draftCatalog();
  const result = analyzeDraftSandbox({
    actions: sampleActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "top",
    league: "LCS",
    limit: 30,
  });
  const recommendation = result.recommendations.find(
    (row) => row.champion === "Vayne",
  );
  const champion = catalog.find((row) => row.name === "Vayne");

  assert.ok(recommendation);
  assert.ok(champion);
  assert.equal(recommendation.sample_games, champion.role_games.top);
  assert.notEqual(recommendation.sample_games, champion.games);
});

test("player champion comfort can change the recommendation order", () => {
  const baseline = analyzeDraftSandbox({
    actions: sampleActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "sup",
    league: "LCS",
    limit: 30,
  });
  const target = baseline.recommendations.at(-1);
  assert.ok(target?.role);
  const personalized = analyzeDraftSandbox({
    actions: sampleActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "sup",
    league: "LCS",
    player_context: {
      red: {
        sup: {
          player: "Test support",
          team: "Test",
          role: "sup",
          rating: 1600,
          raw_rating: 1600,
          sigma: 28,
          n_maps: 100,
          mastery: {
            [target.champion]: { logit: 1.5, n: 50 },
          },
        },
      },
    },
    limit: 30,
  });

  assert.equal(personalized.recommendations[0].champion, target.champion);
  assert.equal(personalized.recommendations[0].player, "Test support");
  assert.equal(personalized.recommendations[0].player_games, 50);
});

test("team and lineup Elo change the combined projection", () => {
  const even = draftScore({
    blue: ["Orianna"],
    red: ["Akali"],
    blue_roles: ["mid"],
    red_roles: ["mid"],
    league: "LCS",
  });
  const strengthAware = draftScore({
    blue: ["Orianna"],
    red: ["Akali"],
    blue_roles: ["mid"],
    red_roles: ["mid"],
    league: "LCS",
    team_elo_diff: 150,
    player_elo_diff: 100,
  });

  assert.equal(even.calibration.strength_source, "even-strength assumption");
  assert.equal(strengthAware.calibration.strength_source, "team + lineup Elo");
  assert.ok(strengthAware.p_blue_combined > even.p_blue_combined);
});

test("decision trace separates the pre-match prior and tempers partial-draft confidence", () => {
  const prior = analyzeDraftSandbox({
    actions: [],
    perspective: "blue",
    next_side: "blue",
    league: "LCS",
    team_elo_diff: 200,
  });
  const result = analyzeDraftSandbox({
    actions: sampleActions,
    perspective: "blue",
    next_side: "red",
    league: "LCS",
    team_elo_diff: 200,
  });
  const firstPickChange =
    100 * (result.timeline[0].projected_wr - prior.current.projected_wr);

  assert.ok(Math.abs(result.timeline[0].delta_pp - firstPickChange) < 0.02);
  assert.ok(result.timeline[1].confidence < 0.6);
  assert.ok(
    result.timeline[result.timeline.length - 1].confidence >
      result.timeline[1].confidence,
  );
});
