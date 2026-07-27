import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import test from "node:test";
import {
  compositionRuntimeMetadata,
  compositionRuntimeSchemaIssues,
  normalizeCompositionPatch,
} from "./draftComposition.ts";
import {
  analyzeDraftSandbox,
  draftCatalog,
  draftScore,
  validateDraftSandboxState,
  type DraftAction,
} from "./draftScore.ts";

const partialActions: DraftAction[] = [
  { side: "blue", champion: "Jarvan IV", role: "jng" },
  { side: "red", champion: "Ezreal", role: "bot" },
  { side: "red", champion: "Naafiri", role: "jng" },
  { side: "blue", champion: "Orianna", role: "mid" },
  { side: "blue", champion: "Jayce", role: "top" },
];
const SANDBOX_PATCH = "16.01";

test("sandbox replays a partial sequence and ranks legal open-role responses", () => {
  const result = analyzeDraftSandbox({
    actions: partialActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
    patch: SANDBOX_PATCH,
    limit: 20,
  });

  assert.equal(result.timeline.length, 5);
  assert.deepEqual(result.open_roles, ["top", "mid", "sup"]);
  assert.ok(result.current.projected_value > 0 && result.current.projected_value < 1);
  assert.equal(result.value_kind, "experimental_composition_policy_value");
  assert.equal(result.probability_status, "withheld_failed_chronological_gate");
  assert.equal(result.candidate_role_policy, "supported_pro_roles_minimum_maps");
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
      (row, index, rows) =>
        index === 0 || rows[index - 1].projected_value >= row.projected_value,
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
    patch: SANDBOX_PATCH,
  });

  assert.equal(result.perspective, "red");
  assert.equal(result.recommendation_side, "blue");
  assert.ok(result.recommendations[0].delta_points > 0);
  assert.ok(
    result.recommendations[0].projected_value >
      1 - result.current.projected_value,
  );
  assert.equal(result.recommendations[0].lookahead_plies, 2);
  assert.deepEqual(
    result.recommendations[0].principal_variation.map((action) => action.side),
    ["blue", "red"],
  );
});

test("sandbox policy reaches the opponent after a same-side double pick", () => {
  const result = analyzeDraftSandbox({
    actions: [{ side: "blue", champion: "Renekton", role: "top" }],
    perspective: "red",
    next_side: "red",
    candidate_role: "jng",
    league: "LCK",
    patch: SANDBOX_PATCH,
    limit: 3,
  });

  assert.equal(result.recommendations[0].lookahead_plies, 2);
  assert.deepEqual(
    result.recommendations[0].principal_variation.map((action) => action.side),
    ["red", "blue"],
  );
  assert.equal(result.search.exhaustive, false);
  assert.ok(result.search.root_legal_actions > result.search.root_evaluated_actions);
  assert.ok(result.search.root_evaluated_actions <= 32);
  assert.equal(result.search.future_beam_width, 8);
});

test("sandbox accepts legacy role-less partial boards with role-agnostic scoring", () => {
  const result = analyzeDraftSandbox({
    actions: [
      { side: "blue", champion: "Aatrox", role: null },
      { side: "red", champion: "Jinx", role: "bot" },
    ],
    perspective: "blue",
    next_side: "red",
    candidate_role: "open",
    league: "LCK",
    patch: SANDBOX_PATCH,
    limit: 5,
  });

  assert.equal(result.model_context, null);
  assert.equal(result.candidate_role, "open");
  assert.ok(result.recommendations.length > 0);
  assert.equal(
    result.note.includes("unresolved roles"),
    true,
  );
});

test("mid-draft recommendations react to ally and enemy composition", () => {
  const base: DraftAction[] = [
    { side: "blue", champion: "Renekton", role: "top" },
    { side: "red", champion: "Sejuani", role: "jng" },
    { side: "red", champion: "Orianna", role: "mid" },
  ];
  const withJinx = analyzeDraftSandbox({
    actions: [...base, { side: "blue", champion: "Jinx", role: "bot" }],
    perspective: "blue",
    next_side: "blue",
    candidate_role: "sup",
    league: "LCS",
    patch: SANDBOX_PATCH,
    limit: 10,
  });
  const withLucian = analyzeDraftSandbox({
    actions: [...base, { side: "blue", champion: "Lucian", role: "bot" }],
    perspective: "blue",
    next_side: "blue",
    candidate_role: "sup",
    league: "LCS",
    patch: SANDBOX_PATCH,
    limit: 10,
  });

  assert.notDeepEqual(
    withJinx.recommendations.map((row) => [
      row.champion,
      row.projected_value,
    ]),
    withLucian.recommendations.map((row) => [
      row.champion,
      row.projected_value,
    ]),
  );
  assert.equal(withJinx.current.audit.probability_pipeline_gate, "failed");
  assert.equal(
    withJinx.current.audit.release_runtime_binding,
    "not_checked",
  );
});

test("sandbox propagates the selected model patch into every branch score", () => {
  const result = analyzeDraftSandbox({
    actions: partialActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
    patch: "16.1",
    limit: 3,
  });

  assert.equal(result.current.audit.calibration, "not_applicable_to_policy_value");
  assert.ok(result.model_context);
  assert.equal(result.model_context.normalized_patch, SANDBOX_PATCH);
  assert.equal(result.model_context.patch_status, "exact");
});

test("sandbox fails closed without an observed patch context", () => {
  const input = {
    actions: partialActions,
    perspective: "red" as const,
    next_side: "red" as const,
    candidate_role: "open" as const,
    league: "LCS",
    limit: 1,
  };
  assert.throws(
    () => analyzeDraftSandbox(input),
    /patch is required; select an observed patch context/i,
  );
  assert.throws(
    () => analyzeDraftSandbox({ ...input, patch: "99.99" }),
    /outside the observed model artifact/i,
  );
});

test("sandbox rejects impossible pick order, next side, and duplicate roles", () => {
  assert.throws(
    () =>
      validateDraftSandboxState(
        [{ side: "red", champion: "Ahri", role: "mid" }],
        "red",
      ),
    /pick 1 must belong to blue side/,
  );
  assert.throws(
    () =>
      validateDraftSandboxState(
        [{ side: "blue", champion: "Ahri", role: "mid" }],
        "blue",
      ),
    /next side must be red/,
  );
  assert.throws(
    () =>
      validateDraftSandboxState(
        [{ side: "blue", champion: "Ahri", role: null }],
        "red",
      ),
    /needs one verified role/,
  );
  assert.throws(
    () =>
      validateDraftSandboxState(
        [
          { side: "blue", champion: "Ahri", role: "mid" },
          { side: "red", champion: "Akali", role: "mid" },
          { side: "red", champion: "Orianna", role: "mid" },
        ],
        "blue",
      ),
    /cannot assign two champions to mid/,
  );
});

test("sandbox exposes every champion for manual what-if states but recommends only supported roles", () => {
  const catalog = draftCatalog();
  assert.ok(catalog.length >= 160);
  assert.deepEqual(catalog.find((row) => row.name === "Alistar")?.roles, [
    "sup",
  ]);
  assert.deepEqual(catalog.find((row) => row.name === "Nautilus")?.roles, [
    "sup",
  ]);
  const result = analyzeDraftSandbox({
    actions: [],
    perspective: "blue",
    next_side: "blue",
    candidate_role: "sup",
    league: "LCS",
    patch: SANDBOX_PATCH,
    limit: 200,
  });
  const supportCandidates = new Map(result.recommendations.map((row) => [row.champion, row]));
  assert.ok(supportCandidates.size > 0);
  assert.ok(supportCandidates.size < catalog.length);
  assert.ok([...supportCandidates.values()].every((row) => row.evidence !== "Unseen role"));
  assert.ok([...supportCandidates.values()].every((row) => row.role === "sup"));
});

test("manual any-role mode does not expand policy search beyond observed roles", () => {
  const shared = {
    actions: partialActions,
    perspective: "red" as const,
    next_side: "red" as const,
    league: "LCS",
    patch: SANDBOX_PATCH,
    limit: 8,
  };
  const observed = analyzeDraftSandbox({
    ...shared,
    candidate_role: "open",
  });
  const manual = analyzeDraftSandbox({
    ...shared,
    candidate_role: "any",
  });

  assert.equal(manual.candidate_role_policy, "supported_pro_roles_minimum_maps");
  assert.ok(
    manual.recommendations.every(
      (row) => row.role && row.evidence !== "Unseen role",
    ),
  );
  assert.deepEqual(
    manual.recommendations.map((row) => [row.champion, row.role]),
    observed.recommendations.map((row) => [row.champion, row.role]),
  );
});

test("excluded champions are removed from the legal recommendation set", () => {
  const baseline = analyzeDraftSandbox({
    actions: partialActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    league: "LCS",
    patch: SANDBOX_PATCH,
  });
  const topChampion = baseline.recommendations[0].champion;
  const excluded = analyzeDraftSandbox({
    actions: partialActions,
    perspective: "red",
    next_side: "red",
    candidate_role: "open",
    excluded: [topChampion],
    league: "LCS",
    patch: SANDBOX_PATCH,
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
  assert.equal(result.model?.patch_status, "pooled_unsupported");
  assert.equal(result.model?.artifact_sha256.length, 64);
  assert.ok(result.calibration.intercept != null);
  assert.ok(result.calibration.slope != null);
  assert.ok(result.calibration.neutral_blue_baseline != null);
  const expectedProbability =
    1 /
    (1 +
      Math.exp(
        -(
          result.calibration.intercept +
          result.calibration.slope * (result.components.model_edge ?? 0)
        ),
      ));
  assert.ok(Math.abs(result.p_blue_draft - expectedProbability) <= 1e-4);
  assert.ok(
    Math.abs(
      result.wr_bump_pp -
        100 *
          (result.p_blue_draft -
            result.calibration.neutral_blue_baseline),
    ) <= 0.02,
  );
  assert.ok(
    result.explanation?.champions.every(
      (row) =>
        row.evidence.role_champion_maps.possible_terms === 1 &&
        row.evidence.ally_synergy_pairs.possible_terms === 4 &&
        row.evidence.enemy_interaction_pairs.possible_terms === 5,
    ),
  );
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
  assert.equal(
    result.calibration.neutral_blue_baseline,
    swapped.calibration.neutral_blue_baseline,
  );
  assert.ok(swapped.uncertainty);
  assert.ok(swapped.uncertainty.p_blue_95[0] <= swapped.p_blue_draft);
  assert.ok(swapped.p_blue_draft <= swapped.uncertainty.p_blue_95[1]);
});

test("complete scoring requires authoritative roles and never falls back", () => {
  const blue = ["Aatrox", "Sejuani", "Ahri", "Jinx", "Leona"];
  const red = ["Gnar", "Vi", "Orianna", "Xayah", "Nautilus"];
  assert.throws(
    () => draftScore({ blue, red, league: "LCK", patch: "16.02" }),
    /authoritative role/,
  );
  assert.throws(
    () =>
      draftScore({
        blue,
        red,
        blue_roles: ["top", "top", "mid", "bot", "sup"],
        red_roles: ["top", "jng", "mid", "bot", "sup"],
        league: "LCK",
        patch: "16.02",
      }),
    /same role twice/,
  );
});

test("complete scoring rejects duplicate and unknown champions", () => {
  const roles = ["top", "jng", "mid", "bot", "sup"];
  assert.throws(
    () =>
      draftScore({
        blue: ["Aatrox", "Sejuani", "Ahri", "Jinx", "Leona"],
        red: ["Aatrox", "Vi", "Orianna", "Xayah", "Nautilus"],
        blue_roles: roles,
        red_roles: roles,
        league: "LCK",
        patch: "16.02",
      }),
    /cannot be selected more than once/,
  );
  assert.throws(
    () =>
      draftScore({
        blue: ["Not A Champion", "Sejuani", "Ahri", "Jinx", "Leona"],
        red: ["Gnar", "Vi", "Orianna", "Xayah", "Nautilus"],
        blue_roles: roles,
        red_roles: roles,
        league: "LCK",
        patch: "16.02",
      }),
    /unknown champion/,
  );
});

test("patch normalization preserves numeric minor versions", () => {
  assert.equal(normalizeCompositionPatch("16.1"), "16.01");
  assert.equal(normalizeCompositionPatch("16.01"), "16.01");
  assert.equal(normalizeCompositionPatch("16.1.123.456"), "16.01");
  assert.equal(normalizeCompositionPatch("16.10"), "16.10");
  assert.equal(normalizeCompositionPatch("not-a-patch"), null);
  assert.equal(normalizeCompositionPatch("16.1garbage"), null);
});

test("runtime schema requires complete uncertainty and calibration contracts", () => {
  const packed = readFileSync(
    new URL("../../data/draft/composition_runtime.json.gz.b64", import.meta.url),
    "utf8",
  );
  type MutableRuntimeFixture = {
    calibration: Record<string, unknown>;
    strength_calibration: Record<string, unknown>;
    feature_specs: Record<string, Record<string, unknown>>;
    artifact_sha256?: unknown;
    model_code_sha256?: unknown;
    training_population_sha256?: unknown;
    numerical_environment?: unknown;
    intercept_se?: unknown;
    role_champion_counts?: unknown;
    low_rank?: Record<string, unknown>;
    uncertainty?: unknown;
    [key: string]: unknown;
  };
  const runtime = JSON.parse(
    gunzipSync(Buffer.from(packed, "base64")).toString("utf8"),
  ) as MutableRuntimeFixture;
  assert.deepEqual(compositionRuntimeSchemaIssues(runtime), []);
  const mutations: Array<(copy: MutableRuntimeFixture) => void> = [
    (copy) => delete copy.calibration.slope,
    (copy) => delete copy.calibration.covariance,
    (copy) => delete copy.artifact_sha256,
    (copy) => delete copy.model_code_sha256,
    (copy) => delete copy.training_population_sha256,
    (copy) => delete copy.numerical_environment,
    (copy) => delete copy.intercept_se,
    (copy) => delete copy.role_champion_counts,
    (copy) => delete copy.low_rank,
    (copy) => delete copy.uncertainty,
    (copy) => {
      if (copy.low_rank) copy.low_rank.rank = 1;
    },
    (copy) => {
      const first = Object.values(copy.feature_specs)[0];
      delete first.se;
    },
  ];
  for (const mutate of mutations) {
    const copy = structuredClone(runtime);
    mutate(copy);
    assert.ok(compositionRuntimeSchemaIssues(copy).length > 0);
  }
  const metadata = compositionRuntimeMetadata();
  assert.equal(metadata?.runtime_status, "available");
  assert.equal(metadata?.strength_calibration_status, "unavailable");
  assert.ok(metadata?.supported_patches.includes("16.01"));
  assert.ok(metadata?.supported_leagues.includes("LCK"));
});

test("available strength calibration requires fit/source/model identity", () => {
  const packed = readFileSync(
    new URL("../../data/draft/composition_runtime.json.gz.b64", import.meta.url),
    "utf8",
  );
  const runtime = JSON.parse(
    gunzipSync(Buffer.from(packed, "base64")).toString("utf8"),
  ) as Record<string, unknown>;
  runtime.strength_calibration = {
    schema_version: "1.0.0",
    status: "available",
    calibration_id: "strength-calibration-v2-test",
    fit_cutoff: "2026-01-01T00:00:00Z",
    holdout_start: "2026-02-01T00:00:00Z",
    source: {
      artifact: "data/lol/models/elo_wr_calibration.json",
      artifact_sha256: "a".repeat(64),
      artifact_version: 2,
    },
    team: {
      model_id: "strength-calibration-v2-test-team",
      intercept: 0.1,
      coef: 2.0,
    },
    player: {
      model_id: "strength-calibration-v2-test-player",
      intercept: 0.1,
      coef: 3.0,
    },
    blend: {
      model_id: "strength-calibration-v2-test-blend",
      intercept: -2.0,
      coef_team: 2.0,
      coef_player: 2.0,
    },
  };
  assert.deepEqual(compositionRuntimeSchemaIssues(runtime), []);

  for (const mutate of [
    (copy: Record<string, unknown>) => {
      delete (
        copy.strength_calibration as Record<string, unknown>
      ).fit_cutoff;
    },
    (copy: Record<string, unknown>) => {
      const strength = copy.strength_calibration as Record<string, unknown>;
      delete (strength.source as Record<string, unknown>).artifact_sha256;
    },
    (copy: Record<string, unknown>) => {
      const strength = copy.strength_calibration as Record<string, unknown>;
      delete (strength.player as Record<string, unknown>).model_id;
    },
    (copy: Record<string, unknown>) => {
      const strength = copy.strength_calibration as Record<string, unknown>;
      delete (strength.blend as Record<string, unknown>).coef_player;
    },
  ]) {
    const copy = structuredClone(runtime);
    mutate(copy);
    assert.ok(compositionRuntimeSchemaIssues(copy).length > 0);
  }
});

test("contextual strength rejects unavailable calibration without defaults", () => {
  const input = {
    blue: ["Aatrox", "Sejuani", "Ahri", "Jinx", "Leona"],
    red: ["Gnar", "Vi", "Orianna", "Xayah", "Nautilus"],
    blue_roles: ["top", "jng", "mid", "bot", "sup"],
    red_roles: ["top", "jng", "mid", "bot", "sup"],
    league: "LCK",
    patch: "16.02",
    team_elo_diff: 50,
  };
  assert.throws(
    () => draftScore(input),
    /explicit source, as-of time, and model ID/,
  );
  assert.throws(
    () =>
      draftScore({
        ...input,
        strength_source: "immutable pre-match producer snapshot",
        strength_as_of: "2026-01-01T00:00:00Z",
        strength_model_id: "team-strength-v1",
      }),
    /Contextual strength calibration is unavailable/,
  );
});

test("sandbox source contains no named public demo fixture", () => {
  const source = readFileSync(new URL("../components/DraftSandbox.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(source, /Example question|Load .*example/i);
  assert.match(source, /initialActions = \[\]/);
  assert.match(source, /sandbox-champion-grid/);
  assert.doesNotMatch(source, /list="sandbox-champions"/);
});

test("sandbox UI defaults to the latest public patch with pooled disclosure", () => {
  const source = readFileSync(
    new URL("../components/DraftSandbox.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /initialPublicPatch \?\? latestObservedPatch \?\? ""/,
  );
  assert.match(source, /if \(!publicPatch\) return;/);
  assert.match(source, /public_patch: publicPatch/);
  assert.match(source, /searchParams\.set\("public_patch", publicPatch\)/);
  assert.match(source, /patchSupportText/);
  assert.match(source, /current observed; values use/);
  assert.match(source, /patch-specific|pooled/);
});

test("sandbox inline role chips for multi-role picks", () => {
  const source = readFileSync(
    new URL("../components/DraftSandbox.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /const legalRolesForChampion/);
  assert.match(source, /const canPickDirectly/);
  assert.match(source, /showRoleChoices/);
  assert.match(source, /sandbox-mini-role-button/);
  assert.match(source, /sandbox-champion-item/);
  assert.match(source, /className=\"sandbox-role-actions\"/);
  assert.match(source, /const explicitRole/);
});

test("sandbox copy distinguishes manual unseen-role states from supported policy search", () => {
  const source = readFileSync(
    new URL("../components/DraftSandbox.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /Supported pro roles/);
  assert.match(source, /Manual any role/);
  assert.match(source, /candidateRole !== "open" && candidateRole !== "any"/);
  assert.match(source, /at least[\s\S]*pro maps/);
  assert.match(source, /champion\.roles\.includes\(candidateRole\)/);
});
