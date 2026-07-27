import assert from "node:assert/strict";
import test from "node:test";
import {
  GET as sandboxGet,
  POST as sandboxPost,
  publicDraftApiError,
} from "../app/api/draft-sandbox/route.ts";
import {
  GET as draftWrGet,
  POST as draftWrPost,
  draftWrWithheldPayload,
} from "../app/api/draft-wr/route.ts";

function request(body: unknown): Request {
  return new Request("http://localhost/api/draft-sandbox", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function json(response: Response): Promise<Record<string, unknown>> {
  return (await response.json()) as Record<string, unknown>;
}

const baseRequest = {
  actions: [],
  perspective: "blue",
  next_side: "blue",
  candidate_role: "open",
  excluded: [],
  league: "LCK",
  public_patch: "26.01",
  limit: 2,
};

test("draft probability route remains withheld with immutable gate evidence", async () => {
  for (const response of await Promise.all([draftWrGet(), draftWrPost()])) {
    assert.equal(response.status, 503);
    const payload = await json(response);
    assert.equal(payload.status, "withheld_failed_chronological_gate");
    assert.equal(payload.served_probability_model, null);
    assert.equal(
      payload.gate_id,
      "draft-probability-chronological-2026-07-27",
    );
    assert.deepEqual(
      (payload.patch_selection as Record<string, unknown>).current_patch,
      {
        public_patch: "26.14",
        source_patch_key: "16.14",
      },
    );
    assert.ok(!("p_blue_draft" in payload));
  }

  const verifiedArtifact = {
    schema_version: "1.0.0",
    created_utc: "2026-07-27T00:00:00Z",
    population: { maps: 16_334 },
    splits: {
      draft: [
        {
          name: "final",
          maps: 2_477,
          date_min: "2026-04-30T00:00:00Z",
          date_max: "2026-07-18T00:00:00Z",
        },
      ],
    },
    draft_probability: {
      artifact_sha256: "d".repeat(64),
      model_code_sha256: "e".repeat(64),
      training_population_sha256: "f".repeat(64),
      evaluation_scope:
        "chronological_pre_final_fit_scored_on_untouched_final",
      release_artifact_role:
        "full_population_refit_for_experimental_policy_utility_only",
      release_refit_includes_final_labels: true,
      gate_status: "failed",
      decision:
        "withhold_numeric_probability; publish pooled composition policy utility only",
      selection_winner: "baseline_overall_blue_rate",
      winner_is_composition: false,
      best_composition_candidate: "draft_candidate",
      final_test: {
        baseline_overall_blue_rate: {
          maps: 2_477,
          log_loss: 0.690582,
          brier: 0.248719,
        },
        draft_candidate: {
          maps: 2_477,
          log_loss: 0.698518,
          brier: 0.251806,
        },
      },
    },
  };
  const verified = draftWrWithheldPayload(verifiedArtifact, {
    status: "matched_release_refit_to_failed_pipeline_gate",
    packId: "v-test",
    runtimeArtifactSha256: "d".repeat(64),
    modelCodeSha256: "e".repeat(64),
    trainingPopulationSha256: "f".repeat(64),
    modelFileSha256: "d".repeat(64),
    gateFileSha256: "a".repeat(64),
  });
  assert.equal(verified.evidence_status, "verified_immutable_pack_artifact");
  assert.equal(verified.evidence?.final_test_maps, 2_477);
  assert.equal(verified.evidence?.composition_log_loss, 0.698518);

  const invalid = draftWrWithheldPayload({});
  assert.equal(invalid.evidence_status, "unavailable_or_invalid");
  assert.equal(invalid.evidence, null);
});

test("sandbox metadata is available only with a validated runtime", async () => {
  const response = await sandboxGet();
  assert.equal(response.status, 200);
  const payload = await json(response);
  assert.equal(payload.value_kind, "experimental_composition_policy_value");
  assert.equal(
    payload.probability_status,
    "withheld_failed_chronological_gate",
  );
  const metadata = payload.model_metadata as Record<string, unknown>;
  assert.equal(metadata.runtime_status, "available");
  assert.equal(String(metadata.artifact_sha256).length, 64);
  assert.ok(!("latest_observed_patch" in metadata));
  assert.deepEqual(metadata.latest_observed_patch_contract, {
    public_patch: "26.14",
    source_patch_key: "16.14",
  });
  assert.equal(
    payload.candidate_role_policy,
    "supported_pro_roles_minimum_maps",
  );
  assert.deepEqual(payload.patch_selection, {
    required: true,
    request_field: "public_patch",
    mode: "current_pooled_or_historical_exact",
    contract: "public 25.x -> source 15.x; public 26.x -> source 16.x",
    current_patch_supported: true,
    current_patch: metadata.latest_observed_patch_contract,
    patch_specific_patches: metadata.supported_patch_contracts,
    pooled_holdout_patches: metadata.observed_holdout_patch_contracts,
    analysis_patches: metadata.analysis_patch_contracts,
  });
});

test("sandbox API requires an explicit observed patch context", async () => {
  const withoutPatch = { ...baseRequest, public_patch: undefined };
  for (const body of [
    withoutPatch,
    { ...baseRequest, public_patch: null },
    { ...baseRequest, public_patch: "" },
    { ...baseRequest, public_patch: "   " },
  ]) {
    const response = await sandboxPost(request(body));
    assert.equal(response.status, 400);
    const payload = await json(response);
    assert.equal(payload.code, "public_patch_required");
    assert.match(String(payload.error), /public_patch/i);
  }

  const unsupported = await sandboxPost(
    request({ ...baseRequest, public_patch: "26.99" }),
  );
  assert.equal(unsupported.status, 400);
  const unsupportedPayload = await json(unsupported);
  assert.equal(unsupportedPayload.code, "unsupported_observed_patch");
  assert.match(String(unsupportedPayload.error), /observed model artifact/i);
});

test("sandbox accepts an observed holdout patch as pooled utility", async () => {
  const patchSelection = (await json(await sandboxGet())).patch_selection as {
    pooled_holdout_patches: Array<{
      public_patch: string;
      source_patch_key: string;
    }>;
  };
  const patch = patchSelection.pooled_holdout_patches.at(-1);
  assert.ok(patch);
  const response = await sandboxPost(
    request({ ...baseRequest, public_patch: patch.public_patch }),
  );
  assert.equal(response.status, 200);
  const payload = await json(response);
  const model = payload.model_context as Record<string, unknown>;
  assert.equal(model.public_patch, patch.public_patch);
  assert.equal(model.source_patch_key, patch.source_patch_key);
  assert.equal(model.patch_status, "pooled_unsupported");
  assert.equal(payload.probability_status, "withheld_failed_chronological_gate");
});

test("sandbox derives an exact source key from the public patch before scoring", async () => {
  const response = await sandboxPost(request(baseRequest));
  assert.equal(response.status, 200);
  const payload = await json(response);
  const model = payload.model_context as Record<string, unknown>;
  assert.deepEqual(payload.patch_context, {
    public_patch: "26.01",
    source_patch_key: "16.01",
  });
  assert.equal(model.public_patch, "26.01");
  assert.equal(model.source_patch_key, "16.01");
  assert.ok(!("normalized_patch" in model));
  assert.equal(model.patch_status, "exact");
  assert.equal(payload.value_kind, "experimental_composition_policy_value");
  assert.equal(
    payload.candidate_role_policy,
    "supported_pro_roles_minimum_maps",
  );
  assert.ok(!("projected_wr" in payload));
});

test("sandbox rejects ambiguous minors and source keys at the public boundary", async () => {
  for (const publicPatch of ["26.1", "16.1", "16.14", "26.14.1"]) {
    const response = await sandboxPost(
      request({ ...baseRequest, public_patch: publicPatch }),
    );
    assert.equal(response.status, 400);
    const payload = await json(response);
    assert.equal(payload.code, "invalid_public_patch");
    assert.match(String(payload.error), /two-digit minor/i);
  }
});

test("sandbox rejects unsupported league, patch, strength, and limits", async () => {
  for (const body of [
    { ...baseRequest, league: "INTL" },
    { ...baseRequest, elo_diff: 50 },
    { ...baseRequest, limit: 0 },
  ]) {
    const response = await sandboxPost(request(body));
    assert.equal(response.status, 400);
  }
});

test("sandbox rejects unknown champions, duplicate champions, and role collisions", async () => {
  const cases = [
    {
      ...baseRequest,
      actions: [{ side: "blue", champion: "Not A Champion", role: "top" }],
      next_side: "red",
    },
    {
      ...baseRequest,
      actions: [
        { side: "blue", champion: "Aatrox", role: "top" },
        { side: "red", champion: "Aatrox", role: "top" },
      ],
      next_side: "red",
    },
    {
      ...baseRequest,
      actions: [
        { side: "blue", champion: "Aatrox", role: "top" },
        { side: "red", champion: "Gnar", role: "top" },
        { side: "red", champion: "Vi", role: "top" },
      ],
      next_side: "blue",
    },
  ];
  for (const body of cases) {
    const response = await sandboxPost(request(body));
    assert.equal(response.status, 400);
  }
});

test("sandbox accepts legacy role-missing actions for open-role adaptation", async () => {
  const response = await sandboxPost(
    request({
      ...baseRequest,
      actions: [{ side: "blue", champion: "Aatrox", role: null }],
      next_side: "red",
    }),
  );
  assert.equal(response.status, 200);
  const payload = await json(response);
  assert.equal(
    (payload.model_context as Record<string, unknown>).public_patch,
    "26.01",
  );
  assert.ok((payload.recommendations as unknown[]).length > 0);
});

test("internal draft API failures never serialize exception text", async () => {
  const original = console.error;
  console.error = () => {};
  try {
    const response = publicDraftApiError(
      new Error("sentinel /private/model/path secret-value"),
      "draft-sandbox",
    );
    assert.equal(response.status, 500);
    const payload = await json(response);
    assert.equal(payload.code, "draft_internal_error");
    assert.match(String(payload.request_id), /^[0-9a-f-]{36}$/i);
    assert.doesNotMatch(JSON.stringify(payload), /sentinel|private|secret/i);
  } finally {
    console.error = original;
  }
});
