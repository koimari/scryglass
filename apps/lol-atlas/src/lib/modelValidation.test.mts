import assert from "node:assert/strict";
import test from "node:test";
import {
  draftProbabilityGateEvidence,
  draftRuntimeBindingEvidence,
  teamRatingGateEvidence,
} from "./modelValidation.ts";

function artifact() {
  return {
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
}

test("accepts reconciled immutable draft-gate evidence", () => {
  const parsed = draftProbabilityGateEvidence(artifact());
  assert.equal(parsed?.finalTestMaps, 2_477);
  assert.equal(parsed?.compositionModelId, "draft_candidate");
  assert.equal(parsed?.compositionLogLoss, 0.698518);
  assert.equal(parsed?.modelCodeSha256, "e".repeat(64));
  assert.equal(parsed?.decision, "withhold_numeric_probability");
});

test("fails closed on a changed gate, malformed clock, or impossible sample", () => {
  const changed = artifact();
  changed.draft_probability.gate_status = "passed";
  assert.equal(draftProbabilityGateEvidence(changed), null);

  const malformed = artifact();
  malformed.splits.draft[0].date_min = "not-a-date";
  assert.equal(draftProbabilityGateEvidence(malformed), null);

  const impossible = artifact();
  impossible.splits.draft[0].maps = 20_000;
  assert.equal(draftProbabilityGateEvidence(impossible), null);
});

test("Sandbox runtime must match the active pack model and exact gate hashes", () => {
  const manifest = {
    pack_id: "v-test",
    files: [
      {
        path: "models/draft_composition.json",
        sha256: "d".repeat(64),
      },
      {
        path: "models/model_validation_2026-07-27.json",
        sha256: "a".repeat(64),
      },
    ],
  };
  const runtime = {
    artifact_sha256: "d".repeat(64),
    model_code_sha256: "e".repeat(64),
    training_population_sha256: "f".repeat(64),
  };
  assert.equal(
    draftRuntimeBindingEvidence(manifest, artifact(), runtime)?.status,
    "matched_release_refit_to_failed_pipeline_gate",
  );
  const drifted = structuredClone(manifest);
  drifted.files[0].sha256 = "0".repeat(64);
  assert.equal(
    draftRuntimeBindingEvidence(drifted, artifact(), runtime),
    null,
  );
});

function teamArtifact() {
  const code = "a".repeat(64);
  const config = "b".repeat(64);
  return {
    schema_version: "1.0.0",
    created_utc: "2026-07-27T00:00:00Z",
    team_rating: {
      model_id: "series_dynamic_bt",
      model_code_sha256: code,
      model_config_sha256: config,
      observation_rows_sha256: "c".repeat(64),
      model_version: `series_dynamic_bt:${code.slice(0, 12)}:${config.slice(0, 12)}`,
      gate_status: "passed",
      final_test: {
        series: 1771,
        log_loss: 0.58621,
        brier: 0.20007,
        ece_10_equal_width: 0.02778,
        format_stratified_calibration: {
          Bo5: { n: 226, ece: 0.09993 },
        },
      },
      paired_primary_comparison: {
        baseline_model_id: "rolling_series_elo",
        score: "log_loss",
        series: 1771,
        candidate_score: 0.58621,
        baseline_score: 0.62166,
        candidate_minus_baseline: 0.58621 - 0.62166,
        confidence_level: 0.95,
        confidence_interval: [-0.04807, -0.02191],
        decision: "superior",
        bootstrap: {
          replicates: 5000,
          block_size_series: 8,
        },
      },
    },
  };
}

test("accepts exact team gate evidence and preserves the Bo5 limitation", () => {
  const parsed = teamRatingGateEvidence(teamArtifact());
  assert.equal(parsed?.finalTestSeries, 1771);
  assert.equal(parsed?.rollingEloLogLoss, 0.62166);
  assert.equal(parsed?.bo5Series, 226);
  assert.equal(parsed?.bo5Ece, 0.09993);
});

test("team evidence fails closed on a non-superior interval or hash drift", () => {
  const interval = teamArtifact();
  interval.team_rating.paired_primary_comparison.confidence_interval = [
    -0.04, 0.01,
  ];
  assert.equal(teamRatingGateEvidence(interval), null);

  const hash = teamArtifact();
  hash.team_rating.model_code_sha256 = "d".repeat(64);
  assert.equal(teamRatingGateEvidence(hash), null);
});
