export type DraftProbabilityGateEvidence = {
  schemaVersion: string;
  artifactCreatedUtc: string;
  populationMaps: number;
  compositionModelId: string;
  artifactSha256: string;
  modelCodeSha256: string;
  trainingPopulationSha256: string;
  finalTestMaps: number;
  finalTestStart: string;
  finalTestEnd: string;
  compositionLogLoss: number;
  overallBaseRateLogLoss: number;
  compositionBrier: number;
  overallBaseRateBrier: number;
  decision: "withhold_numeric_probability";
  gateStatus: "failed";
  evaluationScope: "chronological_pre_final_fit_scored_on_untouched_final";
  releaseArtifactRole: "full_population_refit_for_experimental_policy_utility_only";
  releaseRefitIncludesFinalLabels: true;
};

export type DraftRuntimeBindingEvidence = {
  status: "matched_release_refit_to_failed_pipeline_gate";
  packId: string;
  runtimeArtifactSha256: string;
  modelCodeSha256: string;
  trainingPopulationSha256: string;
  modelFileSha256: string;
  gateFileSha256: string;
};

export type TeamRatingGateEvidence = {
  modelVersion: string;
  finalTestSeries: number;
  logLoss: number;
  brier: number;
  ece: number;
  rollingEloLogLoss: number;
  logLossDifference: number;
  confidenceInterval: [number, number];
  bootstrapReplicates: number;
  bootstrapBlockSeries: number;
  bo5Series: number;
  bo5Ece: number;
};

type UnknownRecord = Record<string, unknown>;

function record(value: unknown): UnknownRecord | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null;
}

function records(value: unknown): UnknownRecord[] | null {
  if (!Array.isArray(value)) return null;
  const parsed = value.map(record);
  return parsed.every((item): item is UnknownRecord => item != null)
    ? parsed
    : null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function positiveInteger(value: unknown): number | null {
  const parsed = finite(value);
  return parsed != null && Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function validClock(value: unknown): value is string {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

function sha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function manifestFileSha(
  manifest: UnknownRecord,
  relativePath: string,
): string | null {
  if (!Array.isArray(manifest.files)) return null;
  const row = manifest.files
    .map(record)
    .find(
      (file) =>
        file?.path === relativePath || file?.relative === relativePath,
    );
  return sha256(row?.sha256) ? row.sha256 : null;
}

/**
 * Bind the process-bundled Sandbox runtime to the exact immutable pack and
 * failed chronological gate currently served to readers. A valid model or
 * gate in isolation is insufficient: every hash must describe one artifact.
 */
export function draftRuntimeBindingEvidence(
  manifestValue: unknown,
  gateArtifact: unknown,
  runtimeMetadata: unknown,
): DraftRuntimeBindingEvidence | null {
  const manifest = record(manifestValue);
  const runtime = record(runtimeMetadata);
  const gate = draftProbabilityGateEvidence(gateArtifact);
  const packId =
    typeof manifest?.pack_id === "string" && manifest.pack_id.trim()
      ? manifest.pack_id.trim()
      : null;
  const runtimeArtifact = runtime?.artifact_sha256;
  const runtimeCode = runtime?.model_code_sha256;
  const runtimePopulation = runtime?.training_population_sha256;
  const modelFileSha = manifest
    ? manifestFileSha(manifest, "models/draft_composition.json")
    : null;
  const gateFileSha = manifest
    ? manifestFileSha(
        manifest,
        "models/model_validation_2026-07-27.json",
      )
    : null;

  if (
    packId == null ||
    gate == null ||
    !sha256(runtimeArtifact) ||
    !sha256(runtimeCode) ||
    !sha256(runtimePopulation) ||
    modelFileSha == null ||
    gateFileSha == null ||
    modelFileSha !== runtimeArtifact ||
    gate.artifactSha256 !== runtimeArtifact ||
    gate.modelCodeSha256 !== runtimeCode ||
    gate.trainingPopulationSha256 !== runtimePopulation
  ) {
    return null;
  }
  return {
    status: "matched_release_refit_to_failed_pipeline_gate",
    packId,
    runtimeArtifactSha256: runtimeArtifact,
    modelCodeSha256: runtimeCode,
    trainingPopulationSha256: runtimePopulation,
    modelFileSha256: modelFileSha,
    gateFileSha256: gateFileSha,
  };
}

/**
 * Validate only the immutable fields used by public draft-gate claims.
 * Unknown or internally inconsistent evidence is rejected rather than
 * replaced with hard-coded metrics.
 */
export function draftProbabilityGateEvidence(
  artifact: unknown,
): DraftProbabilityGateEvidence | null {
  const root = record(artifact);
  const population = record(root?.population);
  const splits = record(root?.splits);
  const draftSplits = records(splits?.draft);
  const finalSplit = draftSplits?.find((split) => split.name === "final") ?? null;
  const draft = record(root?.draft_probability);
  const artifactSha256 = draft?.artifact_sha256;
  const modelCodeSha256 = draft?.model_code_sha256;
  const trainingPopulationSha256 = draft?.training_population_sha256;
  const finalTest = record(draft?.final_test);
  const compositionModelId =
    typeof draft?.best_composition_candidate === "string"
      ? draft.best_composition_candidate
      : null;
  const base = record(finalTest?.baseline_overall_blue_rate);
  const candidate =
    compositionModelId == null ? null : record(finalTest?.[compositionModelId]);

  const populationMaps = positiveInteger(population?.maps);
  const finalTestMaps = positiveInteger(finalSplit?.maps);
  const baseMaps = positiveInteger(base?.maps);
  const candidateMaps = positiveInteger(candidate?.maps);
  const compositionLogLoss = finite(candidate?.log_loss);
  const overallBaseRateLogLoss = finite(base?.log_loss);
  const compositionBrier = finite(candidate?.brier);
  const overallBaseRateBrier = finite(base?.brier);
  const finalTestStart = finalSplit?.date_min;
  const finalTestEnd = finalSplit?.date_max;

  if (
    root?.schema_version !== "1.0.0" ||
    !validClock(root.created_utc) ||
    draft?.gate_status !== "failed" ||
    draft?.evaluation_scope !==
      "chronological_pre_final_fit_scored_on_untouched_final" ||
    draft?.release_artifact_role !==
      "full_population_refit_for_experimental_policy_utility_only" ||
    draft?.release_refit_includes_final_labels !== true ||
    draft?.decision !==
      "withhold_numeric_probability; publish pooled composition policy utility only" ||
    draft?.selection_winner !== "baseline_overall_blue_rate" ||
    draft?.winner_is_composition !== false ||
    !sha256(artifactSha256) ||
    !sha256(modelCodeSha256) ||
    !sha256(trainingPopulationSha256) ||
    compositionModelId == null ||
    populationMaps == null ||
    finalTestMaps == null ||
    baseMaps !== finalTestMaps ||
    candidateMaps !== finalTestMaps ||
    finalTestMaps > populationMaps ||
    !validClock(finalTestStart) ||
    !validClock(finalTestEnd) ||
    Date.parse(finalTestStart) > Date.parse(finalTestEnd) ||
    compositionLogLoss == null ||
    overallBaseRateLogLoss == null ||
    compositionBrier == null ||
    overallBaseRateBrier == null ||
    compositionLogLoss < 0 ||
    overallBaseRateLogLoss < 0 ||
    compositionBrier < 0 ||
    compositionBrier > 1 ||
    overallBaseRateBrier < 0 ||
    overallBaseRateBrier > 1 ||
    compositionLogLoss <= overallBaseRateLogLoss ||
    compositionBrier <= overallBaseRateBrier
  ) {
    return null;
  }

  return {
    schemaVersion: root.schema_version,
    artifactCreatedUtc: root.created_utc,
    populationMaps,
    compositionModelId,
    artifactSha256,
    modelCodeSha256,
    trainingPopulationSha256,
    finalTestMaps,
    finalTestStart,
    finalTestEnd,
    compositionLogLoss,
    overallBaseRateLogLoss,
    compositionBrier,
    overallBaseRateBrier,
    decision: "withhold_numeric_probability",
    gateStatus: "failed",
    evaluationScope:
      "chronological_pre_final_fit_scored_on_untouched_final",
    releaseArtifactRole:
      "full_population_refit_for_experimental_policy_utility_only",
    releaseRefitIncludesFinalLabels: true,
  };
}

/** Validate the immutable fields quoted by the public team methodology. */
export function teamRatingGateEvidence(
  artifact: unknown,
): TeamRatingGateEvidence | null {
  const root = record(artifact);
  const team = record(root?.team_rating);
  const finalTest = record(team?.final_test);
  const formats = record(finalTest?.format_stratified_calibration);
  const bo5 = record(formats?.Bo5);
  const comparison = record(team?.paired_primary_comparison);
  const bootstrap = record(comparison?.bootstrap);
  const interval = Array.isArray(comparison?.confidence_interval)
    ? comparison.confidence_interval
    : null;
  const codeHash = team?.model_code_sha256;
  const configHash = team?.model_config_sha256;
  const modelVersion = team?.model_version;
  const finalTestSeries = positiveInteger(finalTest?.series);
  const comparisonSeries = positiveInteger(comparison?.series);
  const logLoss = finite(finalTest?.log_loss);
  const brier = finite(finalTest?.brier);
  const ece = finite(finalTest?.ece_10_equal_width);
  const rollingEloLogLoss = finite(comparison?.baseline_score);
  const candidateScore = finite(comparison?.candidate_score);
  const logLossDifference = finite(comparison?.candidate_minus_baseline);
  const ciLow = finite(interval?.[0]);
  const ciHigh = finite(interval?.[1]);
  const bootstrapReplicates = positiveInteger(bootstrap?.replicates);
  const bootstrapBlockSeries = positiveInteger(
    bootstrap?.block_size_series,
  );
  const bo5Series = positiveInteger(bo5?.n);
  const bo5Ece = finite(bo5?.ece);
  const close = (left: number, right: number) =>
    Math.abs(left - right) <= 1e-12;

  if (
    root?.schema_version !== "1.0.0" ||
    !validClock(root?.created_utc) ||
    team?.model_id !== "series_dynamic_bt" ||
    !sha256(codeHash) ||
    !sha256(configHash) ||
    !sha256(team?.observation_rows_sha256) ||
    modelVersion !==
      `series_dynamic_bt:${codeHash.slice(0, 12)}:${configHash.slice(0, 12)}` ||
    team?.gate_status !== "passed" ||
    finalTestSeries == null ||
    comparisonSeries !== finalTestSeries ||
    logLoss == null ||
    brier == null ||
    brier < 0 ||
    brier > 1 ||
    ece == null ||
    ece < 0 ||
    ece > 1 ||
    comparison?.baseline_model_id !== "rolling_series_elo" ||
    comparison?.score !== "log_loss" ||
    comparison?.decision !== "superior" ||
    rollingEloLogLoss == null ||
    candidateScore == null ||
    logLossDifference == null ||
    !close(candidateScore, logLoss) ||
    !close(candidateScore - rollingEloLogLoss, logLossDifference) ||
    ciLow == null ||
    ciHigh == null ||
    ciLow > logLossDifference ||
    ciHigh < logLossDifference ||
    ciHigh >= 0 ||
    comparison?.confidence_level !== 0.95 ||
    bootstrapReplicates !== 5000 ||
    bootstrapBlockSeries !== 8 ||
    bo5Series == null ||
    bo5Ece == null ||
    bo5Ece < 0 ||
    bo5Ece > 1
  ) {
    return null;
  }

  return {
    modelVersion,
    finalTestSeries,
    logLoss,
    brier,
    ece,
    rollingEloLogLoss,
    logLossDifference,
    confidenceInterval: [ciLow, ciHigh],
    bootstrapReplicates,
    bootstrapBlockSeries,
    bo5Series,
    bo5Ece,
  };
}
