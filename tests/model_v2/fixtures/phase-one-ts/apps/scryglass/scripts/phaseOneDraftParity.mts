/** Outcome-free TypeScript replay for the frozen phase-one Draft receipts. */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  loadTerminalModelArtifact,
  scoreTerminalDraft,
  terminalDraftFromSides,
  type TerminalAction,
  type TerminalAssignment,
  type TerminalSide,
} from "../src/lib/draftTerminalScore";

type JsonObject = Record<string, unknown>;

function fail(message: string): never {
  throw new Error(message);
}

function object(value: unknown, label: string): JsonObject {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    fail(`${label} must be an object`);
  }
  return value as JsonObject;
}

function sigmoid(value: number): number {
  if (value >= 40) return 1;
  if (value <= -40) return 0;
  return 1 / (1 + Math.exp(-value));
}

function logit(value: number): number {
  if (!(value > 0 && value < 1)) fail("rating probability is outside (0,1)");
  return Math.log(value / (1 - value));
}

function main(): void {
  if (process.argv.length !== 4) {
    fail("usage: phaseOneDraftParity.mts <repo-root> <joint-snapshot>");
  }
  const root = resolve(process.argv[2]);
  const snapshotPath = resolve(root, process.argv[3]);
  const snapshot = object(JSON.parse(readFileSync(snapshotPath, "utf8")), "snapshot");
  const ledger = object(snapshot.draft_ledger_candidate, "draft ledger");
  if (!Array.isArray(ledger.entries)) fail("draft ledger entries must be an array");

  const comparisons = ledger.entries.map((entryValue: unknown) => {
    const entry = object(entryValue, "draft entry");
    const predictionPath = resolve(root, String(entry.prediction_locator));
    const prediction = object(JSON.parse(readFileSync(predictionPath, "utf8")), "prediction");
    const inputs = object(prediction.input_receipts, "prediction inputs");
    const metadata = object(object(inputs.draft_metadata, "draft metadata receipt").value, "draft metadata");
    const ratings = object(object(inputs.ratings_prediction, "ratings receipt").value, "ratings prediction");
    const ratingPrediction = object(
      object(ratings.evaluation_predictions, "ratings predictions")["hierarchical-orgw100-orgv025-retain100"],
      "rating candidate",
    );
    const modelRecord = object(prediction.model, "model record");
    const modelRaw = readFileSync(resolve(root, String(modelRecord.artifact_locator)));
    const model = loadTerminalModelArtifact(modelRaw, {
      expectedArtifactSha256: String(modelRecord.artifact_raw_sha256),
    });

    const blueIsA = String(metadata.blue_organization_id) < String(metadata.red_organization_id);
    const sideA = object(blueIsA ? metadata.blue : metadata.red, "side A") as TerminalSide;
    const sideB = object(blueIsA ? metadata.red : metadata.blue, "side B") as TerminalSide;
    const sideMap: Record<string, "A" | "B"> = blueIsA
      ? { blue: "A", red: "B" }
      : { blue: "B", red: "A" };
    if (!Array.isArray(metadata.actions) || !Array.isArray(metadata.final_assignments)) {
      fail("terminal actions or assignments are missing");
    }
    const actions = metadata.actions.map((item: JsonObject) => ({
      action_id: String(item.action_id),
      slot: Number(item.slot),
      kind: item.kind as TerminalAction["kind"],
      canonical_side: sideMap[String(item.side)],
      champion_id: String(item.champion_id),
      role_set: item.role_set as TerminalAction["role_set"],
    })) as TerminalAction[];
    const finalAssignments = metadata.final_assignments.map((item: JsonObject) => ({
      action_id: String(item.action_id),
      canonical_side: sideMap[String(item.side)],
      champion_id: String(item.champion_id),
      role: item.role as TerminalAssignment["role"],
    })) as TerminalAssignment[];
    const source = object(metadata.source, "draft source");
    // Terminal Draft timing is map-specific.  The ratings receipt may carry the
    // series' scheduled start, which is earlier than later-map draft capture.
    // Phase one binds an independently captured actual map start on each ledger
    // entry, so replay against that exact boundary instead.
    const eventStart = String(entry.actual_map_start_utc);
    const draft = terminalDraftFromSides({
      sideA,
      sideB,
      eventStart,
      sourceAvailableAt: String(source.available_at_utc),
      sourceRecordId: String(source.source_record_id),
      sourcePayloadSha256: String(source.payload_raw_sha256),
      sourceRightsStatus: source.rights_status as "reviewed" | "unknown",
      mode: "neutral",
      actions,
      finalAssignments,
    });
    const replay = object(scoreTerminalDraft(draft, model, { development: true }), "TypeScript replay");
    const index = object(prediction.draft_index, "draft index");
    const predictions = object(prediction.evaluation_predictions, "evaluation predictions");
    const combined = object(predictions.ratings_plus_draft, "combined prediction");
    const tsDraftPA = Number(replay.standardized_map_win_probability_a);
    const tsScaledBlue = blueIsA
      ? Number(replay.calibration_slope ?? model.calibrationSlope) * Number(replay.uncalibrated_logit_a)
      : -Number(replay.calibration_slope ?? model.calibrationSlope) * Number(replay.uncalibrated_logit_a);
    const tsCombinedBlue = sigmoid(logit(Number(ratingPrediction.p_blue)) + tsScaledBlue);
    const pyDraftPA = Number(index.equal_strength_index_a);
    const pyCombinedBlue = Number(combined.p_blue);
    return {
      event_id: String(entry.event_id),
      series_id: String(entry.series_id),
      game_number: Number(entry.game_number),
      prediction_artifact_sha256: String(entry.prediction_artifact_sha256),
      python_draft_index_probability_a: pyDraftPA,
      typescript_draft_index_probability_a: tsDraftPA,
      draft_index_absolute_delta: Math.abs(pyDraftPA - tsDraftPA),
      python_combined_probability_blue: pyCombinedBlue,
      typescript_combined_probability_blue: tsCombinedBlue,
      combined_absolute_delta: Math.abs(pyCombinedBlue - tsCombinedBlue),
    };
  });
  comparisons.sort((a: JsonObject, b: JsonObject) =>
    `${a.event_id}|${a.game_number}`.localeCompare(`${b.event_id}|${b.game_number}`),
  );
  process.stdout.write(`${JSON.stringify({
    schema_version: "scryglass:phase-one-draft-typescript-replay:v1",
    snapshot_artifact_sha256: String(snapshot.artifact_sha256),
    comparisons,
  })}\n`);
}

main();
