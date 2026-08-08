import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

type Request = {
  league: string;
  blue_team: string;
  red_team: string;
  blue_picks: string[];
  red_picks: string[];
  event_id?: string;
  event_start?: string;
  draft_source_available_at?: string;
};

async function main(): Promise<void> {
  const request = JSON.parse(readFileSync(0, "utf8")) as Request;
  if (!request.event_start || !request.draft_source_available_at) {
    process.stdout.write(JSON.stringify({
      available: false,
      authorized: false,
      claim_ceiling: "research_diagnostic_only",
      unavailable_reason: "Pregame diagnostic requires verified event_start and draft_source_available_at timestamps",
      blockers: [
        "event_start_missing",
        "draft_source_available_at_missing",
        "pre_event_draft_provenance_unavailable",
      ],
    }));
    return;
  }
  const repoRoot = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "..",
  );
  const appPath = path.join(repoRoot, "apps", "lol-atlas");
  process.chdir(appPath);
  const terminalModulePath = path.join(appPath, "src", "lib", "draftTerminalScore.ts");
  const terminalModelPath = path.join(
    repoRoot,
    "data",
    "lol",
    "v2",
    "models",
    "draft-terminal",
    "terminal-model-neutral-development-v1.json",
  );
  const terminal = await import(pathToFileURL(terminalModulePath).href);
  const terminalModelRaw = readFileSync(terminalModelPath);
  const terminalModelSha256 = createHash("sha256").update(terminalModelRaw).digest("hex");
  const terminalModel = terminal.loadTerminalModelArtifact(terminalModelRaw, {
    expectedArtifactSha256: terminalModelSha256,
  });
  const sourcePayload = Buffer.from(JSON.stringify({
    blue_picks: request.blue_picks,
    red_picks: request.red_picks,
    event_start: request.event_start,
    draft_source_available_at: request.draft_source_available_at,
    event_id: request.event_id ?? null,
  }), "utf8");
  const sourcePayloadSha256 = createHash("sha256").update(sourcePayload).digest("hex");
  const terminalDraft = terminal.terminalDraftFromSides({
    sideA: {
      top: request.blue_picks[0],
      jungle: request.blue_picks[1],
      mid: request.blue_picks[2],
      bot: request.blue_picks[3],
      support: request.blue_picks[4],
    },
    sideB: {
      top: request.red_picks[0],
      jungle: request.red_picks[1],
      mid: request.red_picks[2],
      bot: request.red_picks[3],
      support: request.red_picks[4],
    },
    eventStart: request.event_start,
    sourceAvailableAt: request.draft_source_available_at,
    sourceRecordId: request.event_id
      ? `manual-event:${request.event_id}`
      : "manual-event:unbound",
    sourcePayloadSha256,
    sourceRightsStatus: "unknown",
    mode: "neutral",
  });
  const terminalScore = terminal.scoreTerminalDraft(
    terminalDraft,
    terminalModel,
    { development: true },
  );
  if (terminalScore.status !== "development_only") {
    throw new Error("terminal Draft Score development replay is unavailable");
  }
  const draftProbability = Number(terminalScore.standardized_map_win_probability_a);

  process.stdout.write(
    JSON.stringify({
      available: true,
      authorized: false,
      match_probability_available: false,
      claim_ceiling: "research_diagnostic_only",
      runtime_as_of: null,
      p_blue: null,
      p_red: null,
      unavailable_reason: "Match probability withheld until exact pre-event rosters and independently validated player and team ratings are registered",
      draft_score: {
        status: terminalScore.status,
        model_kind: "canonical_terminal_neutral",
        model_version: terminalScore.model_version,
        model_artifact_sha256: terminalModelSha256,
        blue: draftProbability,
        red: 1 - draftProbability,
        interval_95: terminalScore.interval_95,
        ledger: terminalScore.ledger,
        claim_ceiling: terminalScore.claim_ceiling,
      },
      strength_expectation: {
        status: "unavailable",
        blue: null,
        red: null,
        team_rating_authorized: false,
        player_rating_authorized: false,
        pre_event_roster_authorized: false,
        blockers: [
          "independently_validated_team_rating_unavailable",
          "independently_validated_player_rating_unavailable",
          "pre_event_roster_authority_unavailable",
        ],
      },
      source: "canonical terminal neutral Draft Score development diagnostic only; match probability unavailable",
    }),
  );
}

main().catch((error) => {
  process.stderr.write(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
