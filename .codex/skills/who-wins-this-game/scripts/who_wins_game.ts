import { readFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROLES = ["top", "jng", "mid", "bot", "sup"] as const;
type Role = (typeof ROLES)[number];
type Side = "blue" | "red";

function parseArgs(argv: string[]): Record<string, string> {
  const args: Record<string, string> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      args[key] = "true";
      continue;
    }
    args[key] = value;
    index += 1;
  }
  return args;
}

function picks(value: string | undefined, label: string): string[] {
  const parsed = (value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (parsed.length !== 5) {
    throw new Error(`${label} requires exactly five comma-separated champions`);
  }
  return parsed;
}

function round(value: number, digits = 4): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function decimalOdds(value: string | undefined, label: string): number | null {
  if (value == null) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 1) {
    throw new Error(`${label} must be decimal odds greater than 1.00`);
  }
  return parsed;
}

function findPlayer(context: any, requested: string): any {
  const key = Object.keys(context.players ?? {}).find(
    (candidate) => candidate.toLowerCase() === requested.toLowerCase(),
  );
  if (!key) throw new Error(`player not found in context: ${requested}`);
  return context.players[key];
}

function findTeam(context: any, requested: string): any | null {
  const normalized = requested.trim().toLowerCase();
  const teams = context.teams ?? [];
  const exact = teams.find(
    (team: any) => String(team.team ?? "").trim().toLowerCase() === normalized,
  );
  if (exact) return exact;
  const prefixMatches = teams.filter((team: any) =>
    String(team.team ?? "").trim().toLowerCase().startsWith(normalized),
  );
  return prefixMatches.length === 1 ? prefixMatches[0] : null;
}

function lineupRating(
  players: Partial<Record<Role, any>>,
): { rating: number | null; missing_roles: Role[]; players: any[] } {
  const missingRoles = ROLES.filter((role) => {
    const rating = Number(players[role]?.rating);
    return !Number.isFinite(rating);
  });
  const rows = ROLES.flatMap((role) => {
    const player = players[role];
    return player
      ? [{ role, player: player.player, rating: Number(player.rating) }]
      : [];
  });
  if (missingRoles.length > 0) {
    return { rating: null, missing_roles: missingRoles, players: rows };
  }
  return {
    rating: rows.reduce((sum, row) => sum + row.rating, 0) / ROLES.length,
    missing_roles: [],
    players: rows,
  };
}

function probabilityView(blueProbability: number): any {
  return {
    blue_pct: round(100 * blueProbability, 2),
    red_pct: round(100 * (1 - blueProbability), 2),
  };
}

function minimumBettableOdds(probability: number): number | null {
  if (probability <= 0.03 || probability >= 1) return null;
  return Math.max(1.05 / probability, 1 / (probability - 0.03));
}

function offeredOddsView(
  probability: number,
  offered: number | null,
  minimum: number | null,
  classificationAvailable: boolean,
): any | null {
  if (offered == null) return null;
  const implied = 1 / offered;
  return {
    offered_odds: round(offered, 3),
    break_even_pct: round(100 * implied, 2),
    model_edge_pp: round(100 * (probability - implied), 2),
    expected_return_pct: round(100 * (offered * probability - 1), 2),
    bettable:
      classificationAvailable &&
      minimum != null &&
      offered + 1e-12 >= minimum,
  };
}

function resolvePlayerContext(
  args: Record<string, string>,
  context: any,
  bluePicks: string[],
  redPicks: string[],
  blueName: string,
  redName: string,
): {
  blue: Partial<Record<Role, any>>;
  red: Partial<Record<Role, any>>;
  evidence: any[];
  applied: boolean;
} {
  const output: {
    blue: Partial<Record<Role, any>>;
    red: Partial<Record<Role, any>>;
    evidence: any[];
    applied: boolean;
  } = { blue: {}, red: {}, evidence: [], applied: false };
  for (const side of ["blue", "red"] as const) {
    const teamName = side === "blue" ? blueName : redName;
    const team = findTeam(context, teamName);
    for (const role of ROLES) {
      const requested = args[`${side}-${role}-player`];
      const rosterPlayer = team?.roster?.find(
        (candidate: any) => candidate.role === role,
      );
      if (!requested && !rosterPlayer) continue;
      const player = requested
        ? findPlayer(context, requested)
        : findPlayer(context, rosterPlayer.player);
      output[side][role] = player;
      output.applied = true;
      const championIndex = ROLES.indexOf(role);
      const champion = (side === "blue" ? bluePicks : redPicks)[championIndex];
      const mastery = player.mastery?.[champion] ?? null;
      output.evidence.push({
        side,
        role,
        player: player.player,
        champion,
        maps: mastery?.n ?? 0,
        effective_maps: mastery?.effective_n ?? 0,
        logit: mastery?.logit ?? 0,
        status: mastery ? "observed" : "neutral_prior",
        source: requested ? "explicit_override" : "team_roster",
      });
    }
  }
  return output;
}

function synergyKey(first: string, second: string): string {
  return [first, second].sort((a, b) => a.localeCompare(b)).join("|");
}

function archetypes(runtime: any, champion: string): string[] {
  return runtime.champion_archetypes?.[champion] ?? [];
}

function synergyRows(runtime: any, team: string[]): any[] {
  const rows: any[] = [];
  for (let firstIndex = 0; firstIndex < team.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < team.length; secondIndex += 1) {
      const first = team[firstIndex];
      const second = team[secondIndex];
      let value = Number(runtime.ally_synergy?.[synergyKey(first, second)]?.logit ?? 0);
      for (const firstTag of archetypes(runtime, first)) {
        for (const secondTag of archetypes(runtime, second)) {
          value += Number(
            runtime.archetype_synergy?.[synergyKey(firstTag, secondTag)]?.logit ?? 0,
          );
        }
      }
      if (value !== 0) rows.push({ champions: [first, second], logit: round(value, 5) });
    }
  }
  return rows
    .sort((left, right) => Math.abs(right.logit) - Math.abs(left.logit))
    .slice(0, 4);
}

function counterRows(runtime: any, attackers: string[], defenders: string[]): any[] {
  const rows: any[] = [];
  for (const attacker of attackers) {
    for (const defender of defenders) {
      let value = Number(runtime.counter_pairs?.[`${attacker}|${defender}`]?.logit ?? 0);
      for (const attackerTag of archetypes(runtime, attacker)) {
        for (const defenderTag of archetypes(runtime, defender)) {
          value += Number(
            runtime.archetype_counters?.[`${attackerTag}|${defenderTag}`]?.logit ?? 0,
          );
        }
      }
      if (value !== 0) {
        rows.push({ champions: [attacker, defender], logit: round(value, 5) });
      }
    }
  }
  return rows
    .sort((left, right) => Math.abs(right.logit) - Math.abs(left.logit))
    .slice(0, 5);
}

function sideNeutral(first: any, secondAsBlue: any): any {
  const probability =
    (first.p_blue_draft + (1 - secondAsBlue.p_blue_draft)) / 2;
  const components = {
    base:
      0.5 *
      ((first.components.win_logit_blue -
        0.03 -
        first.components.win_logit_red) -
        (secondAsBlue.components.win_logit_blue -
          0.03 -
          secondAsBlue.components.win_logit_red)),
    synergy:
      0.5 *
      ((first.components.synergy_logit_blue -
        first.components.synergy_logit_red) -
        (secondAsBlue.components.synergy_logit_blue -
          secondAsBlue.components.synergy_logit_red)),
    counter:
      0.5 *
      (first.components.counter_logit - secondAsBlue.components.counter_logit),
    same_role:
      0.5 *
      (first.components.pair_logit - secondAsBlue.components.pair_logit),
    player_comfort:
      0.5 *
      ((first.components.player_logit_blue -
        first.components.player_logit_red) -
        (secondAsBlue.components.player_logit_blue -
          secondAsBlue.components.player_logit_red)),
  };
  const total = Object.values(components).reduce(
    (sum: number, value) => sum + Number(value),
    0,
  );
  const confidence = 0.5 * (first.confidence + secondAsBlue.confidence);
  const dpPerLogit = 0.5 * (
    first.calibration.dp_per_logit + secondAsBlue.calibration.dp_per_logit
  );
  return {
    first_pct: round(100 * probability, 2),
    second_pct: round(100 * (1 - probability), 2),
    edge_points: round(100 * (2 * probability - 1), 2),
    wr_bump_pp: round(100 * dpPerLogit * total * confidence, 2),
    confidence: round(confidence, 3),
    components: Object.fromEntries(
      [...Object.entries(components), ["total", total]].map(([key, value]) => [
        key,
        round(Number(value), 5),
      ]),
    ),
  };
}

function actualBlue(score: any): any {
  return {
    blue_pct: round(100 * score.p_blue_draft, 2),
    red_pct: round(100 * (1 - score.p_blue_draft), 2),
    edge_points: round(score.draft_edge, 2),
    wr_bump_pp: round(score.wr_bump_pp, 2),
    confidence: score.confidence,
    components: score.components,
  };
}

function compactActual(score: any): any {
  return {
    blue_pct: score.blue_pct,
    red_pct: score.red_pct,
    edge_points: score.edge_points,
    wr_bump_pp: score.wr_bump_pp,
    drivers: {
      base: round(
        score.components.win_logit_blue -
          0.03 -
          score.components.win_logit_red,
        5,
      ),
      synergy: round(
        score.components.synergy_logit_blue -
          score.components.synergy_logit_red,
        5,
      ),
      counter: score.components.counter_logit,
      same_role: score.components.pair_logit,
      player_comfort: round(
        score.components.player_logit_blue -
          score.components.player_logit_red,
        5,
      ),
      total: score.components.win_edge,
    },
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const repo = path.resolve(args.repo ?? "/Users/river/scryglass");
  const app = path.join(repo, "apps", "lol-atlas");
  const scoreModule = path.join(app, "src", "lib", "draftScore.ts");
  const runtimePath = path.join(app, "data", "draft", "runtime.json");
  const blue = picks(args.blue, "blue");
  const red = picks(args.red, "red");
  const blueName = args["blue-name"] ?? "Blue";
  const redName = args["red-name"] ?? "Red";
  const league = args.league ?? null;

  process.chdir(app);
  const scoring = await import(pathToFileURL(scoreModule).href);
  const runtime = JSON.parse(readFileSync(runtimePath, "utf8"));
  const roles = [...ROLES];

  const context = scoring.draftContext();
  const players = resolvePlayerContext(
    args,
    context,
    blue,
    red,
    blueName,
    redName,
  );
  const playerContext = players.applied
    ? { blue: players.blue, red: players.red }
    : undefined;
  const blueTeam = findTeam(context, blueName);
  const redTeam = findTeam(context, redName);
  const blueLineup = lineupRating(players.blue);
  const redLineup = lineupRating(players.red);
  const blueTeamRating = Number(blueTeam?.rating);
  const redTeamRating = Number(redTeam?.rating);
  const missingStrengthInputs = [
    ...(!Number.isFinite(blueTeamRating) ? [`blue team: ${blueName}`] : []),
    ...(!Number.isFinite(redTeamRating) ? [`red team: ${redName}`] : []),
    ...blueLineup.missing_roles.map((role) => `blue lineup role: ${role}`),
    ...redLineup.missing_roles.map((role) => `red lineup role: ${role}`),
  ];
  const strengthAvailable = missingStrengthInputs.length === 0;
  const teamEloDiff = strengthAvailable
    ? blueTeamRating - redTeamRating
    : null;
  const playerEloDiff =
    strengthAvailable &&
    blueLineup.rating != null &&
    redLineup.rating != null
      ? blueLineup.rating - redLineup.rating
      : null;
  const primaryBlue = scoring.draftScore({
    blue,
    red,
    league,
    blue_roles: roles,
    red_roles: roles,
    player_context: playerContext,
    team_elo_diff: teamEloDiff,
    player_elo_diff: playerEloDiff,
  });
  const strengthProbability = strengthAvailable
    ? sigmoid(primaryBlue.components.strength_logit)
    : null;
  const compositeProbability = strengthAvailable
    ? Number(primaryBlue.p_blue_combined)
    : null;
  const blueFairOdds =
    compositeProbability != null ? 1 / compositeProbability : null;
  const redFairOdds =
    compositeProbability != null ? 1 / (1 - compositeProbability) : null;
  const blueMinimumOdds =
    compositeProbability != null
      ? minimumBettableOdds(compositeProbability)
      : null;
  const redMinimumOdds =
    compositeProbability != null
      ? minimumBettableOdds(1 - compositeProbability)
      : null;
  const blueOfferedOdds = decimalOdds(args["blue-odds"], "blue odds");
  const redOfferedOdds = decimalOdds(args["red-odds"], "red odds");
  const market = (args.market ?? "map").trim().toLowerCase();
  const marketState = (args["market-state"] ?? "post-draft-pregame")
    .trim()
    .toLowerCase();
  const supportedMarket = ["map", "map-winner", "single-map"].includes(market);
  const supportedState = ["pregame", "post-draft-pregame"].includes(marketState);
  const classificationAvailable =
    strengthAvailable && supportedMarket && supportedState;

  const roleEvidence = {
    blue: Object.fromEntries(
      blue.map((champion, index) => [
        ROLES[index],
        {
          champion,
          champion_maps: runtime.champ_game_counts?.[champion] ?? 0,
          role_maps:
            runtime.champion_role_counts?.[champion]?.[ROLES[index]] ?? 0,
        },
      ]),
    ),
    red: Object.fromEntries(
      red.map((champion, index) => [
        ROLES[index],
        {
          champion,
          champion_maps: runtime.champ_game_counts?.[champion] ?? 0,
          role_maps:
            runtime.champion_role_counts?.[champion]?.[ROLES[index]] ?? 0,
        },
      ]),
    ),
  };

  const fullOutput = {
    teams: {
      blue: blueTeam?.team ?? blueName,
      red: redTeam?.team ?? redName,
    },
    picks: { blue, red },
    league,
    runtime_as_of: runtime.as_of ?? null,
    model_maps: runtime.n_maps ?? null,
    draft_score: {
      actual_blue: actualBlue(primaryBlue),
      player_context_applied: players.applied,
      player_evidence: players.evidence,
    },
    winning_expectation: strengthAvailable
      ? {
          ...probabilityView(strengthProbability as number),
          source: primaryBlue.calibration.strength_source,
          team_only: probabilityView(
            Number(primaryBlue.calibration.p_team_strength),
          ),
          lineup_only: probabilityView(
            Number(primaryBlue.calibration.p_player_strength),
          ),
        }
      : null,
    composite: compositeProbability != null
      ? probabilityView(compositeProbability)
      : null,
    odds: {
      fair: compositeProbability != null
        ? {
            blue: round(blueFairOdds as number, 3),
            red: round(redFairOdds as number, 3),
          }
        : null,
      minimum_bettable: compositeProbability != null
        ? {
            blue:
              blueMinimumOdds != null ? round(blueMinimumOdds, 3) : null,
            red:
              redMinimumOdds != null ? round(redMinimumOdds, 3) : null,
          }
        : null,
      policy: {
        min_expected_return_pct: 5,
        min_probability_edge_pp: 3,
        learned_threshold: false,
      },
    },
    market_check: {
      market,
      state: marketState,
      classification_available: classificationAvailable,
      unavailable_reason: classificationAvailable
        ? null
        : !strengthAvailable
          ? `unidentified strength inputs: ${missingStrengthInputs.join(", ")}`
          : "bettable classification is limited to a completed-draft, pregame, single-map winner market",
      blue:
        compositeProbability != null
          ? offeredOddsView(
              compositeProbability,
              blueOfferedOdds,
              blueMinimumOdds,
              classificationAvailable,
            )
          : null,
      red:
        compositeProbability != null
          ? offeredOddsView(
              1 - compositeProbability,
              redOfferedOdds,
              redMinimumOdds,
              classificationAvailable,
            )
          : null,
      bookmaker_overround_pct:
        blueOfferedOdds != null && redOfferedOdds != null
          ? round(
              100 *
                (1 / blueOfferedOdds + 1 / redOfferedOdds - 1),
              2,
            )
          : null,
    },
    strength_evidence: {
      available: strengthAvailable,
      unavailable_reason: strengthAvailable
        ? null
        : `unidentified strength inputs: ${missingStrengthInputs.join(", ")}`,
      team_rating: strengthAvailable
        ? {
            blue: round(blueTeamRating, 2),
            red: round(redTeamRating, 2),
            difference: round(teamEloDiff as number, 2),
          }
        : null,
      lineup_rating: strengthAvailable
        ? {
            blue: round(blueLineup.rating as number, 2),
            red: round(redLineup.rating as number, 2),
            difference: round(playerEloDiff as number, 2),
          }
        : null,
      lineups: {
        blue: blueLineup.players,
        red: redLineup.players,
      },
    },
    role_evidence: roleEvidence,
    top_synergies: {
      blue: synergyRows(runtime, blue),
      red: synergyRows(runtime, red),
    },
    top_counters: {
      blue_into_red: counterRows(runtime, blue, red),
      red_into_blue: counterRows(runtime, red, blue),
    },
  };

  const thinRoles = (["blue", "red"] as const).flatMap((side) =>
    Object.entries(roleEvidence[side])
      .filter(([, evidence]: [string, any]) => evidence.role_maps < 25)
      .map(([role, evidence]) => ({ side, role, ...evidence })),
  );
  const compactOutput = {
    teams: fullOutput.teams,
    picks: fullOutput.picks,
    league,
    runtime_as_of: fullOutput.runtime_as_of,
    draft_score: {
      actual_blue: compactActual(fullOutput.draft_score.actual_blue),
      player_context_applied: fullOutput.draft_score.player_context_applied,
      player_evidence: fullOutput.draft_score.player_evidence,
    },
    winning_expectation: fullOutput.winning_expectation,
    composite: fullOutput.composite,
    odds: fullOutput.odds,
    market_check: fullOutput.market_check,
    strength_evidence: fullOutput.strength_evidence,
    thin_roles: thinRoles,
    top_synergies: {
      blue: fullOutput.top_synergies.blue.slice(0, 2),
      red: fullOutput.top_synergies.red.slice(0, 2),
    },
    top_counters: {
      blue_into_red: fullOutput.top_counters.blue_into_red.slice(0, 3),
      red_into_blue: fullOutput.top_counters.red_into_blue.slice(0, 3),
    },
  };

  console.log(
    JSON.stringify(args.details === "true" ? fullOutput : compactOutput, null, 2),
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
