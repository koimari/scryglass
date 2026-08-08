import { NextResponse } from "next/server";
import {
  DRAFT_UNAVAILABLE_RESPONSE,
  publicPredictiveDraftsEnabled,
} from "@/lib/publicDraftGate";
import {
  DRAFT_ROLES,
  analyzeDraftSandbox,
  draftCatalog,
  normalizeDraftChampion,
  type DraftAction,
  type DraftCandidateRole,
  type DraftContextPlayer,
  type DraftPlayerContext,
  type DraftRole,
  type DraftSide,
} from "@/lib/draftScore";
import { readCurrentDraftContext } from "@/lib/draftServer";

export const runtime = "nodejs";

const unavailable = () =>
  NextResponse.json(
    DRAFT_UNAVAILABLE_RESPONSE,
    { status: 503 },
  );

type Body = {
  actions?: Array<{
    side?: string;
    champion?: string;
    role?: string | null;
  }>;
  perspective?: string;
  next_side?: string;
  candidate_role?: string;
  excluded?: string[];
  league?: string | null;
  elo_diff?: number | null;
  blue_team?: string | null;
  red_team?: string | null;
  blue_players?: Partial<Record<DraftRole, string>>;
  red_players?: Partial<Record<DraftRole, string>>;
  limit?: number;
};

function isSide(value: unknown): value is DraftSide {
  return value === "blue" || value === "red";
}

function isRole(value: unknown): value is DraftRole {
  return DRAFT_ROLES.includes(value as DraftRole);
}

function isCandidateRole(value: unknown): value is DraftCandidateRole {
  return value === "open" || value === "any" || isRole(value);
}

function canonicalChampion(value: string): string | null {
  const normalized = normalizeDraftChampion(value);
  return (
    draftCatalog().find(
      (champion) => champion.name.toLocaleLowerCase() === normalized.toLocaleLowerCase(),
    )?.name ?? null
  );
}

export async function GET() {
  if (!publicPredictiveDraftsEnabled()) return unavailable();
  /* Development-only implementation retained below for reproducibility work. */
  /* istanbul ignore next */
  const context = await readCurrentDraftContext();
  return NextResponse.json({
    champions: draftCatalog(),
    roles: DRAFT_ROLES,
    teams: context.teams.map((team) => ({
      team: team.team,
      league: team.league,
      tier: team.tier,
      rating: team.rating,
      roster: team.roster.map(({ player, role, rating, n_maps }) => ({
        player,
        role,
        rating,
        n_maps,
      })),
    })),
    model: "Draft recommendation v4 · interaction + strength context",
    context_as_of: context.as_of,
  });
}

function selectedTeam(
  context: Awaited<ReturnType<typeof readCurrentDraftContext>>,
  name: string | null | undefined,
) {
  if (!name) return null;
  return (
    context.teams.find(
      (team) => team.team.toLocaleLowerCase() === name.toLocaleLowerCase(),
    ) ?? null
  );
}

function lineup(
  context: Awaited<ReturnType<typeof readCurrentDraftContext>>,
  names: Partial<Record<DraftRole, string>> | undefined,
  teamName: string | null | undefined,
): Partial<Record<DraftRole, DraftContextPlayer>> {
  const team = selectedTeam(context, teamName);
  const output: Partial<Record<DraftRole, DraftContextPlayer>> = {};
  for (const role of DRAFT_ROLES) {
    const requested = names?.[role];
    const player = requested
      ? context.players[requested]
      : team?.roster.find((candidate) => candidate.role === role);
    if (!player) continue;
    if (team && player.team !== team.team) continue;
    output[role] = player;
  }
  return output;
}

function lineupRating(players: Partial<Record<DraftRole, DraftContextPlayer>>): number | null {
  const ratings = Object.values(players)
    .map((player) => player?.rating)
    .filter((rating): rating is number => Number.isFinite(rating));
  if (ratings.length < 3) return null;
  return ratings.reduce((sum, rating) => sum + rating, 0) / ratings.length;
}

export async function POST(request: Request) {
  if (!publicPredictiveDraftsEnabled()) return unavailable();
  /* Development-only implementation retained below for reproducibility work. */
  /* istanbul ignore next */
  try {
    const body = (await request.json()) as Body;
    const context = await readCurrentDraftContext();
    if (!isSide(body.perspective) || !isSide(body.next_side)) {
      return NextResponse.json(
        { error: "perspective and next_side must be blue or red" },
        { status: 400 },
      );
    }
    if (body.candidate_role != null && !isCandidateRole(body.candidate_role)) {
      return NextResponse.json({ error: "candidate_role is invalid" }, { status: 400 });
    }
    const rawActions = body.actions ?? [];
    if (rawActions.length > 10) {
      return NextResponse.json({ error: "a draft can contain at most 10 picks" }, { status: 400 });
    }

    const actions: DraftAction[] = [];
    const seen = new Set<string>();
    for (const action of rawActions) {
      const side = isSide(action.side) ? action.side : null;
      if (!side || typeof action.champion !== "string" || !action.champion) {
        return NextResponse.json(
          { error: "every action needs a side and champion" },
          { status: 400 },
        );
      }
      if (action.role != null && !isRole(action.role)) {
        return NextResponse.json({ error: `invalid role for ${action.champion}` }, { status: 400 });
      }
      const champion = canonicalChampion(action.champion ?? "") ?? "";
      if (!champion) {
        return NextResponse.json({ error: `unknown champion: ${action.champion}` }, { status: 400 });
      }
      const key = champion.toLocaleLowerCase();
      if (seen.has(key)) {
        return NextResponse.json({ error: `${champion} is already selected` }, { status: 400 });
      }
      seen.add(key);
      actions.push({
        side: side as DraftSide,
        champion,
        role: (action.role as DraftRole | null | undefined) ?? null,
      });
    }

    const sideCounts = {
      blue: actions.filter((action) => action.side === "blue").length,
      red: actions.filter((action) => action.side === "red").length,
    };
    if (sideCounts.blue > 5 || sideCounts.red > 5) {
      return NextResponse.json({ error: "each side can select at most five champions" }, { status: 400 });
    }

    const excluded = (body.excluded || [])
      .map(canonicalChampion)
      .filter((champion): champion is string => Boolean(champion));
    const eloDiff = body.elo_diff == null ? null : Number(body.elo_diff);
    if (eloDiff != null && !Number.isFinite(eloDiff)) {
      return NextResponse.json({ error: "elo_diff must be numeric" }, { status: 400 });
    }
    const blueTeam = selectedTeam(context, body.blue_team);
    const redTeam = selectedTeam(context, body.red_team);
    if (body.blue_team && !blueTeam) {
      return NextResponse.json({ error: `unknown team: ${body.blue_team}` }, { status: 400 });
    }
    if (body.red_team && !redTeam) {
      return NextResponse.json({ error: `unknown team: ${body.red_team}` }, { status: 400 });
    }
    const blueLineup = lineup(context, body.blue_players, body.blue_team);
    const redLineup = lineup(context, body.red_players, body.red_team);
    const playerContext: DraftPlayerContext = {
      blue: blueLineup,
      red: redLineup,
    };
    const blueLineupRating = lineupRating(blueLineup);
    const redLineupRating = lineupRating(redLineup);
    const teamEloDiff =
      blueTeam && redTeam ? blueTeam.rating - redTeam.rating : eloDiff;
    const playerEloDiff =
      blueLineupRating != null && redLineupRating != null
        ? blueLineupRating - redLineupRating
        : null;

    return NextResponse.json(
      analyzeDraftSandbox({
        actions,
        perspective: body.perspective,
        next_side: body.next_side,
        candidate_role: body.candidate_role as DraftCandidateRole | undefined,
        excluded,
        league: body.league,
        team_elo_diff: teamEloDiff,
        player_elo_diff: playerEloDiff,
        player_context: playerContext,
        limit: body.limit,
      }),
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    );
  }
}
