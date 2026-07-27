import { NextResponse } from "next/server";
import { draftScore } from "@/lib/draftScore";

export const runtime = "nodejs";

type Body = {
  blue?: string[];
  red?: string[];
  league?: string | null;
  patch?: string | null;
  elo_diff?: number | null;
  team_elo_diff?: number | null;
  player_elo_diff?: number | null;
  blue_team?: string | null;
  red_team?: string | null;
  blue_players?: string[] | null;
  red_players?: string[] | null;
  strength_source?: string | null;
  blue_roles?: string[] | null;
  red_roles?: string[] | null;
};

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as Body;
    if (!body.blue?.length || !body.red?.length) {
      return NextResponse.json({ error: "blue and red picks required" }, { status: 400 });
    }
    if (body.blue.length !== 5 || body.red.length !== 5) {
      return NextResponse.json({ error: "need 5 picks per side" }, { status: 400 });
    }
    const result = draftScore({
      blue: body.blue,
      red: body.red,
      league: body.league,
      patch: body.patch,
      elo_diff: body.elo_diff,
      team_elo_diff: body.team_elo_diff,
      player_elo_diff: body.player_elo_diff,
      blue_team: body.blue_team,
      red_team: body.red_team,
      blue_players: body.blue_players,
      red_players: body.red_players,
      strength_source: body.strength_source,
      blue_roles: body.blue_roles,
      red_roles: body.red_roles,
    });
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 500 },
    );
  }
}
