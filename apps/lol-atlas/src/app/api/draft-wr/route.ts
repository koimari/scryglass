import { NextResponse } from "next/server";
import { draftScore } from "@/lib/draftScore";

export const runtime = "nodejs";

type Body = {
  blue?: string[];
  red?: string[];
  league?: string | null;
  elo_diff?: number | null;
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
      elo_diff: body.elo_diff,
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
