import { NextResponse } from "next/server";
import {
  DRAFT_ROLES,
  analyzeDraftSandbox,
  draftCatalog,
  normalizeDraftChampion,
  type DraftAction,
  type DraftCandidateRole,
  type DraftRole,
  type DraftSide,
} from "@/lib/draftScore";

export const runtime = "nodejs";

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

export function GET() {
  return NextResponse.json({
    champions: draftCatalog(),
    roles: DRAFT_ROLES,
    model: "Draft Score v3 · partial-draft counterfactual",
  });
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Body;
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
      if (!isSide(action.side) || !action.champion) {
        return NextResponse.json(
          { error: "every action needs a side and champion" },
          { status: 400 },
        );
      }
      if (action.role != null && !isRole(action.role)) {
        return NextResponse.json({ error: `invalid role for ${action.champion}` }, { status: 400 });
      }
      const champion = canonicalChampion(action.champion);
      if (!champion) {
        return NextResponse.json({ error: `unknown champion: ${action.champion}` }, { status: 400 });
      }
      const key = champion.toLocaleLowerCase();
      if (seen.has(key)) {
        return NextResponse.json({ error: `${champion} is already selected` }, { status: 400 });
      }
      seen.add(key);
      actions.push({ side: action.side, champion, role: action.role ?? null });
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

    return NextResponse.json(
      analyzeDraftSandbox({
        actions,
        perspective: body.perspective,
        next_side: body.next_side,
        candidate_role: body.candidate_role as DraftCandidateRole | undefined,
        excluded,
        league: body.league,
        elo_diff: eloDiff,
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
