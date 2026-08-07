import { NextResponse } from "next/server";
import {
  filterTierlist,
  loadTierlistView,
  TIERLIST_UNAVAILABLE,
  type TierlistQuery,
} from "@/lib/tierlistServer";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function str(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function num(value: string | string[] | undefined): number | undefined {
  const text = str(value);
  if (!text) return undefined;
  const parsed = Number.parseInt(text, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/** Development-only tier-list surface with scope filters. */
export async function GET(req: Request) {
  try {
    const view = loadTierlistView();
    if (!view) {
      return NextResponse.json(TIERLIST_UNAVAILABLE, { status: 503 });
    }
    const url = new URL(req.url);
    const query: TierlistQuery = {
      region: str(url.searchParams.get("region") ?? undefined),
      league: str(url.searchParams.get("league") ?? undefined),
      international: str(url.searchParams.get("international") ?? undefined),
      competition_tier: str(url.searchParams.get("competition_tier") ?? undefined),
      role: str(url.searchParams.get("role") ?? undefined),
      patch: str(url.searchParams.get("patch") ?? undefined),
      played_maps_min: num(url.searchParams.get("played_maps_min") ?? undefined),
    };
    const rows = filterTierlist(view, query);
    return NextResponse.json({
      status: "available",
      generated_at: view.generated_at,
      development_only: view.development_only,
      cells_available: view.cells_available,
      cells_total: view.cells_total,
      query,
      options: view.options,
      rows,
    });
  } catch {
    return NextResponse.json(TIERLIST_UNAVAILABLE, { status: 503 });
  }
}
