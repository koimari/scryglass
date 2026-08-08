import { NextResponse } from "next/server";
import {
  filterTierlist,
  loadTierlistView,
  TIERLIST_UNAVAILABLE,
  TierlistQueryError,
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

/** Production tier-list surface with strict scope filters. */
export async function GET(req: Request) {
  try {
    const view = await loadTierlistView();
    if (!view) {
      return NextResponse.json(TIERLIST_UNAVAILABLE, { status: 503 });
    }
    const url = new URL(req.url);
    const query: TierlistQuery = {
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
      api_version: view.api_version,
      generated_at: view.generated_at,
      as_of: view.as_of,
      development_only: view.development_only,
      publication_eligible: view.publication_eligible,
      cells_available: view.cells_available,
      cells_total: view.cells_total,
      query,
      options: view.options,
      scopes: view.scopes,
      provenance: view.provenance,
      rows,
    }, {
      headers: {
        "Cache-Control": "no-store, max-age=0",
      },
    });
  } catch (error) {
    if (error instanceof TierlistQueryError) {
      return NextResponse.json(
        { api_version: "tierlist-v2", status: "invalid_request", code: error.code, reason: error.message },
        { status: 400 },
      );
    }
    return NextResponse.json(TIERLIST_UNAVAILABLE, { status: 503 });
  }
}
