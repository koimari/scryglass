import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(
    { status: "retired", reason: "The legacy tier-list artifact is retired. Use /tiers." },
    {
      status: 410,
      headers: { "Cache-Control": "private, no-store", Link: "</tiers>; rel=\"alternate\"" },
    },
  );
}
