import { chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { leaderboardRows } from "@/lib/chatData";
import { readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

async function get(request: Request, signal: AbortSignal) {
  const params = searchParams(request);
  const category = clean(params.get("category")) || "rating";
  const role = clean(params.get("role")) || null;
  const tier = clean(params.get("tier")) || null;
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "10", 10) || 10, 1), 20);
  if (category === "teams_draft" || category === "players_draft") {
    const manifest = await readPackManifest(signal);
    return chatJson({
      schema_version: "scryglass:draft-api:v1",
      release_id: manifest.pack_id,
      authority: "unavailable",
      category,
      role,
      tier,
      limit: 0,
      rows: [],
    });
  }
  const rows = await leaderboardRows(category, role, limit, tier, signal);
  return chatJson({
    category,
    role,
    tier,
    limit: rows.length,
    rows,
  });
}

export const GET = secureChatRoute(get);
