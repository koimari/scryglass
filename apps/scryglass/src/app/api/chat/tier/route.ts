import { chatError, chatJson, clean, readChatJson, searchParams, secureChatRoute } from "@/lib/chatApi";
import { getTierRows, queryApiAvailable } from "@/lib/publicData";
import { readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

type TierRow = {
  champion: string;
  role: string;
  patch: string;
  rank: number;
  tier_bucket: string;
  played_maps: number;
  movement?: string | null;
};

async function get(request: Request, signal: AbortSignal) {
  const params = searchParams(request);
  const role = clean(params.get("role"));
  const patch = clean(params.get("patch"));
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "20", 10) || 20, 1), 20);
  try {
    const manifest = await readPackManifest(signal);
    if (queryApiAvailable(manifest)) {
      const result = await getTierRows(manifest, {
        kind: "champion",
        patches: patch ? [patch] : [],
        roles: role ? [role.toLowerCase()] : [],
        order: "rank_asc",
        limit,
        offset: 0,
      }, signal);
      const rows: TierRow[] = result.rows.map((row) => ({
        champion: row.name,
        role: row.role ?? "",
        patch: row.patch ?? "",
        rank: row.rank,
        tier_bucket: typeof row.payload.tier_bucket === "string" ? row.payload.tier_bucket : "",
        played_maps: row.played_maps,
        movement: typeof row.payload.movement === "string" ? row.payload.movement : null,
      }));
      return chatJson({ patch: rows[0]?.patch ?? patch, role: role || "all", rows });
    }
    const tier = await readChatJson<{ options: { patches: string[] }; rows: TierRow[] }>(
      "rankings/tierlists.json",
      signal,
    );
    const patches = tier.options?.patches ?? [];
    const latestPatch = patch || (patches.length ? patches[patches.length - 1] : "");
    let rows = tier.rows.filter((row) => row.patch === latestPatch);
    if (role) {
      const lower = role.toLowerCase();
      rows = rows.filter((row) => row.role.toLowerCase() === lower);
    }
    rows = [...rows].sort((a, b) => a.rank - b.rank);
    return chatJson({ patch: latestPatch, role: role || "all", rows: rows.slice(0, limit) });
  } catch {
    return chatError("Tier lists are unavailable for the current release.");
  }
}

export const GET = secureChatRoute(get);
