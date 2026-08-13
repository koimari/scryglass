import { chatError, chatJson, clean, readChatJson, searchParams } from "@/lib/chatApi";

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

export async function GET(request: Request) {
  const params = searchParams(request);
  const role = clean(params.get("role"));
  const patch = clean(params.get("patch"));
  const limit = Math.min(Math.max(parseInt(params.get("limit") ?? "20", 10) || 20, 1), 100);
  try {
    const tier = await readChatJson<{ options: { patches: string[] }; rows: TierRow[] }>(
      "rankings/tierlists.json",
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
