import { DraftSandbox } from "@/components/DraftSandbox";
import {
  DRAFT_ROLES,
  draftCatalog,
  type DraftAction,
  type DraftRole,
  type DraftSide,
} from "@/lib/draftScore";

export const dynamic = "force-dynamic";

const LEAGUES = new Set(["LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP", "INTL"]);

function parseScenario(
  raw: string | undefined,
  catalogNames: Set<string>,
): { actions: DraftAction[]; excluded: string[] } | undefined {
  if (!raw) return undefined;
  try {
    const decoded = Buffer.from(raw, "base64url").toString("utf8");
    const payload = JSON.parse(decoded) as
      | Array<Record<string, unknown>>
      | { actions?: Array<Record<string, unknown>>; excluded?: unknown[] };
    const parsed = Array.isArray(payload) ? payload : payload.actions;
    if (!Array.isArray(parsed) || parsed.length > 10) return undefined;
    const actions: DraftAction[] = [];
    const seen = new Set<string>();
    for (const item of parsed) {
      const side = item.side;
      const champion = item.champion;
      const role = item.role;
      if ((side !== "blue" && side !== "red") || typeof champion !== "string") return undefined;
      if (!catalogNames.has(champion) || seen.has(champion)) return undefined;
      if (role != null && !DRAFT_ROLES.includes(role as DraftRole)) return undefined;
      seen.add(champion);
      actions.push({
        side: side as DraftSide,
        champion,
        role: (role as DraftRole | null | undefined) ?? null,
      });
    }
    const rawExcluded = Array.isArray(payload) ? [] : payload.excluded ?? [];
    const excluded = rawExcluded.filter(
      (champion): champion is string =>
        typeof champion === "string" &&
        catalogNames.has(champion) &&
        !seen.has(champion),
    );
    return { actions, excluded: [...new Set(excluded)] };
  } catch {
    return undefined;
  }
}

export default async function SandboxPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const catalog = draftCatalog();
  const rawDraft = typeof params.draft === "string" ? params.draft : undefined;
  const scenario = parseScenario(rawDraft, new Set(catalog.map((champion) => champion.name)));
  const perspective =
    params.side === "blue" || params.side === "red" ? params.side : undefined;
  const league =
    typeof params.league === "string" && LEAGUES.has(params.league)
      ? params.league
      : undefined;

  return (
    <DraftSandbox
      catalog={catalog}
      initialActions={scenario?.actions}
      initialExcluded={scenario?.excluded}
      initialPerspective={perspective}
      initialLeague={league}
    />
  );
}
