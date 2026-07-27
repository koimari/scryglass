export type DraftRow = Record<string, unknown>;

export const DRAFT_ROLES = ["top", "jng", "mid", "bot", "sup"] as const;

export type DraftLineup = {
  blue: string[];
  red: string[];
  blueRoles: string[] | null;
  redRoles: string[] | null;
  source: "participants" | "map-picks";
};

function championName(value: unknown): string | null {
  if (value == null) return null;
  const name = String(value).trim();
  if (!name || ["unknown", "nan", "none", "null"].includes(name.toLowerCase())) {
    return null;
  }
  return name;
}

function normalizeRole(value: unknown): (typeof DRAFT_ROLES)[number] | null {
  const role = String(value ?? "").trim().toLowerCase();
  if (role === "top" || role.startsWith("top")) return "top";
  if (role === "jng" || role === "jungle" || role.startsWith("jungler")) return "jng";
  if (role === "mid" || role.startsWith("middle")) return "mid";
  if (role === "bot" || role === "adc" || role.startsWith("bottom")) return "bot";
  if (role === "sup" || role === "utility" || role.startsWith("support")) return "sup";
  return null;
}

function participantLineup(
  players: DraftRow[],
  side: "Blue" | "Red",
): string[] | null {
  const byRole = new Map<(typeof DRAFT_ROLES)[number], string>();
  for (const player of players) {
    if (String(player.side ?? "").trim().toLowerCase() !== side.toLowerCase()) continue;
    const role = normalizeRole(player.position ?? player.role);
    const champion = championName(player.champion);
    if (!role || !champion || byRole.has(role)) return null;
    byRole.set(role, champion);
  }
  const lineup = DRAFT_ROLES.map((role) => byRole.get(role) ?? "");
  if (lineup.some((champion) => !champion) || new Set(lineup).size !== 5) return null;
  return lineup;
}

function mapPicks(map: DraftRow, side: "blue" | "red"): string[] | null {
  const picks = [1, 2, 3, 4, 5].map((index) =>
    championName(map[`${side}_pick${index}`]),
  );
  if (picks.some((pick) => pick == null)) return null;
  const complete = picks as string[];
  return new Set(complete).size === 5 ? complete : null;
}

function hasTenUniqueChampions(blue: string[], red: string[]): boolean {
  const normalized = [...blue, ...red].map((champion) =>
    champion.normalize("NFKC").trim().toLocaleLowerCase(),
  );
  return new Set(normalized).size === 10;
}

/**
 * Final game participants are the authoritative champion composition and also
 * preserve role alignment. Map pick columns are draft-order, so they are only
 * a composition fallback and must not be paired with role-order matchup terms.
 */
export function resolveDraftLineup(
  map: DraftRow,
  players: DraftRow[],
): DraftLineup | null {
  const blueParticipants = participantLineup(players, "Blue");
  const redParticipants = participantLineup(players, "Red");
  if (
    blueParticipants &&
    redParticipants &&
    hasTenUniqueChampions(blueParticipants, redParticipants)
  ) {
    const roles = [...DRAFT_ROLES];
    return {
      blue: blueParticipants,
      red: redParticipants,
      blueRoles: roles,
      redRoles: [...roles],
      source: "participants",
    };
  }

  const bluePicks = mapPicks(map, "blue");
  const redPicks = mapPicks(map, "red");
  if (
    !bluePicks ||
    !redPicks ||
    !hasTenUniqueChampions(bluePicks, redPicks)
  ) {
    return null;
  }
  return {
    blue: bluePicks,
    red: redPicks,
    blueRoles: null,
    redRoles: null,
    source: "map-picks",
  };
}
