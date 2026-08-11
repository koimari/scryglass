import visualIdentities from "@/data/playerVisualIdentities.json";

export type PlayerVisualIdentity = {
  src: string;
  source: string;
  file: string;
};

const PLAYER_PORTRAITS = visualIdentities.identities as Record<string, PlayerVisualIdentity>;

function normalized(value: string | null | undefined): string {
  return String(value ?? "")
    .trim()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replaceAll("ø", "o")
    .replaceAll("Ø", "O")
    .replaceAll("ł", "l")
    .replaceAll("Ł", "L")
    .replaceAll("đ", "d")
    .replaceAll("Đ", "D")
    .replaceAll("ß", "ss")
    .toLowerCase()
    .replaceAll("&", " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export function playerPortrait(
  player: string | null | undefined,
  team: string | null | undefined,
): PlayerVisualIdentity | null {
  if (!player) return null;
  return PLAYER_PORTRAITS[`${normalized(player)}|${normalized(team)}`] ?? null;
}
