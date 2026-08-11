import visualIdentities from "@/data/teamVisualIdentities.json";

type TeamVisualIdentity = {
  src: string;
  source: string;
  file: string;
};

const TEAM_MARKS = visualIdentities.identities as Record<string, TeamVisualIdentity>;

function normalizedTeam(team: string): string {
  return team
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

export function teamMarkUrl(team: string | null | undefined): string | null {
  if (!team) return null;
  return TEAM_MARKS[normalizedTeam(team)]?.src ?? null;
}

export function teamMarkSource(team: string | null | undefined): string | null {
  if (!team) return null;
  return TEAM_MARKS[normalizedTeam(team)]?.source ?? null;
}

export function teamInitials(team: string | null | undefined): string {
  if (!team) return "?";
  const words = team
    .replace(/\s+\([^)]*\)$/, "")
    .split(/\s+/)
    .filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return words.slice(0, 2).map((word) => word[0]).join("").toUpperCase();
}
