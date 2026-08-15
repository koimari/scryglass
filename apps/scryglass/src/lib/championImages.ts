/**
 * Resolve the small champion portrait used by public match and profile views.
 *
 * The active query API may omit its image map while the data refresh is still
 * valid. CommunityDragon is the existing public champion-art source, so the
 * fallback stays on that explicit host and keeps the record useful.
 */

const CHAMPION_ALIASES: Record<string, string> = {
  "nunu & willump": "Nunu",
  "nunu and willump": "Nunu",
  "renata glasc": "Renata",
};

function championKey(name: string): string {
  const trimmed = name.trim();
  const alias = CHAMPION_ALIASES[trimmed.toLowerCase()];
  if (alias) return alias;
  return trimmed
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[’']/g, "")
    .replace(/[^a-zA-Z0-9]/g, "");
}

export function championImageUrl(
  name: string | null | undefined,
  supplied?: string | null,
): string | null {
  const direct = supplied?.trim();
  if (direct) return direct;
  if (!name?.trim()) return null;
  const key = championKey(name);
  if (!key) return null;
  return `https://cdn.communitydragon.org/latest/champion/${encodeURIComponent(key)}/square`;
}
