/** Public Riot patch labels used by the web application. */

const CLIENT_PATCH_RE = /^(15|16)\.(\d{1,2})(?:\.\d+)?$/;
const PUBLIC_PATCH_RE = /^(25|26)\.(\d{1,2})(?:\.\d+)?$/;

export function publicPatchLabel(value: string): string {
  const trimmed = value.trim();
  const clientMatch = CLIENT_PATCH_RE.exec(trimmed);
  if (clientMatch) {
    return `${Number(clientMatch[1]) + 10}.${clientMatch[2].padStart(2, "0")}`;
  }
  const publicMatch = PUBLIC_PATCH_RE.exec(trimmed);
  if (publicMatch) return `${publicMatch[1]}.${publicMatch[2].padStart(2, "0")}`;
  return trimmed;
}

export function samePublicPatch(left: string, right: string): boolean {
  return publicPatchLabel(left) === publicPatchLabel(right);
}

