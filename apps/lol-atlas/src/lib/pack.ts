/** Pack manifest helpers (client + server). */

export type PackFile = {
  path: string;
  relative?: string;
  rows: number | null;
  cols: number | null;
  bytes: number;
  sha256: string;
  columns?: string[] | null;
};

export type PackManifest = {
  pack_id: string;
  schema_version: string;
  created_utc: string;
  filters: {
    years: number[];
    leagues: string;
    leagues_note?: string;
  };
  attribution: string;
  excluded: string[];
  base_url: string | null;
  total_bytes: number;
  total_files: number;
  files: PackFile[];
};

export type TeamRating = {
  team: string;
  mu_total: number;
  mu_regional: number;
  mu_meta: number;
  sigma: number;
};

export type PlayerRating = {
  player: string;
  mu_total: number;
  mu_regional: number;
  mu_meta: number;
  sigma: number;
  n_maps: number;
  last_team: string | null;
};

export type EloCalibration = {
  team: { intercept: number; coef: number; temperature_400?: number };
  player: { intercept: number; coef: number; temperature_400?: number };
};

export async function loadManifest(origin = ""): Promise<PackManifest> {
  const res = await fetch(`${origin}/packs/manifest.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`manifest ${res.status}`);
  return res.json();
}

export function packUrl(manifest: PackManifest, relativePath: string): string {
  const base = (manifest.base_url || `/packs/${manifest.pack_id}`).replace(/\/$/, "");
  return `${base}/${relativePath.replace(/^\//, "")}`;
}

/** logit = a + b*(mu_diff/400); p = sigmoid(logit) for favorite when mu_diff>0 vs even foe */
export function eloToWinProb(
  mu: number,
  foeMu: number,
  cal: { intercept: number; coef: number },
): number {
  const muDiff = mu - foeMu;
  const logit = cal.intercept + cal.coef * (muDiff / 400);
  return 1 / (1 + Math.exp(-logit));
}

export function formatMb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function teamSlug(name: string): string {
  return encodeURIComponent(name.trim());
}

/** Soft ranking: penalize high-σ teams so thin regional ladders don't outrank majors. */
export function softMu(mu: number, sigma: number, floor = 25): number {
  return mu - Math.max(0, sigma - floor);
}
