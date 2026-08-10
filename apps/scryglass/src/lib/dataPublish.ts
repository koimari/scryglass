const PACK_ID = /^v\d{4}\.\d{2}\.\d{2}\.\d{6}$/;
const PACK_FILE = /^(manifest\.json|features\/[a-z0-9_]+\.json)$/;

export type UploadPolicy = {
  allowOverwrite: boolean;
  cacheControlMaxAge: number;
  maximumSizeInBytes: number;
};

/** Keep maintenance uploads inside the small public data surface. */
export function uploadPolicy(pathname: string): UploadPolicy | null {
  if (pathname === "packs/manifest.json" || pathname === "packs/latest.json") {
    return { allowOverwrite: true, cacheControlMaxAge: 60, maximumSizeInBytes: 2_000_000 };
  }
  if (pathname === "rankings/tierlists.json") {
    return { allowOverwrite: true, cacheControlMaxAge: 60, maximumSizeInBytes: 20_000_000 };
  }
  const parts = pathname.split("/");
  if (parts.length >= 3 && parts[0] === "packs" && PACK_ID.test(parts[1])) {
    const relative = parts.slice(2).join("/");
    if (PACK_FILE.test(relative)) {
      return { allowOverwrite: false, cacheControlMaxAge: 31_536_000, maximumSizeInBytes: 25_000_000 };
    }
  }
  return null;
}

export function validPublishSecret(received: string | null, expected: string | undefined): boolean {
  return Boolean(expected && received === `Bearer ${expected}`);
}
