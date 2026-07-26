import { promises as fs } from "node:fs";
import path from "node:path";
import { packUrl, type PackManifest } from "./pack";

const localManifestPath = path.join(process.cwd(), "public", "packs", "manifest.json");

/** Read the small pointer that the refresh workflow updates on each publish. */
export async function readPackManifest(): Promise<PackManifest> {
  return JSON.parse(await fs.readFile(localManifestPath, "utf8")) as PackManifest;
}

/** Load pack JSON from Blob, with a local-copy fallback for development. */
export async function readPackJson<T>(manifest: PackManifest, relativePath: string): Promise<T> {
  if (manifest.base_url?.startsWith("http")) {
    try {
      const response = await fetch(packUrl(manifest, relativePath), { cache: "no-store" });
      if (response.ok) return (await response.json()) as T;
    } catch {
      // The local-only publication path remains useful when Blob is unavailable.
    }
  }

  const localPath = path.join(process.cwd(), "public", "packs", manifest.pack_id, relativePath);
  return JSON.parse(await fs.readFile(localPath, "utf8")) as T;
}
