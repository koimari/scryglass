import { promises as fs } from "node:fs";
import path from "node:path";
import { packUrl, type PackManifest } from "./pack";

const localManifestPath = path.join(process.cwd(), "public", "packs", "manifest.json");

async function readLocalManifest(): Promise<PackManifest> {
  return JSON.parse(await fs.readFile(localManifestPath, "utf8")) as PackManifest;
}

function liveManifestUrl(local: PackManifest): string | null {
  const configured = process.env.SCRYGLASS_PACK_MANIFEST_URL?.trim();
  if (configured) return configured;
  if (!local.base_url?.startsWith("http")) return null;
  const marker = "/packs/";
  const markerAt = local.base_url.indexOf(marker);
  if (markerAt < 0) return null;
  return `${local.base_url.slice(0, markerAt)}${marker}manifest.json`;
}

/** Read the mutable Blob pointer, with the deployed copy as a safe fallback. */
export async function readPackManifest(): Promise<PackManifest> {
  const local = await readLocalManifest();
  const remoteUrl = liveManifestUrl(local);
  if (!remoteUrl) return local;

  try {
    // Blob overwrites can take up to a minute to propagate. A minute-bucketed
    // query avoids a browser/CDN holding yesterday's pointer indefinitely.
    const bucket = Math.floor(Date.now() / 60_000);
    const response = await fetch(`${remoteUrl}?v=${bucket}`, { cache: "no-store" });
    if (response.ok) {
      const remote = (await response.json()) as PackManifest;
      if (remote.pack_id && remote.base_url) return remote;
    }
  } catch {
    // Keep the most recently deployed pointer available during a Blob outage.
  }
  return local;
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
