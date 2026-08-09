import { promises as fs } from "node:fs";
import path from "node:path";
import type { PackManifest } from "./pack";

const packJsonCache = new Map<string, Promise<unknown>>();

function localPackRoot(): string {
  const configured = process.env.SCRYGLASS_PACK_ROOT?.trim();
  return configured ? path.resolve(configured) : path.join(process.cwd(), "public", "packs");
}

/** Read the validated pack pointer from the local publication root. */
export async function readPackManifest(): Promise<PackManifest> {
  return JSON.parse(
    await fs.readFile(path.join(localPackRoot(), "manifest.json"), "utf8"),
  ) as PackManifest;
}

/** Load validated pack JSON from the same local publication root. */
export async function readPackJson<T>(manifest: PackManifest, relativePath: string): Promise<T> {
  const localPath = path.join(localPackRoot(), manifest.pack_id, relativePath);
  let pending = packJsonCache.get(localPath);
  if (!pending) {
    pending = fs.readFile(localPath, "utf8").then((raw) => JSON.parse(raw) as unknown);
    packJsonCache.set(localPath, pending);
    pending.catch(() => packJsonCache.delete(localPath));
  }
  return pending as Promise<T>;
}
