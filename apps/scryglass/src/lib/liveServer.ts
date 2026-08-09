import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { LiveIndex, LivePointer, LiveSnapshot } from "@/lib/live";

function blobRoot(): string | null {
  return process.env.LIVE_BLOB_BASE_URL?.replace(/\/$/, "") || null;
}

async function readLocalJson<T>(relativePath: string): Promise<T | null> {
  const file = path.join(process.cwd(), "public", relativePath.replace(/^\//, ""));
  if (!existsSync(file)) return null;
  try {
    return JSON.parse(await readFile(file, "utf8")) as T;
  } catch {
    return null;
  }
}

async function readRemoteJson<T>(url: string): Promise<T | null> {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

export function liveIndexUrl(): string {
  const root = blobRoot();
  return root ? `${root}/live/index.json` : "/live/index.json";
}

export async function readLiveIndex(): Promise<LiveIndex | null> {
  const root = blobRoot();
  return root ? readRemoteJson<LiveIndex>(`${root}/live/index.json`) : readLocalJson<LiveIndex>("live/index.json");
}

export async function readLiveSnapshot(pointer: LivePointer): Promise<LiveSnapshot | null> {
  const root = blobRoot();
  const url = pointer.snapshot_url || (root ? `${root}/${pointer.snapshot_path}` : `/${pointer.snapshot_path}`);
  return root || url.startsWith("http") ? readRemoteJson<LiveSnapshot>(url) : readLocalJson<LiveSnapshot>(pointer.snapshot_path);
}

export async function readLiveSnapshots(index: LiveIndex | null): Promise<Record<string, LiveSnapshot>> {
  if (!index?.series?.length) return {};
  const pairs = await Promise.all(index.series.map(async (pointer) => [pointer.series_id, await readLiveSnapshot(pointer)] as const));
  return Object.fromEntries(pairs.filter((pair): pair is [string, LiveSnapshot] => Boolean(pair[1])));
}
