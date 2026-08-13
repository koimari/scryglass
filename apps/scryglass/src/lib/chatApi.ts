/** Shared helpers for the /api/chat/* read-only endpoints. */

import { NextResponse } from "next/server";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const CHAT_CACHE_HEADERS = {
  "Cache-Control": "public, max-age=0, s-maxage=21600, stale-while-revalidate=3600",
} as const;

export function chatJson(data: unknown): NextResponse {
  return NextResponse.json({ ok: true, data }, { headers: CHAT_CACHE_HEADERS });
}

export function chatError(reason: string, status = 404): NextResponse {
  return NextResponse.json(
    { ok: false, reason },
    { status, headers: CHAT_CACHE_HEADERS },
  );
}

export async function readChatJson<T>(relativePath: string): Promise<T> {
  const manifest = await readPackManifest();
  return readPackJson<T>(manifest, relativePath);
}

export function searchParams(request: Request): URLSearchParams {
  return new URL(request.url).searchParams;
}

export function clean(value: string | null): string {
  return (value ?? "").trim();
}
