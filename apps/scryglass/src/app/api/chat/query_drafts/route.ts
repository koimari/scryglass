import { chatError, chatJson, clean, searchParams, secureChatRoute } from "@/lib/chatApi";
import { readPackManifest } from "@/lib/serverPack";

export const runtime = "nodejs";

async function get(request: Request, signal: AbortSignal) {
  const question = clean(searchParams(request).get("q"));
  if (!question) return chatError("A team draft-score question is required.", 422);
  const manifest = await readPackManifest(signal);
  return chatJson({
    schema_version: "scryglass:draft-api:v1",
    release_id: manifest.pack_id,
    authority: "unavailable",
  });
}

export const GET = secureChatRoute(get);
