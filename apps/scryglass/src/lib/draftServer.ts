import { draftContext, type DraftContext } from "./draftScore";
import { readPackJson, readPackManifest } from "./serverPack";

let cachedPack = "";
let cachedContext: DraftContext | null = null;

export async function readCurrentDraftContext(): Promise<DraftContext> {
  const manifest = await readPackManifest();
  if (cachedContext && cachedPack === manifest.pack_id) return cachedContext;
  try {
    cachedContext = await readPackJson<DraftContext>(
      manifest,
      "features/draft_context.json",
    );
    cachedPack = manifest.pack_id;
    return cachedContext;
  } catch {
    cachedContext = draftContext();
    cachedPack = manifest.pack_id;
    return cachedContext;
  }
}
