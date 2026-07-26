import { readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

export async function GET() {
  const manifest = await readPackManifest();
  return Response.json(manifest, {
    headers: {
      "Cache-Control": "no-store, max-age=0",
    },
  });
}
