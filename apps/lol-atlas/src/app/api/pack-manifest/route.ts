import {
  PackServiceError,
  readPackManifest,
  type ResolvedPackManifest,
} from "@/lib/serverPack";

export const dynamic = "force-dynamic";

const NO_STORE_HEADERS = {
  "Cache-Control": "no-store, max-age=0",
} as const;

type ManifestReader = () => Promise<ResolvedPackManifest>;

export async function packManifestResponse(
  reader: ManifestReader = readPackManifest,
): Promise<Response> {
  try {
    const manifest = await reader();
    return Response.json(manifest, { headers: NO_STORE_HEADERS });
  } catch (error) {
    const code =
      error instanceof PackServiceError
        ? error.code
        : "PACK_MANIFEST_UNAVAILABLE";
    return Response.json(
      {
        ok: false,
        error: {
          code,
          message: "Pack manifest is temporarily unavailable.",
        },
      },
      {
        status: 503,
        headers: NO_STORE_HEADERS,
      },
    );
  }
}

export async function GET() {
  return packManifestResponse();
}
