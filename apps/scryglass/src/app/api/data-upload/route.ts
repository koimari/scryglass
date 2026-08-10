import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { NextResponse } from "next/server";
import { uploadPolicy, validPublishSecret } from "@/lib/dataPublish";

export const runtime = "nodejs";

export async function POST(request: Request) {
  if (!validPublishSecret(request.headers.get("authorization"), process.env.SCRYGLASS_DATA_PUBLISH_TOKEN)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  try {
    const body = (await request.json()) as HandleUploadBody;
    const result = await handleUpload({
      request,
      body,
      onBeforeGenerateToken: async (pathname) => {
        const policy = uploadPolicy(pathname);
        if (!policy) throw new Error("upload path is outside the public data contract");
        return {
          allowedContentTypes: ["application/json"],
          addRandomSuffix: false,
          validUntil: Date.now() + 10 * 60 * 1000,
          ...policy,
        };
      },
    });
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "upload request failed";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
