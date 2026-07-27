import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { VOID_GRUBS_PUBLICATION } from "./articles";
import { packUrl, type PackManifest } from "./pack";
import {
  validateArticlePublication,
  type ArticlePublicationValidation,
} from "./pstar";
import type {
  PackServerOptions,
  ResolvedPackManifest,
} from "./serverPack";

type ValidPublication = Extract<ArticlePublicationValidation, { ok: true }>;

export type LoadedGrubsArticlePublication = ValidPublication & {
  packId: string;
  href: string;
  sha256: string;
};

function declaredArticle(
  manifest: PackManifest,
): { sha256: string } | null {
  const file = manifest.files.find(
    (entry) =>
      (entry.relative ?? entry.path) === VOID_GRUBS_PUBLICATION.articlePath,
  );
  if (!file?.sha256 || !/^[a-f0-9]{64}$/i.test(file.sha256)) return null;
  return { sha256: file.sha256.toLowerCase() };
}

async function readArticleBytes(
  manifest: PackManifest,
  options: Pick<PackServerOptions, "packRoot" | "fetchImpl">,
): Promise<Uint8Array | null> {
  const resolved = manifest as ResolvedPackManifest;
  if (resolved.source === "remote" && manifest.base_url?.startsWith("http")) {
    try {
      const response = await (options.fetchImpl ?? fetch)(
        packUrl(manifest, VOID_GRUBS_PUBLICATION.articlePath),
        { cache: "no-store" },
      );
      return response.ok ? new Uint8Array(await response.arrayBuffer()) : null;
    } catch {
      return null;
    }
  }

  const packRoot =
    options.packRoot ??
    path.join(process.cwd(), "public", "packs");
  try {
    return await fs.readFile(
      path.join(
        /* turbopackIgnore: true */ packRoot,
        manifest.pack_id,
        ...VOID_GRUBS_PUBLICATION.articlePath.split("/"),
      ),
    );
  } catch {
    return null;
  }
}

/**
 * Load the article from one immutable pack and fail closed on any missing file,
 * manifest hash mismatch, schema drift, stale mechanics, or formula mismatch.
 */
export async function readValidatedGrubsArticlePublication(
  manifest: PackManifest,
  options: Pick<PackServerOptions, "packRoot" | "fetchImpl"> = {},
): Promise<LoadedGrubsArticlePublication | null> {
  const declared = declaredArticle(manifest);
  if (!declared) return null;

  const bytes = await readArticleBytes(manifest, options);
  if (!bytes) return null;
  const actualSha = createHash("sha256").update(bytes).digest("hex");
  if (actualSha !== declared.sha256) return null;

  let raw: unknown;
  try {
    raw = JSON.parse(Buffer.from(bytes).toString("utf8"));
  } catch {
    return null;
  }
  const validation = validateArticlePublication(
    raw,
    VOID_GRUBS_PUBLICATION,
  );
  if (!validation.ok) return null;

  return {
    ...validation,
    packId: manifest.pack_id,
    href: packUrl(manifest, VOID_GRUBS_PUBLICATION.articlePath),
    sha256: actualSha,
  };
}
