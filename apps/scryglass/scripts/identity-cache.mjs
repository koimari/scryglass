import { promises as fs } from "node:fs";
import path from "node:path";

/**
 * Read an operator-provided cache entry, then fall back to the supplied fetch.
 *
 * The cache is deliberately read-only.  A response from the network must not
 * be copied into a path selected through SCRYGLASS_IDENTITY_CACHE.
 */
export async function readOnlyCachedJson({ cacheDir, fileName, url, fetchJson }) {
  if (cacheDir) {
    try {
      return JSON.parse(await fs.readFile(path.join(cacheDir, fileName), "utf8"));
    } catch {
      // Fetch below when the optional local cache does not contain this page.
    }
  }
  return fetchJson(url);
}
