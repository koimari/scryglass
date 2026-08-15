import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import path from "node:path";
import test from "node:test";

import { readOnlyCachedJson } from "./identity-cache.mjs";

test("readOnlyCachedJson uses a supplied cache entry without fetching", async (t) => {
  const cacheDir = await fs.mkdtemp(path.join("/tmp", "scryglass-identity-cache-"));
  t.after(() => fs.rm(cacheDir, { recursive: true, force: true }));
  await fs.writeFile(path.join(cacheDir, "entry.json"), '{"source":"cache"}\n', "utf8");

  let fetches = 0;
  const value = await readOnlyCachedJson({
    cacheDir,
    fileName: "entry.json",
    url: "https://example.test/entry.json",
    fetchJson: async () => {
      fetches += 1;
      return { source: "network" };
    },
  });

  assert.deepEqual(value, { source: "cache" });
  assert.equal(fetches, 0);
});

test("readOnlyCachedJson does not write a fetched response into the cache", async (t) => {
  const cacheDir = await fs.mkdtemp(path.join("/tmp", "scryglass-identity-cache-"));
  t.after(() => fs.rm(cacheDir, { recursive: true, force: true }));

  const value = await readOnlyCachedJson({
    cacheDir,
    fileName: "network.json",
    url: "https://example.test/network.json",
    fetchJson: async () => ({ source: "network" }),
  });

  assert.deepEqual(value, { source: "network" });
  await assert.rejects(
    fs.access(path.join(cacheDir, "network.json")),
    { code: "ENOENT" },
  );
});
