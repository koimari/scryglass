import assert from "node:assert/strict";
import test from "node:test";

import { handleDataPublished, type DataPublishedDependencies } from "../app/api/data-published/handler";
import type { PackManifest } from "./pack";

const RELEASE_ID = "v2026.08.15.160000";
const OTHER_RELEASE_ID = "v2026.08.15.160001";

function request(releaseId: string, ip: string): Request {
  return new Request("https://scryglass.test/api/data-published", {
    method: "POST",
    headers: {
      authorization: "Bearer route-test-secret",
      "content-type": "application/json",
      "x-forwarded-for": ip,
    },
    body: JSON.stringify({ release_id: releaseId }),
  });
}

function dependencies(
  readRemotePackManifest: DataPublishedDependencies["readRemotePackManifest"],
  events: string[],
): DataPublishedDependencies {
  return {
    readRemotePackManifest,
    revalidateTag: () => {
      events.push("tag");
    },
    revalidatePath: (path, type) => {
      events.push(`path:${type}:${path}`);
    },
  };
}

function manifest(releaseId: string): PackManifest {
  return { pack_id: releaseId } as PackManifest;
}

test("matching publication release invalidates the manifest and route targets", async () => {
  const previousSecret = process.env.SCRYGLASS_DATA_PUBLISH_TOKEN;
  process.env.SCRYGLASS_DATA_PUBLISH_TOKEN = "route-test-secret";
  const events: string[] = [];
  try {
    const response = await handleDataPublished(
      request(RELEASE_ID, "198.51.100.101"),
      dependencies(async () => manifest(RELEASE_ID), events),
    );
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      revalidated: true,
      requested_release_id: RELEASE_ID,
      served_release_id: RELEASE_ID,
      matches: true,
    });
    assert.equal(events[0], "tag");
    assert.equal(events.length, 1 + 26);
    assert.ok(events.includes("path:page:/elo"));
    assert.ok(events.includes("path:route:/api/assets/[...path]"));
  } finally {
    if (previousSecret === undefined) delete process.env.SCRYGLASS_DATA_PUBLISH_TOKEN;
    else process.env.SCRYGLASS_DATA_PUBLISH_TOKEN = previousSecret;
  }
});

test("mismatched publication release returns 409 without invalidation", async () => {
  const previousSecret = process.env.SCRYGLASS_DATA_PUBLISH_TOKEN;
  process.env.SCRYGLASS_DATA_PUBLISH_TOKEN = "route-test-secret";
  const events: string[] = [];
  try {
    const response = await handleDataPublished(
      request(RELEASE_ID, "198.51.100.102"),
      dependencies(async () => manifest(OTHER_RELEASE_ID), events),
    );
    assert.equal(response.status, 409);
    assert.deepEqual(await response.json(), {
      revalidated: false,
      requested_release_id: RELEASE_ID,
      served_release_id: OTHER_RELEASE_ID,
      matches: false,
    });
    assert.deepEqual(events, []);
  } finally {
    if (previousSecret === undefined) delete process.env.SCRYGLASS_DATA_PUBLISH_TOKEN;
    else process.env.SCRYGLASS_DATA_PUBLISH_TOKEN = previousSecret;
  }
});

test("manifest failure rejects without invalidation", async () => {
  const previousSecret = process.env.SCRYGLASS_DATA_PUBLISH_TOKEN;
  process.env.SCRYGLASS_DATA_PUBLISH_TOKEN = "route-test-secret";
  const events: string[] = [];
  try {
    await assert.rejects(
      handleDataPublished(
        request(RELEASE_ID, "198.51.100.103"),
        dependencies(async () => {
          throw new Error("manifest unavailable");
        }, events),
      ),
      /manifest unavailable/,
    );
    assert.deepEqual(events, []);
  } finally {
    if (previousSecret === undefined) delete process.env.SCRYGLASS_DATA_PUBLISH_TOKEN;
    else process.env.SCRYGLASS_DATA_PUBLISH_TOKEN = previousSecret;
  }
});
