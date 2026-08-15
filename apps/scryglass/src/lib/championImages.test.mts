import assert from "node:assert/strict";
import test from "node:test";
import { championImageUrl } from "./championImages";

test("uses a published champion image when one is available", () => {
  assert.equal(
    championImageUrl("Ahri", "https://cdn.communitydragon.org/latest/champion/103/square"),
    "https://cdn.communitydragon.org/latest/champion/103/square",
  );
});

test("normalizes champion display names for the public fallback", () => {
  assert.equal(
    championImageUrl("Jarvan IV"),
    "https://cdn.communitydragon.org/latest/champion/JarvanIV/square",
  );
  assert.equal(
    championImageUrl("K'Sante"),
    "https://cdn.communitydragon.org/latest/champion/KSante/square",
  );
});

test("keeps missing champion names unavailable", () => {
  assert.equal(championImageUrl(null), null);
  assert.equal(championImageUrl("   "), null);
});

