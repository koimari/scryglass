import assert from "node:assert/strict";
import test from "node:test";

import { REVALIDATED_TARGETS } from "./revalidationTargets";

test("release invalidation uses route targets for API and asset handlers", () => {
  const targetByPath = new Map(REVALIDATED_TARGETS.map((target) => [target.path, target.type]));

  for (const path of [
    "/packs/manifest.json",
    "/rankings/tierlists.json",
    "/rankings/tierlists-latest.json",
    "/api/public-data/tierlists",
    "/api/chat/navigation",
    "/api/chat/query_players",
    "/api/assets/[...path]",
  ]) {
    assert.equal(targetByPath.get(path), "route", path);
  }
});

test("release invalidation keeps page targets for rendered routes", () => {
  const targetByPath = new Map(REVALIDATED_TARGETS.map((target) => [target.path, target.type]));

  for (const path of [
    "/",
    "/elo",
    "/matches",
    "/tiers",
    "/chat",
    "/elo/player/[player]",
    "/elo/team/[team]",
    "/matches/[game]",
  ]) {
    assert.equal(targetByPath.get(path), "page", path);
  }
});
