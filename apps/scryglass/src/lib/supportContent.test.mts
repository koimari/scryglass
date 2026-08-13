import assert from "node:assert/strict";
import test from "node:test";
import { METHODOLOGY_SECTIONS, NAVIGATION_HELP, matchTopic } from "./supportContent";

test("methodology sections cover every topic", () => {
  for (const topic of ["ratings", "grades", "tiers", "draft", "matches", "schedule", "elo", "all"]) {
    assert.ok(METHODOLOGY_SECTIONS[topic as keyof typeof METHODOLOGY_SECTIONS], topic);
  }
});

test("matchTopic resolves free text to a section", () => {
  assert.equal(matchTopic("how does the draft win share work"), "draft");
  assert.equal(matchTopic("what do the grades mean"), "grades");
  assert.equal(matchTopic("tell me about tier lists"), "tiers");
  assert.equal(matchTopic("when does the next game happen"), "schedule");
  assert.equal(matchTopic("how is the elo rating computed"), "ratings");
  assert.equal(matchTopic("no signal here"), null);
});

test("navigation help lists the site pages with paths", () => {
  assert.ok(NAVIGATION_HELP.length >= 5);
  const paths = NAVIGATION_HELP.map((entry) => entry.path);
  assert.ok(paths.includes("/elo"));
  assert.ok(paths.includes("/matches"));
  assert.ok(paths.includes("/tiers"));
  assert.ok(paths.includes("/methodology"));
  assert.ok(paths.includes("/chat"));
});
