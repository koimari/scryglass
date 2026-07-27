import assert from "node:assert/strict";
import test from "node:test";
import { resolveDraftLineup } from "./draftLineup.ts";

const roles = ["top", "jng", "mid", "bot", "sup"];

function participants(side: "Blue" | "Red", champions: string[]) {
  return roles.map((position, index) => ({
    side,
    position,
    playername: `${side}-${position}`,
    champion: champions[index],
  }));
}

test("uses verified participant champions when map pick columns are empty", () => {
  const blue = ["Olaf", "Nasus", "Syndra", "Caitlyn", "Bard"];
  const red = ["K'Sante", "Trundle", "Cassiopeia", "Ashe", "Seraphine"];
  const result = resolveDraftLineup(
    {},
    [...participants("Blue", blue), ...participants("Red", red)],
  );

  assert.deepEqual(result, {
    blue,
    red,
    blueRoles: roles,
    redRoles: roles,
    source: "participants",
  });
});

test("prefers role-aligned participants over draft-order map picks", () => {
  const map = {
    blue_pick1: "Syndra",
    blue_pick2: "Bard",
    blue_pick3: "Caitlyn",
    blue_pick4: "Nasus",
    blue_pick5: "Olaf",
    red_pick1: "Ashe",
    red_pick2: "Seraphine",
    red_pick3: "Cassiopeia",
    red_pick4: "Trundle",
    red_pick5: "K'Sante",
  };
  const blue = ["Olaf", "Nasus", "Syndra", "Caitlyn", "Bard"];
  const red = ["K'Sante", "Trundle", "Cassiopeia", "Ashe", "Seraphine"];
  const result = resolveDraftLineup(
    map,
    [...participants("Blue", blue), ...participants("Red", red)],
  );

  assert.equal(result?.source, "participants");
  assert.deepEqual(result?.blue, blue);
  assert.deepEqual(result?.red, red);
});

test("uses complete map picks without claiming role alignment", () => {
  const map = Object.fromEntries(
    ["blue", "red"].flatMap((side) =>
      roles.map((role, index) => [`${side}_pick${index + 1}`, `${side}-${role}`]),
    ),
  );
  const result = resolveDraftLineup(map, []);

  assert.equal(result?.source, "map-picks");
  assert.equal(result?.blue.length, 5);
  assert.equal(result?.red.length, 5);
  assert.equal(result?.blueRoles, null);
  assert.equal(result?.redRoles, null);
});

test("fails closed when neither source has five champions per side", () => {
  const players = [
    ...participants("Blue", ["A", "B", "C", "D", "E"]),
    ...participants("Red", ["F", "G", "H", "I", ""]),
  ];
  assert.equal(resolveDraftLineup({}, players), null);
});

test("fails closed when a champion appears on both sides", () => {
  const blue = ["Olaf", "Nasus", "Syndra", "Caitlyn", "Bard"];
  const red = ["Olaf", "Trundle", "Cassiopeia", "Ashe", "Seraphine"];
  assert.equal(
    resolveDraftLineup(
      {},
      [...participants("Blue", blue), ...participants("Red", red)],
    ),
    null,
  );
});
