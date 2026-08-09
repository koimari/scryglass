import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";

/**
 * Regression for the `listen EPERM` validation failure (issue #43).
 *
 * The old script invoked the `tsx` CLI directly (`tsx --test ...`).  The tsx
 * CLI opens an IPC pipe in the operating-system temp directory to talk to its
 * worker; when that directory cannot host the pipe (EPERM), the runner died
 * before collecting a single test.  The deterministic form runs the Node test
 * runner with tsx loaded as an import hook only:
 *
 *   node --import tsx --test src/lib/*.test.mts
 *
 * That form never creates the tsx IPC listener, so the failure mode cannot
 * reappear through package.json.
 */
test("package test script avoids the tsx CLI IPC listener (listen EPERM regression)", () => {
  const raw = readFileSync(path.join(process.cwd(), "package.json"), "utf8");
  const pkg = JSON.parse(raw) as { scripts?: Record<string, string> };
  const script = pkg.scripts?.test ?? "";
  const tokens = script.split(/\s+/).filter(Boolean);

  assert.equal(
    tokens[0],
    "node",
    "test script must run through the Node binary, not the tsx CLI",
  );
  assert.ok(
    tokens.includes("--import") && tokens[tokens.indexOf("--import") + 1] === "tsx",
    "tsx must be loaded as a Node import hook, never invoked as a CLI",
  );
  assert.ok(
    tokens.includes("--test"),
    "test script must use the Node built-in test runner",
  );
  // The tsx CLI form (`tsx --test ...`) opens an IPC pipe in the temp
  // directory; assert no standalone `tsx` command token exists anywhere in
  // the script (including after `&&`/`||` chain operators).
  const commandPositions = [0, ...script
    .split(/(?:&&|\|\|)/)
    .map((part) => part.split(/\s+/).filter(Boolean).slice(0, 1)[0])
    .filter(Boolean)
    .map((first) => first.replace(/^npx\s+/, ""))];
  for (const command of commandPositions) {
    assert.notEqual(
      command,
      "tsx",
      `test script must not invoke the tsx CLI (${command}); its IPC pipe fails with listen EPERM in restricted temp directories`,
    );
  }
  assert.ok(
    tokens.some((token) => token.includes(".test.mts")),
    "test script must target the TypeScript test suite",
  );
});
