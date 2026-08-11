import assert from "node:assert/strict";
import test from "node:test";
import { sameTimestamp } from "./health.ts";

test("health timestamps compare the same instant across UTC formats", () => {
  assert.equal(
    sameTimestamp("2026-08-11T10:50:41Z", "2026-08-11T10:50:41+00:00"),
    true,
  );
  assert.equal(
    sameTimestamp("2026-08-11T10:50:41Z", "2026-08-11T10:50:42+00:00"),
    false,
  );
  assert.equal(sameTimestamp(null, "2026-08-11T10:50:41Z"), false);
});
