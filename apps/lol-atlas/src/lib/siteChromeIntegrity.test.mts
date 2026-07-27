import assert from "node:assert/strict";
import test from "node:test";
import {
  packClockLabels,
  packSourceLabel,
  packYearsLabel,
  type PublicPackStatus,
} from "../components/SiteChrome.tsx";

test("site chrome never collapses publication and observation clocks", () => {
  const manifest = {
    pack_id: "v-test",
    degraded: false,
    clocks: {
      publication: { value: "2026-07-27T12:00:00Z" },
      data_through: { value: "2026-07-25T12:00:00Z" },
    },
  } satisfies PublicPackStatus;
  const labels = packClockLabels(
    manifest,
    Date.parse("2026-07-27T13:00:00Z"),
  );
  assert.deepEqual(labels, {
    published: "1h ago",
    dataThrough: "2d ago",
    degraded: false,
  });
});

test("missing observation cutoff stays unavailable rather than using publication time", () => {
  const manifest = {
    pack_id: "v-test",
    degraded: true,
    clocks: {
      publication: { value: "2026-07-27T12:00:00Z" },
      data_through: { value: null },
    },
  } satisfies PublicPackStatus;
  const labels = packClockLabels(
    manifest,
    Date.parse("2026-07-27T13:00:00Z"),
  );
  assert.equal(labels.published, "1h ago");
  assert.equal(labels.dataThrough, null);
  assert.equal(labels.degraded, true);
});

test("footer source label follows typed pack provenance", () => {
  assert.equal(
    packSourceLabel({
      source_provenance: {
        sources: [{ source: "oe" }, { source: "grid" }],
      },
    }),
    "Oracle’s Elixir + GRID",
  );
  assert.equal(packSourceLabel(null), "Source provenance unavailable");
});

test("footer years derive from validated manifest filters", () => {
  assert.equal(
    packYearsLabel({ filters: { years: [2026, 2025, 2026] } }),
    "2025–2026",
  );
  assert.equal(packYearsLabel({ filters: { years: ["2026", null] } }), null);
  assert.equal(packYearsLabel(null), null);
});
