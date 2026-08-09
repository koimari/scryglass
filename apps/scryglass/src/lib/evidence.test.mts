import assert from "node:assert/strict";
import test from "node:test";
import { EVIDENCE_CONTRACT, evidenceInfo, evidenceState, type EvidenceFields } from "./evidence";

/**
 * Adversarial fixtures for the validated evidence contract (issue #46).
 * Thresholds are exact: a row at exactly 95% relative precision is NOT
 * settled; stale, inactive, wide, fallback, disconnected, OOD, and
 * unsupported rows fail closed.
 */

function settledFields(over: Partial<EvidenceFields> = {}): EvidenceFields {
  return {
    evidence_interval_width: 2 * 1.959963984540054 * 25, // sigma 25
    evidence_precision_ratio: 1.0,
    evidence_stability: 0.4,
    evidence_freshness_days: 2,
    evidence_support_coverage: 1.0,
    evidence_fallback: 0,
    evidence_active: 1,
    evidence_disconnected: 0,
    evidence_ood: 0,
    ...over,
  };
}

test("a fully supported fresh stable row is settled", () => {
  assert.equal(evidenceState(settledFields()), "settled");
});

test("exactly 95% relative precision is NOT settled (strict threshold)", () => {
  const fields = settledFields({
    evidence_precision_ratio: EVIDENCE_CONTRACT.settledPrecisionRatio,
  });
  assert.equal(evidenceState(fields), "observed");
});

test("precision just above 95% with everything else met is settled", () => {
  const fields = settledFields({
    evidence_precision_ratio: EVIDENCE_CONTRACT.settledPrecisionRatio + 1e-9,
  });
  assert.equal(evidenceState(fields), "settled");
});

test("unknown stability never settles (uncertainty is not map count)", () => {
  assert.equal(
    evidenceState(settledFields({ evidence_stability: null })),
    "thin",
  );
  // A large map count alone cannot settle a row without stability evidence.
  assert.equal(
    evidenceState(
      settledFields({ evidence_stability: null, evidence_support_coverage: 1.0 }),
    ),
    "thin",
  );
});

test("stability above the threshold is observed, not settled", () => {
  const fields = settledFields({
    evidence_stability: EVIDENCE_CONTRACT.settledStability + 0.001,
  });
  assert.equal(evidenceState(fields), "observed");
});

test("stale source rows fail closed before any precision claim", () => {
  assert.equal(
    evidenceState(settledFields({ evidence_freshness_days: EVIDENCE_CONTRACT.staleDays + 1 })),
    "stale",
  );
  assert.equal(
    evidenceState(settledFields({ evidence_freshness_days: null })),
    "stale",
  );
});

test("inactive rows fail closed", () => {
  assert.equal(evidenceState(settledFields({ evidence_active: 0 })), "inactive");
  assert.equal(evidenceState(settledFields({ evidence_active: 2 })), "unsupported");
});

test("wide intervals fail closed", () => {
  assert.equal(
    evidenceState(settledFields({ evidence_interval_width: EVIDENCE_CONTRACT.wideIntervalWidth + 1 })),
    "wide_interval",
  );
});

test("fallback rows fail closed", () => {
  assert.equal(evidenceState(settledFields({ evidence_fallback: 1 })), "fallback");
});

test("disconnected rows fail closed", () => {
  assert.equal(evidenceState(settledFields({ evidence_disconnected: 1 })), "disconnected");
});

test("out-of-distribution rows fail closed", () => {
  assert.equal(evidenceState(settledFields({ evidence_ood: 1 })), "ood");
});

test("rows missing any required field are unsupported and never settled", () => {
  const required: (keyof EvidenceFields)[] = [
    "evidence_interval_width",
    "evidence_precision_ratio",
    "evidence_support_coverage",
    "evidence_fallback",
    "evidence_active",
    "evidence_disconnected",
    "evidence_ood",
  ];
  for (const key of required) {
    const fields = settledFields({ [key]: undefined });
    assert.equal(evidenceState(fields), "unsupported", `${key} must fail closed`);
  }
  // Non-finite numbers are unsupported, not settled.
  assert.equal(
    evidenceState(settledFields({ evidence_interval_width: Number.NaN })),
    "unsupported",
  );
  assert.equal(
    evidenceState(settledFields({ evidence_precision_ratio: Number.POSITIVE_INFINITY })),
    "unsupported",
  );
});

test("legacy packs without evidence fields are unsupported", () => {
  assert.equal(evidenceState({}), "unsupported");
});

test("a fresh row just outside the fresh window is observed, not settled", () => {
  const fields = settledFields({ evidence_freshness_days: EVIDENCE_CONTRACT.freshDays + 1 });
  assert.equal(evidenceState(fields), "observed");
});

test("support coverage below the target is observed or thin, never settled", () => {
  const fields = settledFields({ evidence_support_coverage: 0.99 });
  assert.equal(evidenceState(fields), "observed");
  const weak = settledFields({ evidence_support_coverage: 0.49 });
  assert.equal(evidenceState(weak), "thin");
});

test("evidenceInfo renders the basis and keeps sigma/games as diagnostics", () => {
  const info = evidenceInfo(settledFields(), 25, 40);
  assert.equal(info.label, "High confidence");
  assert.ok(info.detail.includes("95% interval"));
  assert.ok(info.detail.includes("precision"));
  assert.ok(info.detail.includes("since last game"));
  assert.ok(info.detail.includes("support"));
  assert.ok(info.detail.includes("stability"));
  assert.equal(info.sigma, 25);
  assert.equal(info.games, 40);
});

test("unsupported rows render a fail-closed label and explanation", () => {
  const info = evidenceInfo({}, 25, 40);
  assert.equal(info.label, "Data unavailable");
  assert.ok(info.layman.includes("does not support"));
});

test("public confidence labels avoid internal evidence terms", () => {
  assert.equal(evidenceInfo(settledFields(), 25, 40).label, "High confidence");
  assert.equal(
    evidenceInfo(
      settledFields({ evidence_precision_ratio: EVIDENCE_CONTRACT.settledPrecisionRatio }),
      30,
      40,
    ).label,
    "Medium confidence",
  );
  assert.equal(
    evidenceInfo(
      settledFields({ evidence_interval_width: EVIDENCE_CONTRACT.wideIntervalWidth + 1 }),
      60,
      10,
    ).label,
    "Low confidence",
  );
});
