import { createHash, timingSafeEqual } from "node:crypto";

const RELEASE_ID = /^v\d{4}\.\d{2}\.\d{2}\.\d{6}$/;

export function validReleaseId(value: unknown): value is string {
  return typeof value === "string" && RELEASE_ID.test(value);
}

export function validPublishSecret(received: string | null, expected: string | undefined): boolean {
  if (!expected || !received) return false;
  const receivedDigest = createHash("sha256").update(received, "utf8").digest();
  const expectedDigest = createHash("sha256").update(`Bearer ${expected}`, "utf8").digest();
  return timingSafeEqual(receivedDigest, expectedDigest);
}

export function validDiagnosticSecret(
  received: string | null,
  diagnosticSecret: string | undefined,
): boolean {
  return validPublishSecret(received, diagnosticSecret);
}
