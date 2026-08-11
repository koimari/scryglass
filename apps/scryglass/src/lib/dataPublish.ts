const RELEASE_ID = /^v\d{4}\.\d{2}\.\d{2}\.\d{6}$/;

export function validReleaseId(value: unknown): value is string {
  return typeof value === "string" && RELEASE_ID.test(value);
}

export function validPublishSecret(received: string | null, expected: string | undefined): boolean {
  return Boolean(expected && received === `Bearer ${expected}`);
}
