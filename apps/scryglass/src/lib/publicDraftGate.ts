import { canonicalDraftServingAuthorityAvailable } from "./draftTerminalServer";

export const DRAFT_UNAVAILABLE_MESSAGE =
  "Draft Score is not published yet; its independent validation is incomplete.";

export const DRAFT_UNAVAILABLE_RESPONSE = {
  status: "unavailable",
  error: {
    code: "model_not_promoted",
    message: DRAFT_UNAVAILABLE_MESSAGE,
    retryable: false,
    missing_fields: ["independent_l2_authority"],
    stale_fields: [],
  },
} as const;

export const DRAFT_INTERNAL_ERROR_RESPONSE = {
  status: "unavailable",
  error: {
    code: "internal_error",
    message: "Draft Score is temporarily unavailable.",
    retryable: true,
    missing_fields: [],
    stale_fields: [],
  },
} as const;

/**
 * Legacy exploratory routes remain closed in the public lane. Environment
 * variables are intentionally not an authority mechanism; the canonical v2
 * route uses the artifact-backed gate below.
 */
export function publicPredictiveDraftsEnabled(): boolean {
  return false;
}

/** The v2 route has a real artifact-backed gate; legacy routes remain closed. */
export function canonicalPublicPredictiveDraftsEnabled(): boolean {
  return canonicalDraftServingAuthorityAvailable();
}
