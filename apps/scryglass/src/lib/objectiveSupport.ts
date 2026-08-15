import { publicPatchLabel } from "./patchIdentity";

export type TeamObjectiveKey =
  | "towers"
  | "dragons"
  | "barons"
  | "void_grubs"
  | "heralds"
  | "atakhans"
  | "inhibitors";

export const TEAM_OBJECTIVE_FIELDS: ReadonlyArray<{ key: TeamObjectiveKey; label: string }> = [
  { key: "towers", label: "Towers" },
  { key: "dragons", label: "Dragons" },
  { key: "barons", label: "Barons" },
  { key: "void_grubs", label: "Grubs" },
  { key: "heralds", label: "Heralds" },
  { key: "atakhans", label: "Atakhan" },
  { key: "inhibitors", label: "Inhibitors" },
];

function patchParts(value: string | null | undefined): [number, number] | null {
  if (!value) return null;
  const match = /^(\d+)\.(\d{1,2})(?:\.\d+)?$/.exec(publicPatchLabel(value));
  if (!match) return null;
  return [Number(match[1]), Number(match[2])];
}

/** Atakhan existed in the accepted 25.x source window and was removed in 26.01. */
export function supportsAtakhans(patch: string | null | undefined): boolean {
  const parts = patchParts(patch);
  return parts?.[0] === 25;
}

export function objectiveFieldsForPatch(
  patch: string | null | undefined,
): ReadonlyArray<{ key: TeamObjectiveKey; label: string }> {
  return TEAM_OBJECTIVE_FIELDS.filter(({ key }) => key !== "atakhans" || supportsAtakhans(patch));
}
