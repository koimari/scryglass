export type PatchContract = Readonly<{
  public_patch: string;
  source_patch_key: string;
}>;

type SourcePatchMetadata = {
  latest_observed_patch: string | null;
  observed_holdout_patches: string[];
  analysis_patches: string[];
  supported_patches: string[];
};

const PUBLIC_TO_SOURCE_MAJOR = new Map<number, number>([
  [25, 15],
  [26, 16],
]);
const SOURCE_TO_PUBLIC_MAJOR = new Map<number, number>([
  [15, 25],
  [16, 26],
]);
const EXACT_PATCH = /^(\d+)\.(\d{2})$/;
const PATCH_OR_BUILD = /^(?:patch\s*)?(\d+)\.(\d{2})(?:\.\d+)*$/i;

function normalizedMatch(
  value: unknown,
  pattern: RegExp,
): { major: number; minor: string } | null {
  if (typeof value !== "string") return null;
  const match = pattern.exec(value.trim());
  if (!match) return null;
  const major = Number(match[1]);
  if (!Number.isSafeInteger(major)) return null;
  return { major, minor: match[2] };
}

/** Normalize an exact major.two-digit-minor patch identity. */
export function normalizeExactPatch(value: unknown): string | null {
  const match = normalizedMatch(value, EXACT_PATCH);
  return match ? `${match.major}.${match.minor}` : null;
}

/** Normalize an exact patch or full build while preserving its two-digit minor. */
export function normalizePatchOrBuild(value: unknown): string | null {
  const match = normalizedMatch(value, PATCH_OR_BUILD);
  return match ? `${match.major}.${match.minor}` : null;
}

export function patchContractFromPublic(
  value: unknown,
): PatchContract | null {
  const publicPatch = normalizeExactPatch(value);
  if (!publicPatch) return null;
  const [majorText, minor] = publicPatch.split(".");
  const sourceMajor = PUBLIC_TO_SOURCE_MAJOR.get(Number(majorText));
  if (sourceMajor == null) return null;
  return {
    public_patch: publicPatch,
    source_patch_key: `${sourceMajor}.${minor}`,
  };
}

export function patchContractFromSource(
  value: unknown,
): PatchContract | null {
  const sourcePatchKey = normalizeExactPatch(value);
  if (!sourcePatchKey) return null;
  const [majorText, minor] = sourcePatchKey.split(".");
  const publicMajor = SOURCE_TO_PUBLIC_MAJOR.get(Number(majorText));
  if (publicMajor == null) return null;
  return {
    public_patch: `${publicMajor}.${minor}`,
    source_patch_key: sourcePatchKey,
  };
}

export function patchContractsFromSource(
  values: readonly string[],
): PatchContract[] {
  const contracts = new Map<string, PatchContract>();
  for (const value of values) {
    const contract = patchContractFromSource(value);
    if (contract) contracts.set(contract.source_patch_key, contract);
  }
  return [...contracts.values()];
}

/**
 * Remove ambiguous source-key fields from metadata crossing a public API
 * boundary and replace them with explicitly named public/source contracts.
 */
export function exposePatchContracts<T extends SourcePatchMetadata>(
  metadata: T,
): Omit<T, keyof SourcePatchMetadata> & {
  latest_observed_patch_contract: PatchContract | null;
  observed_holdout_patch_contracts: PatchContract[];
  analysis_patch_contracts: PatchContract[];
  supported_patch_contracts: PatchContract[];
} {
  const {
    latest_observed_patch,
    observed_holdout_patches,
    analysis_patches,
    supported_patches,
    ...rest
  } = metadata;
  return {
    ...rest,
    latest_observed_patch_contract:
      patchContractFromSource(latest_observed_patch),
    observed_holdout_patch_contracts:
      patchContractsFromSource(observed_holdout_patches),
    analysis_patch_contracts: patchContractsFromSource(analysis_patches),
    supported_patch_contracts: patchContractsFromSource(supported_patches),
  };
}
