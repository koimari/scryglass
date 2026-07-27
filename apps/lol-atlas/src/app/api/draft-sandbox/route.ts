import { NextResponse } from "next/server";
import {
  DRAFT_PICK_ORDER,
  DRAFT_POLICY_MIN_ROLE_GAMES,
  DRAFT_ROLES,
} from "@/lib/draftRules";
import {
  CompositionRuntimeUnavailableError,
  compositionRuntimeMetadata,
} from "@/lib/draftComposition";
import {
  exposePatchContracts,
  patchContractFromPublic,
  patchContractFromSource,
  patchContractsFromSource,
} from "@/lib/patch";
import { draftRuntimeBindingEvidence } from "@/lib/modelValidation";
import { readPackJson, readPackManifest } from "@/lib/serverPack";
import {
  analyzeDraftSandbox,
  draftCatalog,
  normalizeDraftChampion,
  type DraftAction,
  type DraftCandidateRole,
  type DraftRole,
  type DraftSide,
} from "@/lib/draftScore";

export const runtime = "nodejs";

type Body = {
  actions?: Array<{
    side?: string;
    champion?: string;
    role?: string | null;
  }>;
  perspective?: string;
  next_side?: string;
  candidate_role?: string;
  excluded?: string[];
  league?: string | null;
  public_patch?: string | null;
  elo_diff?: number | null;
  limit?: number;
};

const PUBLIC_INTERNAL_ERROR =
  "Draft analysis could not be completed. Please retry with a new request.";
const PUBLIC_RUNTIME_ERROR =
  "Draft analysis is unavailable because its versioned model runtime did not pass validation.";

function requestId(): string {
  return crypto.randomUUID();
}

export function publicDraftApiError(
  error: unknown,
  route: "draft-sandbox" | "draft-wr",
): NextResponse {
  const id = requestId();
  console.error(`[${route}] request failed`, { requestId: id, error });
  return NextResponse.json(
    {
      error: PUBLIC_INTERNAL_ERROR,
      code: "draft_internal_error",
      request_id: id,
    },
    { status: 500 },
  );
}

function runtimeUnavailable(): NextResponse {
  return NextResponse.json(
    {
      error: PUBLIC_RUNTIME_ERROR,
      code: "composition_runtime_unavailable",
    },
    { status: 503 },
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isSide(value: unknown): value is DraftSide {
  return value === "blue" || value === "red";
}

function isRole(value: unknown): value is DraftRole {
  return DRAFT_ROLES.includes(value as DraftRole);
}

function isCandidateRole(value: unknown): value is DraftCandidateRole {
  return value === "open" || value === "any" || isRole(value);
}

function canonicalChampion(
  value: string,
  catalog: ReturnType<typeof draftCatalog>,
): string | null {
  const normalized = normalizeDraftChampion(value);
  return (
    catalog.find(
      (champion) => champion.name.toLocaleLowerCase() === normalized.toLocaleLowerCase(),
    )?.name ?? null
  );
}

async function currentBoundRuntime() {
  try {
    const metadata = compositionRuntimeMetadata();
    if (!metadata) return null;
    const manifest = await readPackManifest();
    const gateArtifact = await readPackJson(
      manifest,
      "models/model_validation_2026-07-27.json",
    );
    const binding = draftRuntimeBindingEvidence(
      manifest,
      gateArtifact,
      metadata,
    );
    return binding ? { metadata, binding } : null;
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    const bound = await currentBoundRuntime();
    if (!bound) return runtimeUnavailable();
    const { metadata, binding } = bound;
    const currentPatch = patchContractFromSource(
      metadata.latest_observed_patch,
    );
    return NextResponse.json({
      champions: draftCatalog(),
      roles: DRAFT_ROLES,
      model: "Experimental composition policy value · bounded beam minimax",
      value_kind: "experimental_composition_policy_value",
      probability_status: "withheld_failed_chronological_gate",
      candidate_role_policy: "supported_pro_roles_minimum_maps",
      candidate_role_minimum_maps: DRAFT_POLICY_MIN_ROLE_GAMES,
      patch_selection: {
        required: true,
        request_field: "public_patch",
        mode: "current_pooled_or_historical_exact",
        contract: "public 25.x -> source 15.x; public 26.x -> source 16.x",
        current_patch_supported: currentPatch != null,
        current_patch: currentPatch,
        patch_specific_patches: patchContractsFromSource(
          metadata.supported_patches,
        ),
        pooled_holdout_patches: patchContractsFromSource(
          metadata.observed_holdout_patches,
        ),
        analysis_patches: patchContractsFromSource(
          metadata.analysis_patches,
        ),
      },
      runtime_binding: binding,
      model_metadata: exposePatchContracts(metadata),
    });
  } catch (error) {
    if (error instanceof CompositionRuntimeUnavailableError) {
      return runtimeUnavailable();
    }
    return publicDraftApiError(error, "draft-sandbox");
  }
}

export async function POST(request: Request) {
  try {
    let parsed: unknown;
    try {
      parsed = await request.json();
    } catch {
      return NextResponse.json(
        { error: "request body must be valid JSON" },
        { status: 400 },
      );
    }
    if (!isRecord(parsed)) {
      return NextResponse.json(
        { error: "request body must be an object" },
        { status: 400 },
      );
    }
    const body = parsed as Body;
    const bound = await currentBoundRuntime();
    if (!bound) return runtimeUnavailable();
    const { metadata, binding } = bound;
    const catalog = draftCatalog();
    if (!isSide(body.perspective) || !isSide(body.next_side)) {
      return NextResponse.json(
        { error: "perspective and next_side must be blue or red" },
        { status: 400 },
      );
    }
    if (body.candidate_role != null && !isCandidateRole(body.candidate_role)) {
      return NextResponse.json({ error: "candidate_role is invalid" }, { status: 400 });
    }
    if (body.actions != null && !Array.isArray(body.actions)) {
      return NextResponse.json({ error: "actions must be an array" }, { status: 400 });
    }
    const rawActions = body.actions ?? [];
    if (rawActions.length > 10) {
      return NextResponse.json({ error: "a draft can contain at most 10 picks" }, { status: 400 });
    }

    const actions: DraftAction[] = [];
    const seen = new Set<string>();
    const hasVerifiedRoles = rawActions.every(
      (action) =>
        typeof action === "object" &&
        action != null &&
        typeof action.role === "string" &&
        isRole(action.role),
    );
    for (const [index, action] of rawActions.entries()) {
      if (!isRecord(action)) {
        return NextResponse.json(
          { error: `pick ${index + 1} must be an object` },
          { status: 400 },
        );
      }
      if (!isSide(action.side) || !action.champion) {
        return NextResponse.json(
          { error: "every action needs a side and champion" },
          { status: 400 },
        );
      }
      if (action.side !== DRAFT_PICK_ORDER[index]) {
        return NextResponse.json(
          {
            error: `pick ${index + 1} must belong to ${DRAFT_PICK_ORDER[index]} side`,
          },
          { status: 400 },
        );
      }
      if (action.role != null && !isRole(action.role)) {
        return NextResponse.json({ error: `invalid role for ${action.champion}` }, { status: 400 });
      }
      if (typeof action.champion !== "string") {
        return NextResponse.json(
          { error: `pick ${index + 1} champion must be text` },
          { status: 400 },
        );
      }
      const champion = canonicalChampion(action.champion, catalog);
      if (!champion) {
        return NextResponse.json({ error: `unknown champion: ${action.champion}` }, { status: 400 });
      }
      const key = champion.toLocaleLowerCase();
      if (seen.has(key)) {
        return NextResponse.json({ error: `${champion} is already selected` }, { status: 400 });
      }
      seen.add(key);
      actions.push({ side: action.side, champion, role: (action.role as DraftRole) ?? null });
    }
    const expectedNextSide = DRAFT_PICK_ORDER[actions.length] ?? null;
    if (expectedNextSide && body.next_side !== expectedNextSide) {
      return NextResponse.json(
        { error: `next_side must be ${expectedNextSide} after ${actions.length} picks` },
        { status: 400 },
      );
    }

    const sideCounts = {
      blue: actions.filter((action) => action.side === "blue").length,
      red: actions.filter((action) => action.side === "red").length,
    };
    if (sideCounts.blue > 5 || sideCounts.red > 5) {
      return NextResponse.json({ error: "each side can select at most five champions" }, { status: 400 });
    }
    if (hasVerifiedRoles) {
      for (const side of ["blue", "red"] as const) {
        const roles = actions
          .filter((action) => action.side === side && action.role)
          .map((action) => action.role as DraftRole);
        if (new Set(roles).size !== roles.length) {
          return NextResponse.json(
            { error: `${side} side cannot assign two champions to the same role` },
            { status: 400 },
          );
        }
      }
    }

    if (
      body.excluded != null &&
      (!Array.isArray(body.excluded) ||
        body.excluded.some((champion) => typeof champion !== "string"))
    ) {
      return NextResponse.json(
        { error: "excluded must be an array of champion names" },
        { status: 400 },
      );
    }
    const unknownExcluded = (body.excluded || []).find(
      (champion) => !canonicalChampion(champion, catalog),
    );
    if (unknownExcluded) {
      return NextResponse.json(
        { error: `unknown excluded champion: ${unknownExcluded}` },
        { status: 400 },
      );
    }
    const excluded = (body.excluded || []).map(
      (champion) => canonicalChampion(champion, catalog)!,
    );
    if (
      new Set(excluded.map((champion) => champion.toLocaleLowerCase())).size !==
      excluded.length
    ) {
      return NextResponse.json(
        { error: "excluded champions must be unique" },
        { status: 400 },
      );
    }
    const excludedKeys = new Set(excluded.map((champion) => champion.toLocaleLowerCase()));
    const overlap = [...seen].find((champion) => excludedKeys.has(champion));
    if (overlap) {
      return NextResponse.json(
        { error: `${overlap} cannot be both selected and unavailable` },
        { status: 400 },
      );
    }
    const eloDiff = body.elo_diff == null ? null : Number(body.elo_diff);
    if (eloDiff != null && !Number.isFinite(eloDiff)) {
      return NextResponse.json({ error: "elo_diff must be numeric" }, { status: 400 });
    }
    if (eloDiff != null) {
      return NextResponse.json(
        {
          error:
            "elo_diff is not accepted by the composition-only Sandbox policy",
        },
        { status: 400 },
      );
    }
    if (typeof body.league !== "string" || !body.league.trim()) {
      return NextResponse.json(
        { error: "league is required" },
        { status: 400 },
      );
    }
    const league = body.league.trim().toUpperCase();
    if (!metadata.supported_leagues.includes(league)) {
      return NextResponse.json(
        {
          error: `league ${league} is outside this model artifact`,
          supported_leagues: metadata.supported_leagues,
        },
        { status: 400 },
      );
    }
    if (
      typeof body.public_patch !== "string" ||
      !body.public_patch.trim()
    ) {
      return NextResponse.json(
        {
          error:
            "public_patch is required; use an exact public patch such as 26.14",
          code: "public_patch_required",
          analysis_patches: patchContractsFromSource(
            metadata.analysis_patches,
          ),
        },
        { status: 400 },
      );
    }
    const patchContract = patchContractFromPublic(body.public_patch);
    if (!patchContract) {
      return NextResponse.json(
        {
          error:
            "public_patch must be an exact 25.x or 26.x patch with a two-digit minor, such as 26.14",
          code: "invalid_public_patch",
        },
        { status: 400 },
      );
    }
    if (
      !metadata.analysis_patches.includes(patchContract.source_patch_key)
    ) {
      return NextResponse.json(
        {
          error: `public patch ${patchContract.public_patch} is outside the observed model artifact`,
          code: "unsupported_observed_patch",
          analysis_patches: patchContractsFromSource(
            metadata.analysis_patches,
          ),
        },
        { status: 400 },
      );
    }
    const limit = body.limit == null ? 12 : Number(body.limit);
    if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
      return NextResponse.json(
        { error: "limit must be an integer from 1 to 200" },
        { status: 400 },
      );
    }

    const analysis = analyzeDraftSandbox({
      actions,
      perspective: body.perspective,
      next_side: body.next_side,
      candidate_role: body.candidate_role as DraftCandidateRole | undefined,
      excluded,
      league,
      patch: patchContract.source_patch_key,
      elo_diff: null,
      limit,
    });
    const modelContext = analysis.model_context
      ? Object.fromEntries(
          Object.entries(analysis.model_context).filter(
            ([key]) => key !== "normalized_patch",
          ),
        )
      : {};
    return NextResponse.json({
      ...analysis,
      current: {
        ...analysis.current,
        audit: {
          ...analysis.current.audit,
          release_runtime_binding: "matched",
        },
      },
      runtime_binding: binding,
      patch_context: patchContract,
      model_context: {
        ...modelContext,
        ...patchContract,
        active_pack_id: binding.packId,
        runtime_binding_status: binding.status,
      },
    });
  } catch (error) {
    if (error instanceof CompositionRuntimeUnavailableError) {
      return runtimeUnavailable();
    }
    return publicDraftApiError(error, "draft-sandbox");
  }
}
