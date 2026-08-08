export type ElementId =
  | "infernal"
  | "mountain"
  | "ocean"
  | "cloud"
  | "hextech"
  | "chemtech";

export type TeamSide = "A" | "B";
export type MapSide = "blue" | "red";
export type Role = "Top" | "Jungle" | "Mid" | "Bot" | "Support";

export const ELEMENTS: ElementId[] = [
  "infernal",
  "mountain",
  "ocean",
  "cloud",
  "hextech",
  "chemtech",
];

export const ROLES: Role[] = ["Top", "Jungle", "Mid", "Bot", "Support"];

const CHAMPION_RESIDUAL_SCHEMA_VERSIONS = new Set([
  "elemental-drake-explorer-v3",
  "elemental-drake-explorer-v4",
]);

export type ChampionSnapshot = {
  champion: string;
  role: string;
  tags: string[];
};

export type PilotTeam = {
  side: MapSide;
  name: string;
  composition: ChampionSnapshot[];
};

export type ObservedCapture = {
  globalIndex: number;
  element: ElementId;
  timeSeconds: number;
  ownerSide: MapSide;
  ownerName: string;
  ownerStack: number;
};

export type PilotGame = {
  id: string;
  tournament: string;
  patch: string;
  complete: boolean;
  date: string;
  league: string;
  region: string;
  regionLabel: string;
  competitionLevel: "tier1" | "international" | "other-pro";
  competitionLevelLabel: string;
  teams: PilotTeam[];
  observedCaptures: ObservedCapture[];
};

export type Mechanic = {
  id: ElementId;
  name: string;
  short: string;
  perStack: string;
  unit: string;
  value: number;
  directTags: string[];
  source: string;
  sourceUrl: string;
};

export type CoverageRegion = {
  id: string;
  label: string;
  games: number;
  tierOneGames: number;
  internationalGames: number;
  otherProGames: number;
  competitions: Array<{ name: string; games: number }>;
};

export type CompetitionCoverage = {
  eligibleGames: number;
  tierOneGames: number;
  internationalGames: number;
  otherProGames: number;
  unclassifiedGames: number;
  regions: CoverageRegion[];
  taxonomy: string;
};

export type RoleCatalog = {
  status: "ready" | "unavailable";
  source: string;
  appearances: number;
  games: number;
  minimumRandomAppearances: number;
  roles: Array<{
    role: Role;
    champions: Array<{ name: string; appearances: number }>;
  }>;
};

export type ChampionCatalogEntry = {
  name: string;
  tags: string[];
  proGameAppearances: number;
  allocationKind?: "reconciled-allocation";
  allocationSource?:
    | "element-specific-direct-or-fallback"
    | "archetype-fallback"
    | "team-common-only";
  fallback: string | null;
  elementEvidence?: Partial<
    Record<
      ElementId,
      {
        source:
          | "direct-residual"
          | "archetype-fallback"
          | "team-common-only";
        championEligible?: boolean;
        eligible?: boolean;
        directEligible?: boolean;
        exposureEligible?: boolean;
        featureName: string | null;
        trainingGames: number;
        trainingSeries: number;
        ownershipGames?: number;
        nonOwnershipGames?: number;
        orgRosters?: number;
        failedExposureRules?: string[];
        failedRules?: string[];
        vocabularyProvenance?:
          | "publication-audit-vocabulary"
          | "post-audit-full-refit"
          | null;
        provenance?:
          | "publication-audit-vocabulary"
          | "post-audit-full-refit"
          | null;
        individualCellValidated?: boolean;
        status:
          | "ready"
          | "withheld"
          | "below-threshold"
          | "unobserved"
          | "unsupported";
      }
    >
  >;
};

export type ChampionElementEvidence = NonNullable<
  NonNullable<ChampionCatalogEntry["elementEvidence"]>[ElementId]
>;

export type RuntimeFeature = {
  name: string;
  weight: number;
  clipLow: number;
  clipHigh: number;
  observedMin: number;
  observedMax: number;
  family?: string;
  champion?: string;
  element?: ElementId;
};

export type EffectiveRuntime = {
  format: "effective-raw-logit-v1";
  intercept: number;
  features: RuntimeFeature[];
  standardizationFolded: boolean;
  clipProtocol: string;
  reconciliation: string;
};

export type ModelDiagnostics = {
  trainRows: number;
  holdoutRows: number;
  trainGames: number;
  holdoutGames: number;
  trainSeries: number;
  holdoutSeries: number;
  holdoutStart: string;
  holdoutEnd: string;
  postHoldoutRows: number;
  auc: number;
  brier: number;
  nullBrier: number;
  logLoss: number;
  ece10: number;
  selectedAlpha: number;
  innerValidationBrier: number;
  weighting: string;
};

export type ExplorerModel = {
  schemaVersion: string;
  status: "ready";
  provenance?: {
    featureBuilderVersion?: string;
    inputs?: Record<
      string,
      {
        file?: string;
        bytes: number;
        sha256: string;
      }
    >;
    [key: string]: unknown;
  };
  cohort: {
    completedGames: number;
    modeledGames: number;
    series: number;
    captures: number;
    mirroredPerspectiveRows: number;
    dateMin: string;
    dateMax: string;
  };
  featureSchema: {
    version: string;
    elements: ElementId[];
    archetypes: string[];
    stateInputs: string[];
    legalStateRules: string[];
    inventoryInputs: {
      maximumPerTeam: number;
      maximumGlobalStage: number;
    };
    championAllocation?: {
      kind: "reconciled-allocation";
      taggedSource: "archetype-fallback";
      untaggedSource: "team-common-only";
      championSpecificEmpiricalEvidence: boolean;
      separatelyFittedChampionEffects: boolean;
      directResidualFamily?: {
        family: string;
        status: "ready" | "withheld" | "unavailable";
        familyGate: number;
        sourceWhenReady: "direct-residual";
        fallbackWhenWithheld: string;
        interpretation: string;
      };
    };
    genericDraftScoreContext?: {
      appliedToDragonEstimate: false;
      championEffectsApplied: false;
      allySynergyApplied: false;
      enemyCounterApplied: false;
      reason: string;
    };
  };
  championCatalog: ChampionCatalogEntry[];
  stageReference: Array<{
    stage: number;
    medianSeconds: number;
    perspectiveRows: number;
  }>;
  support: {
    exactComposition: {
      uniqueFiveChampionSides: number;
      medianGamesPerExactSide: number;
      maximumGamesForOneExactSide: number;
      interpretation: string;
    };
    soulPerspectiveRows: number;
    outOfDistributionRule: string;
  };
  models: {
    jointState: {
      estimand: string;
      diagnostics: ModelDiagnostics;
      runtime: EffectiveRuntime;
      overallElementRankings: {
        estimand: string;
        estimands?: Record<string, string>;
        unit: string;
        rankings: OverallElementRanking[];
        support: {
          modeledGames: number;
          actualDraftPerspectives: number;
          resolvedCaptures?: number;
          soulCaptures?: number;
          stageReferencePerspectiveRows?: Record<string, number>;
          observedFirstCapturesByElement?: Record<ElementId, number>;
          observedSecondCapturesByElement?: Record<ElementId, number>;
          observedMapPhaseCapturesByElement?: Record<ElementId, number>;
          observedSoulCapturesByElement?: Record<ElementId, number>;
          legalFirstContextsPerElement?: number;
          legalSecondContextsPerElement?: number;
          legalOpeningPairsPerMapElement?: number;
          openingOwnerAssignmentsPerPair?: number;
          legalMapPathsPerElement?: number;
          mapCaptureIncrementsPerElement?: number;
          mapPathCountsByCaptureLength?: Record<string, number>;
          /** Compatibility with artifacts generated before the stage matrix. */
          firstStageReferencePerspectiveRows?: number;
          soulStageReferencePerspectiveRows?: number;
          legalOpeningPairsPerSoulElement?: number;
          championResidualApplied: boolean;
          championResidualFeatureCount: number;
        };
        weighting: Record<string, string>;
        reference: {
          state?: string;
          comparison?: string;
          baseline?: Record<string, unknown>;
          firstCapture: Record<string, unknown>;
          secondCapture?: Record<string, unknown>;
          mapPhaseCapture?: Record<string, unknown>;
          perfectControlSoul?: Record<string, unknown>;
        };
        reconciliation: string;
        ordering: string;
        pointEstimateCaveat?: string;
      };
      championResidual?: {
        status: "ready" | "withheld" | "unavailable";
        familyGate: number;
        diagnostics?: {
          publicationExpansionAudit?: {
            status: "ready" | "withheld" | "not-run";
            reason?: string;
            [key: string]: unknown;
          };
          [key: string]: unknown;
        };
        vocabularies?: {
          publication?: {
            cells: number;
            champions: number;
            degreesOfFreedom: number;
            frozenFrom?: string;
            [key: string]: unknown;
          } | null;
          [key: string]: unknown;
        };
        [key: string]: unknown;
      };
    };
    captureAllocation: {
      estimand: string;
      diagnostics: ModelDiagnostics;
      runtime: EffectiveRuntime;
      counterfactualProtocol: string;
      selectionDiagnostics: {
        resolvedCaptures: number;
        rawCapturerWinRate: number;
        stateLagSeconds: {
          median: number;
          p95: number;
          maximumAllowed: number;
        };
        byStage: Array<{
          stage: number;
          captures: number;
          rawCapturerWinRate: number;
        }>;
        interpretation: string;
      };
    };
  };
  publicWording: {
    champions: string;
    lines: string;
    allocation: string;
    draftContext?: string;
  };
  controls: string[];
  limitations: string[];
};

export type StudyArtifact = {
  metadata: {
    generatedAt: string;
    patches: string[];
    provider: string;
    estimationStatus: string;
    explorerModelSource?: {
      file: string;
      bytes: number;
      sha256: string;
      schemaVersion: string;
    } | null;
  };
  mechanics: Mechanic[];
  competitionCoverage: CompetitionCoverage;
  roleCatalog: RoleCatalog;
  pilotGames: PilotGame[];
  explorerModel: ExplorerModel;
};

export type Capture = {
  element: ElementId;
  owner: TeamSide;
};

export type Inventory = Record<ElementId, number>;

export type TimelineState = {
  stage: number;
  inventoryA: Inventory;
  inventoryB: Inventory;
  soulA: ElementId | null;
  soulB: ElementId | null;
};

export type ChampionDifferentialSource =
  | "champion-informed"
  | "mixed"
  | "archetype-prior-only"
  | "unsupported";

export type ChampionEvidenceSummary = {
  source: ChampionDifferentialSource;
  activeElements: ElementId[];
  informedElements: ElementId[];
  activeEvidence: ChampionElementEvidence[];
  leastSupportedElement: ElementId | null;
  minimumTrainingGames: number | null;
  minimumTrainingSeries: number | null;
  minimumOwnershipGames: number | null;
  minimumNonOwnershipGames: number | null;
  failedExposureRules: string[];
  vocabularyProvenance:
    | "publication-audit-vocabulary"
    | "post-audit-full-refit"
    | null;
};

export type CurvePoint = {
  stage: number;
  minute: number;
  supportRows: number;
  supportStatus: "baseline" | "supported" | "low-support";
  teamADeltaPp: number;
  teamBDeltaPp: number;
  /**
   * Cumulative champion adjustments for every dragon held by the focused
   * champion's team at this stage, evaluated against the same-time 0/0
   * inventory reference. These are not latest-capture increments.
   */
  championCumulativePp: number[];
  championDifferentialSource: ChampionDifferentialSource[];
  teamContextPp: number;
  soul: { team: TeamSide; element: ElementId } | null;
};

export type AllocationEstimate = {
  element: ElementId;
  stage: number;
  teamADeltaPp: number;
  probabilityIfA: number;
  probabilityIfB: number;
};

export type TakeLeaveEstimate = {
  status:
    | "breakeven-within-support"
    | "leave-favored-at-zero"
    | "no-crossover-within-support"
    | "gold-support-unavailable";
  element: ElementId;
  stage: number;
  minute: number;
  focus: TeamSide;
  takeProbability: number;
  leaveProbability: number;
  differencePp: number;
  breakevenGold: number | null;
  maxCrossMapGold: number;
  verdict:
    | "take-favored"
    | "leave-favored"
    | "effectively-even"
    | "outside-supported-gold-range";
  supportRows: number;
  supportStatus: "supported" | "low-support";
};

export type ElementRanking = {
  element: ElementId;
  firstCapturePp: number;
  perfectControlSoulPp: number;
};

export type OverallElementRanking = {
  element: ElementId;
  firstCapturePp: number;
  secondCapturePp?: number;
  mapPhaseCapturePp?: number;
  /** Compatibility with artifacts generated before the stage matrix. */
  perfectControlSoulPp?: number;
};

const EMPTY_INVENTORY = (): Inventory => ({
  infernal: 0,
  mountain: 0,
  ocean: 0,
  cloud: 0,
  hextech: 0,
  chemtech: 0,
});

export function formatClock(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

export function formatPp(value: number, digits = 2): string {
  if (Math.abs(value) < 0.005) return "0.00 pp";
  return `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)} pp`;
}

export function titleCaseTag(tag: string): string {
  return tag
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function captureElementAt(
  stage: number,
  first: ElementId,
  second: ElementId,
  rift: ElementId,
): ElementId {
  if (stage === 1) return first;
  if (stage === 2) return second;
  return rift;
}

export function legalElementChoice(
  slot: "first" | "second" | "rift",
  candidate: ElementId,
  first: ElementId,
  second: ElementId,
  rift: ElementId,
): boolean {
  if (slot === "first") return candidate !== second && candidate !== rift;
  if (slot === "second") return candidate !== first && candidate !== rift;
  return candidate !== first && candidate !== second;
}

export function randomDistinctElements(
  random: () => number = Math.random,
): [ElementId, ElementId, ElementId] {
  const pool = [...ELEMENTS];
  for (let index = pool.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [pool[index], pool[target]] = [pool[target], pool[index]];
  }
  return [pool[0], pool[1], pool[2]];
}

export function randomRoleTeam(
  roleCatalog: RoleCatalog,
  allowedChampions: Set<string>,
  excluded: Iterable<string> = [],
  random: () => number = Math.random,
): string[] {
  const used = new Set(excluded);
  return ROLES.map((role) => {
    const roleEntry = roleCatalog.roles.find((entry) => entry.role === role);
    const candidates = (roleEntry?.champions ?? []).filter(
      (candidate) =>
        candidate.appearances >= roleCatalog.minimumRandomAppearances &&
        allowedChampions.has(candidate.name) &&
        !used.has(candidate.name),
    );
    const fallback = (roleEntry?.champions ?? []).filter(
      (candidate) =>
        allowedChampions.has(candidate.name) && !used.has(candidate.name),
    );
    const pool = candidates.length ? candidates : fallback;
    if (!pool.length) {
      const champion =
        Array.from(allowedChampions).find((name) => !used.has(name)) ?? "";
      if (champion) used.add(champion);
      return champion;
    }
    const totalWeight = pool.reduce(
      (sum, candidate) => sum + Math.max(1, candidate.appearances),
      0,
    );
    let ticket = random() * totalWeight;
    let selected = pool[pool.length - 1].name;
    for (const candidate of pool) {
      ticket -= Math.max(1, candidate.appearances);
      if (ticket <= 0) {
        selected = candidate.name;
        break;
      }
    }
    used.add(selected);
    return selected;
  });
}

export function sanitizeOwners(owners: TeamSide[]): TeamSide[] {
  let a = 0;
  let b = 0;
  const legal: TeamSide[] = [];
  for (const owner of owners.slice(0, 7)) {
    if (a === 4 || b === 4) break;
    legal.push(owner);
    if (owner === "A") a += 1;
    else b += 1;
  }
  return legal;
}

export function buildCaptures(
  owners: TeamSide[],
  first: ElementId,
  second: ElementId,
  rift: ElementId,
): Capture[] {
  return sanitizeOwners(owners).map((owner, index) => ({
    owner,
    element: captureElementAt(index + 1, first, second, rift),
  }));
}

export function timelineStates(captures: Capture[]): TimelineState[] {
  const inventoryA = EMPTY_INVENTORY();
  const inventoryB = EMPTY_INVENTORY();
  const states: TimelineState[] = [
    {
      stage: 0,
      inventoryA: { ...inventoryA },
      inventoryB: { ...inventoryB },
      soulA: null,
      soulB: null,
    },
  ];
  let soulA: ElementId | null = null;
  let soulB: ElementId | null = null;
  for (const [index, capture] of captures.entries()) {
    const inventory = capture.owner === "A" ? inventoryA : inventoryB;
    inventory[capture.element] += 1;
    const total = ELEMENTS.reduce((sum, element) => sum + inventory[element], 0);
    if (total === 4) {
      if (capture.owner === "A") soulA = capture.element;
      else soulB = capture.element;
    }
    states.push({
      stage: index + 1,
      inventoryA: { ...inventoryA },
      inventoryB: { ...inventoryB },
      soulA,
      soulB,
    });
    if (soulA || soulB) break;
  }
  return states;
}

export function ownerCounts(owners: TeamSide[]): { A: number; B: number } {
  return owners.reduce(
    (counts, owner) => {
      counts[owner] += 1;
      return counts;
    },
    { A: 0, B: 0 },
  );
}

export function defaultPilot(study: StudyArtifact): PilotGame {
  return (
    study.pilotGames.find(
      (game) => game.competitionLevel === "tier1" && game.league === "LCK",
    ) ??
    study.pilotGames.find((game) => game.competitionLevel === "tier1") ??
    study.pilotGames[0]
  );
}

export function pilotSelections(game: PilotGame): {
  teamAName: string;
  teamBName: string;
  teamA: string[];
  teamB: string[];
  owners: TeamSide[];
  first: ElementId;
  second: ElementId;
  rift: ElementId;
} {
  const teamA = game.teams.find((team) => team.side === "blue") ?? game.teams[0];
  const teamB = game.teams.find((team) => team.side === "red") ?? game.teams[1];
  const ordered = (team: PilotTeam | undefined) =>
    ROLES.map(
      (role) =>
        team?.composition.find(
          (champion) => champion.role.toLowerCase() === role.toLowerCase(),
        )?.champion ?? "",
    );
  const captures = [...game.observedCaptures].sort(
    (left, right) => left.globalIndex - right.globalIndex,
  );
  const fallback: ElementId[] = ["infernal", "mountain", "ocean"];
  const first = captures[0]?.element ?? fallback[0];
  const second =
    captures[1]?.element && captures[1].element !== first
      ? captures[1].element
      : fallback.find((element) => element !== first) ?? "mountain";
  const rift =
    captures[2]?.element &&
    ![first, second].includes(captures[2].element)
      ? captures[2].element
      : fallback.find((element) => ![first, second].includes(element)) ?? "ocean";
  return {
    teamAName: teamA?.name ?? "Team A",
    teamBName: teamB?.name ?? "Team B",
    teamA: ordered(teamA),
    teamB: ordered(teamB),
    owners: sanitizeOwners(
      captures.map((capture) => (capture.ownerSide === "blue" ? "A" : "B")),
    ),
    first,
    second,
    rift,
  };
}

function logistic(logit: number): number {
  if (logit >= 0) {
    const exponent = Math.exp(-logit);
    return 1 / (1 + exponent);
  }
  const exponent = Math.exp(logit);
  return exponent / (1 + exponent);
}

function scoreLogit(
  runtime: EffectiveRuntime,
  values: Record<string, number>,
): number {
  let score = runtime.intercept;
  for (const feature of runtime.features) {
    const raw = values[feature.name] ?? 0;
    const clamped = Math.max(feature.clipLow, Math.min(feature.clipHigh, raw));
    score += clamped * feature.weight;
  }
  return score;
}

function directFeatureName(
  entry: ChampionCatalogEntry | undefined,
  element: ElementId,
  enabledFeatureNames: ReadonlySet<string> | undefined,
): string | null {
  const evidence = entry?.elementEvidence?.[element];
  const eligible =
    evidence?.championEligible ??
    evidence?.eligible ??
    evidence?.directEligible ??
    false;
  const provenance =
    evidence?.vocabularyProvenance ?? evidence?.provenance ?? null;
  if (
    !evidence ||
    evidence.source !== "direct-residual" ||
    !eligible ||
    evidence.status !== "ready" ||
    (provenance !== "publication-audit-vocabulary" &&
      provenance !== "post-audit-full-refit") ||
    !evidence.featureName ||
    !enabledFeatureNames?.has(evidence.featureName)
  ) {
    return null;
  }
  return evidence.featureName;
}

export function enabledChampionResidualFeatures(
  model: ExplorerModel,
): ReadonlySet<string> {
  const residual = model.models.jointState.championResidual;
  const publicationAudit =
    residual?.diagnostics?.publicationExpansionAudit;
  const publicationVocabulary = residual?.vocabularies?.publication;
  if (
    !CHAMPION_RESIDUAL_SCHEMA_VERSIONS.has(model.schemaVersion) ||
    residual?.status !== "ready" ||
    !Number.isFinite(residual.familyGate) ||
    residual.familyGate <= 0 ||
    publicationAudit?.status !== "ready" ||
    !publicationVocabulary ||
    !Number.isFinite(publicationVocabulary.cells) ||
    publicationVocabulary.cells <= 0 ||
    !Number.isFinite(publicationVocabulary.champions) ||
    publicationVocabulary.champions <= 0 ||
    !Number.isFinite(publicationVocabulary.degreesOfFreedom) ||
    publicationVocabulary.degreesOfFreedom <= 0 ||
    publicationVocabulary.frozenFrom !==
      "full-cohort-after-family-audits"
  ) {
    return new Set();
  }
  const runtimeNames = new Set(
    model.models.jointState.runtime.features.map((feature) => feature.name),
  );
  const residualRuntimeNames = new Set(
    model.models.jointState.runtime.features
      .filter((feature) =>
        feature.name.startsWith("champion_direct_inventory::"),
      )
      .map((feature) => feature.name),
  );
  if (residualRuntimeNames.size !== publicationVocabulary.cells) {
    return new Set();
  }
  const enabled = new Set<string>();
  for (const champion of model.championCatalog) {
    for (const element of ELEMENTS) {
      const featureName = directFeatureName(champion, element, runtimeNames);
      if (featureName) enabled.add(featureName);
    }
  }
  if (
    enabled.size !== publicationVocabulary.cells ||
    [...enabled].some((featureName) => !residualRuntimeNames.has(featureName))
  ) {
    return new Set();
  }
  return enabled;
}

function traitCounts(
  champions: string[],
  catalog: Map<string, ChampionCatalogEntry>,
  activeMask = (1 << champions.length) - 1,
): Record<string, number> {
  const counts: Record<string, number> = {};
  champions.forEach((champion, index) => {
    if ((activeMask & (1 << index)) === 0) return;
    for (const tag of catalog.get(champion)?.tags ?? []) {
      counts[tag] = (counts[tag] ?? 0) + 1;
    }
  });
  return counts;
}

function baseFeatures(
  ownTraits: Record<string, number>,
  oppTraits: Record<string, number>,
  archetypes: string[],
  minute: number,
): Record<string, number> {
  const values: Record<string, number> = {};
  for (const tag of archetypes) {
    const difference = (ownTraits[tag] ?? 0) - (oppTraits[tag] ?? 0);
    values[`trait_diff_${tag}`] = difference;
    values[`trait_diff_${tag}_x_minute`] = difference * minute;
  }
  return values;
}

function addInventoryFeatures(
  values: Record<string, number>,
  prefix: "pre" | "post",
  ownInventory: Inventory,
  oppInventory: Inventory,
  ownTraits: Record<string, number>,
  oppTraits: Record<string, number>,
  archetypes: string[],
  minute: number,
  ownMainScale = 1,
  oppMainScale = 1,
): void {
  for (const element of ELEMENTS) {
    const own = ownInventory[element];
    const opp = oppInventory[element];
    const difference = own * ownMainScale - opp * oppMainScale;
    values[`${prefix}_inventory_diff_${element}`] = difference;
    values[`${prefix}_inventory_diff_${element}_x_minute`] = difference * minute;
    for (const tag of archetypes) {
      const ownTag = ownTraits[tag] ?? 0;
      const oppTag = oppTraits[tag] ?? 0;
      values[`${prefix}_${element}_own_trait_${tag}`] =
        own * ownTag - opp * oppTag;
      values[`${prefix}_${element}_enemy_trait_${tag}`] =
        own * oppTag - opp * ownTag;
    }
  }
}

function addChampionInventoryResidualFeatures(
  values: Record<string, number>,
  ownChampions: string[],
  oppChampions: string[],
  ownInventory: Inventory,
  oppInventory: Inventory,
  catalog: Map<string, ChampionCatalogEntry>,
  enabledFeatureNames: ReadonlySet<string> | undefined,
  ownMask: number,
  oppMask: number,
): void {
  ownChampions.forEach((champion, index) => {
    if ((ownMask & (1 << index)) === 0) return;
    const entry = catalog.get(champion);
    for (const element of ELEMENTS) {
      const featureName = directFeatureName(
        entry,
        element,
        enabledFeatureNames,
      );
      if (!featureName) continue;
      values[featureName] =
        (values[featureName] ?? 0) + ownInventory[element];
    }
  });
  oppChampions.forEach((champion, index) => {
    if ((oppMask & (1 << index)) === 0) return;
    const entry = catalog.get(champion);
    for (const element of ELEMENTS) {
      const featureName = directFeatureName(
        entry,
        element,
        enabledFeatureNames,
      );
      if (!featureName) continue;
      values[featureName] =
        (values[featureName] ?? 0) - oppInventory[element];
    }
  });
}

function addSoulFeatures(
  values: Record<string, number>,
  ownSoul: ElementId | null,
  oppSoul: ElementId | null,
  ownTraits: Record<string, number>,
  oppTraits: Record<string, number>,
  archetypes: string[],
  minute: number,
  ownMainScale = 1,
  oppMainScale = 1,
): void {
  for (const element of ELEMENTS) {
    const own = ownSoul === element ? 1 : 0;
    const opp = oppSoul === element ? 1 : 0;
    const difference = own * ownMainScale - opp * oppMainScale;
    values[`soul_after_${element}`] = difference;
    values[`soul_after_${element}_x_minute`] = difference * minute;
    for (const tag of archetypes) {
      const ownTag = ownTraits[tag] ?? 0;
      const oppTag = oppTraits[tag] ?? 0;
      values[`soul_after_${element}_own_trait_${tag}`] =
        own * ownTag - opp * oppTag;
      values[`soul_after_${element}_enemy_trait_${tag}`] =
        own * oppTag - opp * ownTag;
    }
  }
}

type ScoringContext = {
  teamA: string[];
  teamB: string[];
  catalog: Map<string, ChampionCatalogEntry>;
  archetypes: string[];
  minute: number;
  state: TimelineState;
  /** Team A minus Team B incremental gold, in thousands. Defaults to neutral. */
  goldDiffK?: number;
  inventoryTraitMaskA?: number;
  inventoryTraitMaskB?: number;
  directResidualFeatureNames?: ReadonlySet<string>;
};

function perspectiveTraits(
  context: ScoringContext,
  inventorySpecific = false,
): {
  traitsA: Record<string, number>;
  traitsB: Record<string, number>;
} {
  return {
    traitsA: traitCounts(
      context.teamA,
      context.catalog,
      inventorySpecific ? (context.inventoryTraitMaskA ?? 31) : 31,
    ),
    traitsB: traitCounts(
      context.teamB,
      context.catalog,
      inventorySpecific ? (context.inventoryTraitMaskB ?? 31) : 31,
    ),
  };
}

function jointPerspectiveFeatures(
  context: ScoringContext,
  perspective: TeamSide,
): Record<string, number> {
  const baseTraits = perspectiveTraits(context);
  const inventoryTraits = perspectiveTraits(context, true);
  const ownBaseTraits =
    perspective === "A" ? baseTraits.traitsA : baseTraits.traitsB;
  const oppBaseTraits =
    perspective === "A" ? baseTraits.traitsB : baseTraits.traitsA;
  const ownInventoryTraits =
    perspective === "A" ? inventoryTraits.traitsA : inventoryTraits.traitsB;
  const oppInventoryTraits =
    perspective === "A" ? inventoryTraits.traitsB : inventoryTraits.traitsA;
  const ownInventory =
    perspective === "A" ? context.state.inventoryA : context.state.inventoryB;
  const oppInventory =
    perspective === "A" ? context.state.inventoryB : context.state.inventoryA;
  const ownChampions =
    perspective === "A" ? context.teamA : context.teamB;
  const oppChampions =
    perspective === "A" ? context.teamB : context.teamA;
  const ownSoul = perspective === "A" ? context.state.soulA : context.state.soulB;
  const oppSoul = perspective === "A" ? context.state.soulB : context.state.soulA;
  const values = baseFeatures(
    ownBaseTraits,
    oppBaseTraits,
    context.archetypes,
    context.minute,
  );
  const goldDiffK = context.goldDiffK ?? 0;
  values.gold_diff_k = perspective === "A" ? goldDiffK : -goldDiffK;
  addInventoryFeatures(
    values,
    "post",
    ownInventory,
    oppInventory,
    ownInventoryTraits,
    oppInventoryTraits,
    context.archetypes,
    context.minute,
  );
  addChampionInventoryResidualFeatures(
    values,
    ownChampions,
    oppChampions,
    ownInventory,
    oppInventory,
    context.catalog,
    context.directResidualFeatureNames,
    31,
    31,
  );
  addSoulFeatures(
    values,
    ownSoul,
    oppSoul,
    ownInventoryTraits,
    oppInventoryTraits,
    context.archetypes,
    context.minute,
  );
  return values;
}

export function reconciledJointProbabilityA(
  runtime: EffectiveRuntime,
  context: ScoringContext,
): number {
  const logitA = scoreLogit(runtime, jointPerspectiveFeatures(context, "A"));
  const logitB = scoreLogit(runtime, jointPerspectiveFeatures(context, "B"));
  return logistic(0.5 * (logitA - logitB));
}

function allocationPerspectiveFeatures(
  context: ScoringContext,
  preState: TimelineState,
  currentElement: ElementId,
  recipient: TeamSide,
  perspective: TeamSide,
): Record<string, number> {
  const { traitsA, traitsB } = perspectiveTraits(context);
  const ownTraits = perspective === "A" ? traitsA : traitsB;
  const oppTraits = perspective === "A" ? traitsB : traitsA;
  const ownPre =
    perspective === "A" ? preState.inventoryA : preState.inventoryB;
  const oppPre =
    perspective === "A" ? preState.inventoryB : preState.inventoryA;
  const ownPost =
    perspective === "A" ? context.state.inventoryA : context.state.inventoryB;
  const oppPost =
    perspective === "A" ? context.state.inventoryB : context.state.inventoryA;
  const ownSoul = perspective === "A" ? context.state.soulA : context.state.soulB;
  const oppSoul = perspective === "A" ? context.state.soulB : context.state.soulA;
  const took = recipient === perspective ? 1 : 0;
  const direction = took * 2 - 1;
  const values = baseFeatures(
    ownTraits,
    oppTraits,
    context.archetypes,
    context.minute,
  );
  addInventoryFeatures(
    values,
    "pre",
    ownPre,
    oppPre,
    ownTraits,
    oppTraits,
    context.archetypes,
    context.minute,
  );
  addInventoryFeatures(
    values,
    "post",
    ownPost,
    oppPost,
    ownTraits,
    oppTraits,
    context.archetypes,
    context.minute,
  );
  addSoulFeatures(
    values,
    ownSoul,
    oppSoul,
    ownTraits,
    oppTraits,
    context.archetypes,
    context.minute,
  );
  values[`allocation_${currentElement}`] = direction;
  values[`allocation_${currentElement}_x_minute`] = direction * context.minute;
  for (const tag of context.archetypes) {
    const ownTag = ownTraits[tag] ?? 0;
    const oppTag = oppTraits[tag] ?? 0;
    values[`allocation_${currentElement}_own_trait_${tag}`] =
      took * ownTag - (1 - took) * oppTag;
    values[`allocation_${currentElement}_enemy_trait_${tag}`] =
      took * oppTag - (1 - took) * ownTag;
  }
  return values;
}

function stateAfterAllocation(
  preState: TimelineState,
  element: ElementId,
  recipient: TeamSide,
): TimelineState {
  const inventoryA = { ...preState.inventoryA };
  const inventoryB = { ...preState.inventoryB };
  const inventory = recipient === "A" ? inventoryA : inventoryB;
  inventory[element] += 1;
  const total = ELEMENTS.reduce((sum, id) => sum + inventory[id], 0);
  return {
    stage: preState.stage + 1,
    inventoryA,
    inventoryB,
    soulA: recipient === "A" && total === 4 ? element : null,
    soulB: recipient === "B" && total === 4 ? element : null,
  };
}

function allocationProbabilityA(
  runtime: EffectiveRuntime,
  context: Omit<ScoringContext, "state">,
  preState: TimelineState,
  element: ElementId,
  recipient: TeamSide,
): number {
  const state = stateAfterAllocation(preState, element, recipient);
  const fullContext: ScoringContext = { ...context, state };
  const logitA = scoreLogit(
    runtime,
    allocationPerspectiveFeatures(
      fullContext,
      preState,
      element,
      recipient,
      "A",
    ),
  );
  const logitB = scoreLogit(
    runtime,
    allocationPerspectiveFeatures(
      fullContext,
      preState,
      element,
      recipient,
      "B",
    ),
  );
  return logistic(0.5 * (logitA - logitB));
}

function stageReference(
  model: ExplorerModel,
  stage: number,
): { minute: number; rows: number } {
  const reference = model.stageReference.find((item) => item.stage === stage);
  return {
    minute: (reference?.medianSeconds ?? 0) / 60,
    rows: (reference?.perspectiveRows ?? 0) / 2,
  };
}

function baselineState(stage: number): TimelineState {
  return {
    stage,
    inventoryA: EMPTY_INVENTORY(),
    inventoryB: EMPTY_INVENTORY(),
    soulA: null,
    soulB: null,
  };
}

function popcount(value: number): number {
  let count = 0;
  let remaining = value;
  while (remaining) {
    count += remaining & 1;
    remaining >>>= 1;
  }
  return count;
}

const COMBINATIONS_4 = [1, 4, 6, 4, 1];

function shapleyFive(value: (mask: number) => number): number[] {
  const allocation = Array.from({ length: 5 }, () => 0);
  for (let champion = 0; champion < 5; champion += 1) {
    for (let subset = 0; subset < 32; subset += 1) {
      const bit = 1 << champion;
      if (subset & bit) continue;
      const size = popcount(subset);
      const weight = 1 / (5 * COMBINATIONS_4[size]);
      allocation[champion] += weight * (value(subset | bit) - value(subset));
    }
  }
  return allocation;
}

export function summarizeChampionEvidence(
  entry: ChampionCatalogEntry | undefined,
  inventory: Inventory,
  enabledFeatureNames: ReadonlySet<string>,
): ChampionEvidenceSummary {
  const activeElements = ELEMENTS.filter((element) => inventory[element] > 0);
  const informedElements = activeElements.filter((element) =>
    Boolean(directFeatureName(entry, element, enabledFeatureNames)),
  );
  const activeEvidence = activeElements.flatMap((element) => {
    const evidence = entry?.elementEvidence?.[element];
    return evidence ? [evidence] : [];
  });
  const support = activeElements.flatMap((element) => {
    const evidence = entry?.elementEvidence?.[element];
    return evidence &&
      Number.isFinite(evidence.trainingGames) &&
      Number.isFinite(evidence.trainingSeries)
      ? [{ element, evidence }]
      : [];
  });
  const leastSupported = support.length
    ? support.reduce((least, candidate) =>
        candidate.evidence.trainingGames < least.evidence.trainingGames
          ? candidate
          : least,
      )
    : null;
  const provenance = informedElements.flatMap((element) => {
    const evidence = entry?.elementEvidence?.[element];
    const value =
      evidence?.vocabularyProvenance ?? evidence?.provenance ?? null;
    return value ? [value] : [];
  });
  const source: ChampionDifferentialSource =
    activeElements.length === 0
      ? "unsupported"
      : informedElements.length === activeElements.length
        ? "champion-informed"
        : informedElements.length > 0
          ? "mixed"
          : entry?.tags.length
            ? "archetype-prior-only"
            : "unsupported";
  return {
    source,
    activeElements,
    informedElements,
    activeEvidence,
    leastSupportedElement: leastSupported?.element ?? null,
    minimumTrainingGames: leastSupported?.evidence.trainingGames ?? null,
    minimumTrainingSeries: leastSupported?.evidence.trainingSeries ?? null,
    minimumOwnershipGames:
      leastSupported?.evidence.ownershipGames ?? null,
    minimumNonOwnershipGames:
      leastSupported?.evidence.nonOwnershipGames ?? null,
    failedExposureRules: Array.from(
      new Set(
        activeEvidence.flatMap(
          (evidence) =>
            evidence.failedExposureRules ?? evidence.failedRules ?? [],
        ),
      ),
    ),
    vocabularyProvenance: provenance.includes("post-audit-full-refit")
      ? "post-audit-full-refit"
      : provenance.includes("publication-audit-vocabulary")
        ? "publication-audit-vocabulary"
        : null,
  };
}

function championDifferentialSource(
  entry: ChampionCatalogEntry | undefined,
  inventory: Inventory,
  enabledFeatureNames: ReadonlySet<string>,
): ChampionDifferentialSource {
  return summarizeChampionEvidence(entry, inventory, enabledFeatureNames).source;
}

export function buildCurve(
  model: ExplorerModel,
  teamA: string[],
  teamB: string[],
  captures: Capture[],
  focus: TeamSide,
): CurvePoint[] {
  const runtime = model.models.jointState.runtime;
  const catalog = new Map(
    model.championCatalog.map((champion) => [champion.name, champion]),
  );
  const directResidualFeatureNames = enabledChampionResidualFeatures(model);
  const states = timelineStates(captures);
  return states.map((state) => {
    if (state.stage === 0) {
      return {
        stage: 0,
        minute: 0,
        supportRows: 0,
        supportStatus: "baseline",
        teamADeltaPp: 0,
        teamBDeltaPp: 0,
        championCumulativePp: [0, 0, 0, 0, 0],
        championDifferentialSource: [
          "unsupported",
          "unsupported",
          "unsupported",
          "unsupported",
          "unsupported",
        ],
        teamContextPp: 0,
        soul: null,
      };
    }
    const reference = stageReference(model, state.stage);
    const common = {
      teamA,
      teamB,
      catalog,
      archetypes: model.featureSchema.archetypes,
      minute: reference.minute,
      directResidualFeatureNames,
    };
    const probability = reconciledJointProbabilityA(runtime, {
      ...common,
      state,
    });
    const baseline = reconciledJointProbabilityA(runtime, {
      ...common,
      state: baselineState(state.stage),
    });
    const teamADeltaPp = (probability - baseline) * 100;
    const focusedTotal = focus === "A" ? teamADeltaPp : -teamADeltaPp;
    const focusedChampions = focus === "A" ? teamA : teamB;
    const focusedInventory =
      focus === "A" ? state.inventoryA : state.inventoryB;
    const championDifferentialSources = focusedChampions.map((champion) =>
      championDifferentialSource(
        catalog.get(champion),
        focusedInventory,
        directResidualFeatureNames,
      ),
    );
    const focusedFeatureSets = focusedChampions.map((champion) => {
      const entry = catalog.get(champion);
      return new Set(
        ELEMENTS.flatMap((element) => {
          const featureName = directFeatureName(
            entry,
            element,
            directResidualFeatureNames,
          );
          return featureName ? [featureName] : [];
        }),
      );
    });
    const fixedDirectFeatures = new Set(directResidualFeatureNames);
    for (const featureSet of focusedFeatureSets) {
      for (const featureName of featureSet) {
        fixedDirectFeatures.delete(featureName);
      }
    }
    const directCache = new Map<number, number>();
    const directValue = (mask: number) => {
      const existing = directCache.get(mask);
      if (existing !== undefined) return existing;
      const enabledFeatures = new Set(fixedDirectFeatures);
      focusedFeatureSets.forEach((featureSet, index) => {
        if ((mask & (1 << index)) === 0) return;
        for (const featureName of featureSet) enabledFeatures.add(featureName);
      });
      const pathProbabilityA = reconciledJointProbabilityA(runtime, {
        ...common,
        state,
        directResidualFeatureNames: enabledFeatures,
      });
      const baselineProbabilityA = reconciledJointProbabilityA(runtime, {
        ...common,
        state: baselineState(state.stage),
        directResidualFeatureNames: enabledFeatures,
      });
      const deltaA = (pathProbabilityA - baselineProbabilityA) * 100;
      const result = focus === "A" ? deltaA : -deltaA;
      directCache.set(mask, result);
      return result;
    };
    const championResidualPp = shapleyFive(directValue);

    const archetypeCache = new Map<number, number>();
    const archetypeValue = (mask: number) => {
      const existing = archetypeCache.get(mask);
      if (existing !== undefined) return existing;
      const traitMasks =
        focus === "A"
          ? {
              inventoryTraitMaskA: mask,
              inventoryTraitMaskB: 31,
            }
          : {
              inventoryTraitMaskA: 31,
              inventoryTraitMaskB: mask,
            };
      const pathProbabilityA = reconciledJointProbabilityA(runtime, {
        ...common,
        ...traitMasks,
        state,
      });
      const baselineProbabilityA = reconciledJointProbabilityA(runtime, {
        ...common,
        ...traitMasks,
        state: baselineState(state.stage),
      });
      const deltaA = (pathProbabilityA - baselineProbabilityA) * 100;
      const result = focus === "A" ? deltaA : -deltaA;
      archetypeCache.set(mask, result);
      return result;
    };
    const archetypePriorPp = shapleyFive(archetypeValue);
    const championCumulativePp = championResidualPp.map(
      (contribution, index) => {
        const source = championDifferentialSources[index];
        if (source === "unsupported") return 0;
        if (source === "archetype-prior-only") return archetypePriorPp[index];
        return contribution;
      },
    );
    const teamContextPp =
      focusedTotal -
      championCumulativePp.reduce(
        (sum, contribution) => sum + contribution,
        0,
      );
    return {
      stage: state.stage,
      minute: reference.minute,
      supportRows: reference.rows,
      supportStatus: reference.rows < 500 ? "low-support" : "supported",
      teamADeltaPp,
      teamBDeltaPp: -teamADeltaPp,
      championCumulativePp,
      championDifferentialSource: championDifferentialSources,
      teamContextPp,
      soul: state.soulA
        ? { team: "A", element: state.soulA }
        : state.soulB
          ? { team: "B", element: state.soulB }
          : null,
    };
  });
}

export function buildAllocationEstimate(
  model: ExplorerModel,
  teamA: string[],
  teamB: string[],
  captures: Capture[],
): AllocationEstimate | null {
  if (!captures.length) return null;
  const stage = captures.length;
  const preState = timelineStates(captures.slice(0, -1)).at(-1);
  if (!preState) return null;
  const element = captures.at(-1)?.element;
  if (!element) return null;
  const reference = stageReference(model, stage);
  const context = {
    teamA,
    teamB,
    catalog: new Map(
      model.championCatalog.map((champion) => [champion.name, champion]),
    ),
    archetypes: model.featureSchema.archetypes,
    minute: reference.minute,
  };
  const probabilityIfA = allocationProbabilityA(
    model.models.captureAllocation.runtime,
    context,
    preState,
    element,
    "A",
  );
  const probabilityIfB = allocationProbabilityA(
    model.models.captureAllocation.runtime,
    context,
    preState,
    element,
    "B",
  );
  return {
    element,
    stage,
    teamADeltaPp: (probabilityIfA - probabilityIfB) * 100,
    probabilityIfA,
    probabilityIfB,
  };
}

/**
 * Compare a selected elemental capture with conceding that same capture while
 * gaining cross-map gold elsewhere. This is a neutral-state, associational
 * gold-equivalent from the joint-state runtime, not a causal objective policy.
 */
export function buildTakeLeaveEstimate(
  model: ExplorerModel,
  teamA: string[],
  teamB: string[],
  captures: Capture[],
  stage: number,
  focus: TeamSide,
  crossMapGold: number,
): TakeLeaveEstimate | null {
  if (!Number.isInteger(stage) || stage < 1 || stage > captures.length) {
    return null;
  }
  const selectedCapture = captures[stage - 1];
  const preState = timelineStates(captures.slice(0, stage - 1)).at(-1);
  if (
    !selectedCapture ||
    !preState ||
    preState.stage !== stage - 1 ||
    preState.soulA ||
    preState.soulB
  ) {
    return null;
  }

  const opponent: TeamSide = focus === "A" ? "B" : "A";
  const takeState = stateAfterAllocation(
    preState,
    selectedCapture.element,
    focus,
  );
  const leaveState = stateAfterAllocation(
    preState,
    selectedCapture.element,
    opponent,
  );
  const reference = stageReference(model, stage);
  const runtime = model.models.jointState.runtime;
  const common = {
    teamA,
    teamB,
    catalog: new Map(
      model.championCatalog.map((champion) => [champion.name, champion]),
    ),
    archetypes: model.featureSchema.archetypes,
    minute: reference.minute,
    directResidualFeatureNames: enabledChampionResidualFeatures(model),
  };
  const focusedProbability = (
    state: TimelineState,
    focusedGold: number,
  ): number => {
    const teamAGoldDiffK =
      ((focus === "A" ? 1 : -1) * Math.max(0, focusedGold)) / 1_000;
    const probabilityA = reconciledJointProbabilityA(runtime, {
      ...common,
      state,
      goldDiffK: teamAGoldDiffK,
    });
    return focus === "A" ? probabilityA : 1 - probabilityA;
  };

  const requestedCrossMapGold = Number.isFinite(crossMapGold)
    ? Math.max(0, crossMapGold)
    : 0;
  const takeProbability = focusedProbability(takeState, 0);
  const leaveProbability = focusedProbability(
    leaveState,
    requestedCrossMapGold,
  );
  const differencePp = (takeProbability - leaveProbability) * 100;
  const goldFeature = runtime.features.find(
    (feature) => feature.name === "gold_diff_k",
  );
  const positiveObservedGoldK = goldFeature
    ? Math.max(
        0,
        Math.min(goldFeature.observedMax, -goldFeature.observedMin),
      )
    : 0;
  const maxCrossMapGold = Math.floor(positiveObservedGoldK * 1_000);

  let status: TakeLeaveEstimate["status"] = "gold-support-unavailable";
  let breakevenGold: number | null = null;
  if (goldFeature && maxCrossMapGold > 0) {
    const differenceAtZero =
      takeProbability - focusedProbability(leaveState, 0);
    const probabilityTolerance = 0.00005;
    if (Math.abs(differenceAtZero) <= probabilityTolerance) {
      status = "breakeven-within-support";
      breakevenGold = 0;
    } else if (differenceAtZero < 0) {
      status = "leave-favored-at-zero";
      breakevenGold = 0;
    } else {
      const differenceAtMaximum =
        takeProbability -
        focusedProbability(leaveState, maxCrossMapGold);
      if (differenceAtMaximum > 0) {
        status = "no-crossover-within-support";
      } else {
        let low = 0;
        let high = maxCrossMapGold;
        for (let iteration = 0; iteration < 48; iteration += 1) {
          const midpoint = (low + high) / 2;
          const difference =
            takeProbability - focusedProbability(leaveState, midpoint);
          if (difference > 0) low = midpoint;
          else high = midpoint;
        }
        status = "breakeven-within-support";
        breakevenGold = Math.round((low + high) / 2);
      }
    }
  }

  const verdict: TakeLeaveEstimate["verdict"] =
    requestedCrossMapGold > maxCrossMapGold
      ? "outside-supported-gold-range"
      : Math.abs(differencePp) < 0.005
        ? "effectively-even"
        : differencePp > 0
          ? "take-favored"
          : "leave-favored";

  return {
    status,
    element: selectedCapture.element,
    stage,
    minute: reference.minute,
    focus,
    takeProbability,
    leaveProbability,
    differencePp,
    breakevenGold,
    maxCrossMapGold,
    verdict,
    supportRows: reference.rows,
    supportStatus: reference.rows < 500 ? "low-support" : "supported",
  };
}

export function buildElementRankings(
  model: ExplorerModel,
  teamA: string[],
  teamB: string[],
  focus: TeamSide,
): ElementRanking[] {
  const runtime = model.models.jointState.runtime;
  const catalog = new Map(
    model.championCatalog.map((champion) => [champion.name, champion]),
  );
  const common = {
    teamA,
    teamB,
    catalog,
    archetypes: model.featureSchema.archetypes,
    directResidualFeatureNames: enabledChampionResidualFeatures(model),
  };
  const focusedDelta = (state: TimelineState, minute: number) => {
    const probability = reconciledJointProbabilityA(runtime, {
      ...common,
      minute,
      state,
    });
    const baseline = reconciledJointProbabilityA(runtime, {
      ...common,
      minute,
      state: baselineState(state.stage),
    });
    const deltaA = (probability - baseline) * 100;
    return focus === "A" ? deltaA : -deltaA;
  };
  const stageOne = stageReference(model, 1);
  const stageFour = stageReference(model, 4);
  const rankings = ELEMENTS.map((element) => {
    const firstInventoryA = EMPTY_INVENTORY();
    const firstInventoryB = EMPTY_INVENTORY();
    (focus === "A" ? firstInventoryA : firstInventoryB)[element] = 1;
    const firstCapturePp = focusedDelta(
      {
        stage: 1,
        inventoryA: firstInventoryA,
        inventoryB: firstInventoryB,
        soulA: null,
        soulB: null,
      },
      stageOne.minute,
    );
    const openings = ELEMENTS.filter((candidate) => candidate !== element);
    const soulEffects: number[] = [];
    for (let left = 0; left < openings.length; left += 1) {
      for (let right = left + 1; right < openings.length; right += 1) {
        const inventoryA = EMPTY_INVENTORY();
        const inventoryB = EMPTY_INVENTORY();
        const inventory = focus === "A" ? inventoryA : inventoryB;
        inventory[openings[left]] = 1;
        inventory[openings[right]] = 1;
        inventory[element] = 2;
        soulEffects.push(
          focusedDelta(
            {
              stage: 4,
              inventoryA,
              inventoryB,
              soulA: focus === "A" ? element : null,
              soulB: focus === "B" ? element : null,
            },
            stageFour.minute,
          ),
        );
      }
    }
    return {
      element,
      firstCapturePp,
      perfectControlSoulPp:
        soulEffects.reduce((sum, value) => sum + value, 0) / soulEffects.length,
    };
  });
  return rankings.sort(
    (left, right) =>
      right.perfectControlSoulPp - left.perfectControlSoulPp ||
      right.firstCapturePp - left.firstCapturePp,
  );
}

export function higherConversionCandidates(
  mechanic: Mechanic,
  champions: string[],
  catalog: Map<string, ChampionCatalogEntry>,
): Array<{ champion: string; tags: string[] }> {
  return champions
    .map((champion) => ({
      champion,
      tags: (catalog.get(champion)?.tags ?? []).filter((tag) =>
        mechanic.directTags.includes(tag),
      ),
    }))
    .filter((candidate) => candidate.tags.length > 0);
}
