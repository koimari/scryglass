import Link from "next/link";
import {
  TierListExplorer,
  type TierFilterState,
  type TierResponse,
} from "@/components/TierListExplorer";
import { getTierFacets, getTierScope, queryApiAvailable } from "@/lib/publicData";
import { publicPatchLabel, samePublicPatch } from "@/lib/patchIdentity";
import { readPackManifest } from "@/lib/serverPack";
import type { TierScope } from "@/lib/tierBoard";
import styles from "./TiersPage.module.css";

export const metadata = {
  title: "Tier Lists — Scryglass",
  description:
    "Patch-wide champion strength, matchup shape, and unpicked structural alternatives.",
};

type PageProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

function first(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function sourcePatchFor(value: string | undefined, options: string[]): string {
  if (!value) return "";
  return options.find((option) => samePublicPatch(option, value)) ?? "";
}

function oneOf(value: string | undefined, options: string[]): string {
  return value && options.includes(value) ? value : "";
}

function minimumGames(value: string | undefined): number {
  const parsed = Number.parseInt(value ?? "", 10);
  return [1, 3, 5, 10, 20].includes(parsed) ? parsed : 5;
}

function latestPatch(options: string[]): string {
  return [...options].sort((left, right) => {
    const [leftMajor, leftMinor] = publicPatchLabel(left).split(".").map(Number);
    const [rightMajor, rightMinor] = publicPatchLabel(right).split(".").map(Number);
    return rightMajor - leftMajor || rightMinor - leftMinor;
  })[0] ?? "";
}

export default async function TiersPage({ searchParams }: PageProps) {
  const query = await searchParams;
  const manifest = await readPackManifest();
  const boundedQueries = queryApiAvailable(manifest);
  let initialData: TierResponse | undefined;
  let initialFilters: TierFilterState | undefined;

  if (boundedQueries) {
    const facets = await getTierFacets(manifest);
    const sourcePatch = sourcePatchFor(first(query.patch), facets.options.patches) || latestPatch(facets.options.patches);
    const patch = publicPatchLabel(sourcePatch);
    const role = oneOf(first(query.role), facets.options.roles);
    const region = oneOf(first(query.region), facets.options.regions);
    const league = oneOf(first(query.league), facets.options.leagues);
    const tier = oneOf(first(query.tier), facets.options.tiers);
    const min = minimumGames(first(query.min));
    const scope = sourcePatch
      ? await getTierScope(manifest, { patch: sourcePatch, role, region, league, tier, similarityLimit: 100 })
      : null;
    const scopedRows = scope?.rows ?? [];
    const selectedScope = scope?.scope ?? null;
    const scopes: TierScope[] = selectedScope
      ? [selectedScope]
      : facets.scopes
          .filter((item) => samePublicPatch(item.patch, sourcePatch))
          .map((item) => ({
            scope_id: item.scope_id,
            scope_kind: "patch" as const,
            role: item.role,
            patch: item.patch,
            as_of: manifest.created_utc,
            status: "production" as const,
            row_count: item.row_count,
            regional_views: item.regions,
          }));
    initialData = {
      status: patch ? "available" : "unavailable",
      reason: patch ? undefined : "The current release has no published tier scope.",
      generated_at: manifest.created_utc,
      as_of: selectedScope?.as_of ?? manifest.created_utc,
      options: facets.options,
      scopes,
      rows: scopedRows,
      structural_similarity: role ? scope?.structural_similarity ?? undefined : undefined,
      champion_images: scope?.champion_images ?? {},
    };
    initialFilters = { patch, role, region, league, tier, minimumGames: min };
  }

  return (
    <div className={styles.page} data-scryglass-release={manifest.pack_id}>
      <header className={styles.header}>
        <div>
          <h1>Tier Lists</h1>
          <p>
            Check what the patch rewards, how matchups change, and which unused
            champions can fill a similar job. Every performance board pools the
            accepted professional games in that patch.
          </p>
        </div>
        <div className={styles.provenance}>
          <span>patch-wide model</span>
          <span>role-aware</span>
          <span>OE source</span>
        </div>
      </header>
      <TierListExplorer
        key={initialFilters ? `${initialFilters.patch}|${initialFilters.role}|${initialFilters.region}|${initialFilters.league}|${initialFilters.tier}|${initialFilters.minimumGames}` : "legacy"}
        initialData={initialData}
        initialFilters={initialFilters}
        serverFiltered={boundedQueries}
      />
      <footer className={styles.footer}>
        <p>
          Method:{" "}
          <Link href="/methodology">Read the method</Link>. Performance boards
          require verified appearances. Unpicked alternatives compare role and
          function profiles with played champions. They do not estimate hidden
          strength or recommend a draft pick.
        </p>
      </footer>
    </div>
  );
}
