import { Suspense } from "react";
import { BrowseMatches } from "@/components/BrowseMaps";
import { OperationalHeader } from "@/components/OperationalHeader";
import { packUpdatedLabel } from "@/lib/pack";
import { readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

export default async function BrowsePage() {
  const man = await readPackManifest();
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="matches-page space-y-6">
      <OperationalHeader
        title="Matches"
        description="Completed series and games in the current data pack."
        meta={
          <>
            <span>
              <strong>Updated</strong> {packUpdatedLabel(man)}
            </span>
            <span>
              <strong>Pack</strong> {man.pack_id}
            </span>
          </>
        }
      />
      <Suspense fallback={<div className="skeleton-block" />}>
        <BrowseMatches baseUrl={baseUrl} years={man.filters.years} />
      </Suspense>
    </div>
  );
}
