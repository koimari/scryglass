import { Suspense } from "react";
import { HeadToHead } from "@/components/HeadToHead";
import { OperationalHeader } from "@/components/OperationalHeader";
import { packUpdatedLabel } from "@/lib/pack";
import { readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

export default async function HeadToHeadPage() {
  const man = await readPackManifest();
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="h2h-page space-y-6">
      <OperationalHeader
        title="Head-to-head"
        description="Compare completed series between two teams."
        meta={
          <span>
            <strong>Updated</strong> {packUpdatedLabel(man)}
          </span>
        }
      />
      <Suspense fallback={<div className="skeleton-block" />}>
        <HeadToHead baseUrl={baseUrl} years={man.filters.years} />
      </Suspense>
    </div>
  );
}
