import { Suspense } from "react";
import { HeadToHead } from "@/components/HeadToHead";
import { packUpdatedLabel } from "@/lib/pack";
import { readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

export default async function HeadToHeadPage() {
  const man = await readPackManifest();
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="space-y-6">
      <header className="page-header">
        <p className="blog-kicker">Matches · Head-to-head</p>
        <h1 className="font-display mt-2 text-3xl">Head-to-head</h1>
        <p className="lede">
          Series between two orgs, then the board. Model checklist lives on the match page.
        </p>
        <div className="micro-log mt-4">
          <span>
            <strong>Last updated</strong> {packUpdatedLabel(man)}
          </span>
        </div>
      </header>
      <Suspense fallback={<div className="skeleton-block" />}>
        <HeadToHead baseUrl={baseUrl} years={man.filters.years} />
      </Suspense>
    </div>
  );
}
