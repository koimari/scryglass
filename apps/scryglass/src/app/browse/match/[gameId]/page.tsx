import { MatchLoader } from "@/components/MatchLoader";
import { OperationalHeader } from "@/components/OperationalHeader";
import { readPackManifest } from "@/lib/serverPack";

export const dynamic = "force-dynamic";

type Props = {
  params: Promise<{ gameId: string }>;
  searchParams: Promise<{ year?: string }>;
};

export default async function MatchPage({ params, searchParams }: Props) {
  const { gameId: raw } = await params;
  const sp = await searchParams;
  const gameId = decodeURIComponent(raw);
  const yearHint = sp.year ? Number(sp.year) : undefined;

  const man = await readPackManifest();
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="match-page space-y-4">
      <OperationalHeader
        title="Match"
        description="Game result, player rows, and validation status."
        meta={<span>{gameId}</span>}
      />
      <MatchLoader
        baseUrl={baseUrl}
        years={man.filters.years}
        gameId={gameId}
        yearHint={Number.isFinite(yearHint) ? yearHint : undefined}
      />
    </div>
  );
}
