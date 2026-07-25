import { promises as fs } from "fs";
import path from "path";
import { MatchLoader } from "@/components/MatchLoader";
import type { PackManifest } from "@/lib/pack";

type Props = {
  params: Promise<{ gameId: string }>;
  searchParams: Promise<{ year?: string }>;
};

export default async function MatchPage({ params, searchParams }: Props) {
  const { gameId: raw } = await params;
  const sp = await searchParams;
  const gameId = decodeURIComponent(raw);
  const yearHint = sp.year ? Number(sp.year) : undefined;

  const man = JSON.parse(
    await fs.readFile(path.join(process.cwd(), "public", "packs", "manifest.json"), "utf8"),
  ) as PackManifest;
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="space-y-4">
      <header className="page-header" style={{ marginBottom: "1rem" }}>
        <h1 className="text-2xl font-semibold tracking-tight">Match board</h1>
        <p className="lede" style={{ marginTop: "0.35rem" }}>
          <span className="font-mono text-sm">{gameId}</span>
        </p>
      </header>
      <MatchLoader
        baseUrl={baseUrl}
        years={man.filters.years}
        gameId={gameId}
        yearHint={Number.isFinite(yearHint) ? yearHint : undefined}
      />
    </div>
  );
}
