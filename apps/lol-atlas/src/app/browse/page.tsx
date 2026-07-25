import Link from "next/link";
import { promises as fs } from "fs";
import path from "path";
import { BrowseMaps } from "@/components/BrowseMaps";
import type { PackManifest } from "@/lib/pack";

export default async function BrowsePage() {
  const man = JSON.parse(
    await fs.readFile(path.join(process.cwd(), "public", "packs", "manifest.json"), "utf8"),
  ) as PackManifest;
  const baseUrl = man.base_url || `/packs/${man.pack_id}`;

  return (
    <div className="space-y-6">
      <header className="page-header">
        <p className="blog-kicker">Warehouse · Maps</p>
        <h1 className="font-display mt-2 text-3xl">Browse maps</h1>
        <p className="lede">
          One row per OE map ({man.filters.years.join("–")}). Filter, then open a ticker board.{" "}
          <Link href="/browse/head-to-head" className="row-link">
            Head-to-head →
          </Link>
        </p>
      </header>
      <BrowseMaps baseUrl={baseUrl} years={man.filters.years} />
    </div>
  );
}
