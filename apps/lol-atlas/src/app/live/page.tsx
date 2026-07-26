import { LiveBoard } from "@/components/LiveBoard";
import { liveIndexUrl, readLiveIndex, readLiveSnapshots } from "@/lib/liveServer";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Live game state — Scryglass",
  description: "Verified live League of Legends game state and conditional model estimate.",
};

export default async function LivePage() {
  const index = await readLiveIndex();
  const snapshots = await readLiveSnapshots(index);
  return <LiveBoard initialIndex={index} initialSnapshots={snapshots} liveIndexUrl={liveIndexUrl()} />;
}
