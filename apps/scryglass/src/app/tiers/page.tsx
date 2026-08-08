import Link from "next/link";
import { TierListExplorer } from "@/components/TierListExplorer";
import styles from "./TiersPage.module.css";

export const metadata = {
  title: "Role tier lists — Scryglass",
  description:
    "Production champion tier lists by role, league level, matchup shape, and weekly rank movement.",
};

export default function TiersPage() {
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Role tier lists</h1>
          <p>
            Every role has its own board. Blind tiers reward champions whose
            weak matchups remain strong. Counter tiers reward champions that
            beat the widest range of same-role picks after team-strength control.
          </p>
        </div>
        <div className={styles.provenance}>
          <span>all roles</span>
          <span>Tier 1 to Tier 3</span>
          <span>source-bound watermark</span>
        </div>
      </header>
      <TierListExplorer />
      <footer className={styles.footer}>
        <p>
          Method:{" "}
          <Link href="/methodology">methodology</Link> ·{" "}
          <Link href="/reproduce">reproduce</Link> · Tier Value =
          the champion-role value in the approved tier-list artifact. Played-only
          means a champion must have verified appearances in the exact
          league/event and role. The patch shown on each board is the latest
          source watermark. Positive rank movement means a climb
          from the previous artifact.
        </p>
      </footer>
    </main>
  );
}
