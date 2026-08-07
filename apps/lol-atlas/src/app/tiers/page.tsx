import Link from "next/link";
import { TierListExplorer } from "@/components/TierListExplorer";
import styles from "./TiersPage.module.css";

export const metadata = {
  title: "Role tier lists — Scryglass",
  description:
    "Development-only role x league x current-patch champion tier lists with region, international, and competition-tier filters.",
};

export default function TiersPage() {
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Role tier lists</h1>
          <p>
            Development-only tier lists: incremental composition value per role
            from the frozen terminal Draft Score candidate, played-only
            membership, and descriptive atom-based counterability. Rank
            eligibility is off; these are not outcome-calibrated rankings.
          </p>
        </div>
        <div className={styles.provenance}>
          <span>development-only</span>
          <span>no rank eligibility</span>
          <span>counterability weight 0</span>
        </div>
      </header>
      <TierListExplorer />
      <footer className={styles.footer}>
        <p>
          Method:{" "}
          <Link href="/methodology">methodology</Link> ·{" "}
          <Link href="/reproduce">reproduce</Link> · Tier Value =
          champion-role composition logit from the frozen development
          candidate (equal-strength composition index, not a win
          probability). Played-only means a champion must have verified
          appearances in the exact league/event, patch, and role.
        </p>
      </footer>
    </main>
  );
}
