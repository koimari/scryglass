import Link from "next/link";
import { TierListExplorer } from "@/components/TierListExplorer";
import styles from "./TiersPage.module.css";

export const metadata = {
  title: "Champion draft board — Scryglass",
  description:
    "Patch-wide champion strength, blind stability, counter reach, and regional context.",
};

export default function TiersPage() {
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Champion draft board</h1>
          <p>
            Start with the strongest first picks. Check blind stability, counter
            reach, and responses to a specific champion. The main board pools
            every accepted professional map in the patch.
          </p>
        </div>
        <div className={styles.provenance}>
          <span>patch-wide model</span>
          <span>role-aware</span>
          <span>OE source</span>
        </div>
      </header>
      <TierListExplorer />
      <footer className={styles.footer}>
        <p>
          Method:{" "}
          <Link href="/methodology">Read the method</Link>. A champion must have
          verified appearances in the selected patch and role. Regional context
          keeps the patch-wide fit fixed and changes the observed league pool.
          Matchup edges are descriptive model comparisons, not raw win rates or
          draft recommendations.
        </p>
      </footer>
    </main>
  );
}
