import Link from "next/link";
import { TierListExplorer } from "@/components/TierListExplorer";
import styles from "./TiersPage.module.css";

export const metadata = {
  title: "Champion draft board — Scryglass",
  description:
    "Patch-wide champion strength, matchup shape, and unpicked structural alternatives.",
};

export default function TiersPage() {
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Champion draft board</h1>
          <p>
            Check what the patch rewards, how matchups change, and which unused
            champions can fill a similar job. Every performance board pools the
            accepted professional games in that patch.
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
          <Link href="/methodology">Read the method</Link>. Performance boards
          require verified appearances. Unpicked alternatives compare role and
          function profiles with played champions. They do not estimate hidden
          strength or recommend a draft pick.
        </p>
      </footer>
    </main>
  );
}
