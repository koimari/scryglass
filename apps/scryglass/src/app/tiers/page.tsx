import Link from "next/link";
import { TierListExplorer } from "@/components/TierListExplorer";
import styles from "./TiersPage.module.css";

export const metadata = {
  title: "Role tier lists — Scryglass",
  description:
    "Patch-wide champion tier lists by role across professional play.",
};

export default function TiersPage() {
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1>Role tier lists</h1>
          <p>
            Each patch pools eligible professional games across regions,
            leagues, and tournaments. Every role has its own board.
          </p>
        </div>
        <div className={styles.provenance}>
          <span>all roles</span>
          <span>patch-wide pool</span>
          <span>cached display file</span>
        </div>
      </header>
      <TierListExplorer />
      <footer className={styles.footer}>
        <p>
          Method:{" "}
          <Link href="/methodology">Read the method</Link>. A champion must have
          verified appearances in the selected patch and role. Positive
          movement means that the champion climbed since the prior update.
        </p>
      </footer>
    </main>
  );
}
