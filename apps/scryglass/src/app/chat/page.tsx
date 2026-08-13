import type { Metadata } from "next";
import SupportChat from "@/components/SupportChat";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "Chat — Scryglass",
  description: "Ask Scryglass about players, teams, matches, ratings, tier lists, schedules, and methodology.",
};

export default function ChatPage() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Ask Scryglass</h1>
        <p>
          Ask a question in plain language. Answers use the published Scryglass data.
        </p>
      </header>
      <SupportChat />
    </div>
  );
}
