import type { Metadata } from "next";
import SupportChat from "@/components/SupportChat";

export const metadata: Metadata = {
  title: "Support — Scryglass",
  description: "Ask about players, teams, matches, ratings, tier lists, schedules, and methodology.",
};

export default function SupportPage() {
  return (
    <main className="page" style={{ maxWidth: "52rem", margin: "0 auto", padding: "1.5rem 1rem" }}>
      <header>
        <p className="scope">Scryglass · Support</p>
        <h1>Ask Scryglass</h1>
        <p className="lede">
          Ask a question in plain language — the assistant routes it to the live public data
          and shows the real answer.
        </p>
      </header>
      <SupportChat />
    </main>
  );
}
