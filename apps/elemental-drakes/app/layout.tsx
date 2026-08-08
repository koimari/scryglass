import type { Metadata } from "next";
import { Hanken_Grotesk } from "next/font/google";

import "./globals.css";

const hankenGrotesk = Hanken_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
  variable: "--font-hanken-grotesk",
});

export const metadata: Metadata = {
  title: "Elemental Dragon Value — Scryglass",
  description:
    "Compare how dragon sequences change estimated map-win probability for two lineups.",
  openGraph: {
    title: "What is each dragon worth?",
    description: "Associational model of 6,382 professional games.",
    type: "article",
  },
  twitter: {
    card: "summary",
    title: "What is each dragon worth?",
    description: "Compare dragon sequences for any two lineups.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html className={hankenGrotesk.variable} lang="en">
      <body>{children}</body>
    </html>
  );
}
