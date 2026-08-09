import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import { SiteFooter, SiteHeader } from "@/components/SiteChrome";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ThemeScript } from "@/components/ThemeScript";
import "./globals.css";

const ui = IBM_Plex_Sans({
  variable: "--font-ui",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Scryglass — LoL research by koi",
  description:
    "Independent League of Legends research publication: essays, Dual Elo ratings, and reproducible Oracle’s Elixir packs (2025–2026).",
  robots: { index: true, follow: true },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${ui.variable} ${mono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <ThemeScript />
      </head>
      <body className="min-h-full flex flex-col">
        <ThemeProvider>
          <SiteHeader />
          <main className="site-main">{children}</main>
          <SiteFooter />
        </ThemeProvider>
      </body>
    </html>
  );
}
