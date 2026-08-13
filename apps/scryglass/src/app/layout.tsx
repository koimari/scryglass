import type { Metadata } from "next";
import { headers } from "next/headers";
import {
  Atkinson_Hyperlegible_Mono,
  Atkinson_Hyperlegible_Next,
  Instrument_Serif,
} from "next/font/google";
import { SiteFooter, SiteHeader } from "@/components/SiteChrome";
import { SmoothScroll } from "@/components/SmoothScroll";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ThemeScript } from "@/components/ThemeScript";
import "lenis/dist/lenis.css";
import "./globals.css";
import SupportChat from "@/components/SupportChat";

const ui = Atkinson_Hyperlegible_Next({
  variable: "--font-ui",
  subsets: ["latin"],
  weight: "variable",
  adjustFontFallback: false,
});

const mono = Atkinson_Hyperlegible_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: "variable",
  adjustFontFallback: false,
});

const display = Instrument_Serif({
  variable: "--font-display",
  subsets: ["latin"],
  weight: "400",
});

export const metadata: Metadata = {
  title: "Scryglass — League of Legends ratings",
  description:
    "Quick team ratings, player ratings, and champion tier lists for professional League of Legends.",
  robots: { index: true, follow: true },
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const nonce = (await headers()).get("x-nonce") ?? undefined;
  return (
    <html
      lang="en"
      className={`${ui.variable} ${mono.variable} ${display.variable}`}
      suppressHydrationWarning
    >
      <head>
        <ThemeScript nonce={nonce} />
      </head>
      <body>
        <ThemeProvider>
          <SmoothScroll />
          <SiteHeader />
          <main className="site-main">{children}</main>
          <SiteFooter />
          <SupportChat floating />
        </ThemeProvider>
      </body>
    </html>
  );
}
