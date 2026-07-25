"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/elo", label: "Elo" },
  { href: "/grubs", label: "Grubs" },
  { href: "/browse", label: "Browse" },
  { href: "/browse/head-to-head", label: "H2H" },
  { href: "/reproduce", label: "Reproduce" },
  { href: "/methodology", label: "Method" },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden />
          <span className="brand-name">Scryglass</span>
        </Link>
        <nav className="site-nav" aria-label="Primary">
          {LINKS.map((l) => {
            const current =
              l.href === "/"
                ? pathname === "/"
                : l.href === "/browse"
                  ? pathname === "/browse" || pathname.startsWith("/browse/match")
                  : pathname === l.href || pathname.startsWith(`${l.href}/`);
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={current ? "page" : undefined}
                className="nav-link"
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <ThemeToggle />
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      Map rows from Oracle&apos;s Elixir · Dual Elo &amp; calibration are research estimates · Pack{" "}
      <span className="font-mono">2025–2026</span>
    </footer>
  );
}
