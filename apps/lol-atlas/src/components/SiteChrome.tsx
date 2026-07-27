"use client";

import { ArrowUpRightIcon, ListIcon, XIcon } from "@phosphor-icons/react";
import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ThemeToggle } from "./ThemeToggle";

const LINKS = [
  { href: "/articles", label: "Articles" },
  { href: "/elo", label: "Ratings" },
  { href: "/browse", label: "Matches" },
  { href: "/browse/head-to-head", label: "H2H" },
  { href: "/sandbox", label: "Sandbox" },
  { href: "/methodology", label: "Method" },
  { href: "/reproduce", label: "Reproduce" },
];

function isCurrent(pathname: string, href: string): boolean {
  if (href === "/articles") return pathname === href || pathname.startsWith("/articles/");
  if (href === "/browse") return pathname === href || pathname.startsWith("/browse/match");
  return pathname === href || pathname.startsWith(`${href}/`);
}

function relativeFreshness(raw: string | undefined): string | null {
  if (!raw) return null;
  const time = Date.parse(raw);
  if (!Number.isFinite(time)) return null;
  const minutes = Math.max(0, Math.round((Date.now() - time) / 60_000));
  if (minutes < 2) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function SiteHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [updated, setUpdated] = useState<string | null>(null);
  const latestPackRef = useRef<string | null>(null);
  const menuToggleRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      fetch("/api/pack-manifest", { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .then((manifest: { pack_id?: string; created_utc?: string } | null) => {
          if (cancelled) return;
          setUpdated(relativeFreshness(manifest?.created_utc));
          const nextPack = manifest?.pack_id ?? null;
          if (latestPackRef.current && nextPack && latestPackRef.current !== nextPack) {
            router.refresh();
          }
          latestPackRef.current = nextPack;
        })
        .catch(() => {
          if (!cancelled) setUpdated(null);
        });
    };
    refresh();
    const timer = window.setInterval(refresh, 60_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [router]);

  useEffect(() => {
    if (!menuOpen) return;
    const firstLink = menuRef.current?.querySelector<HTMLAnchorElement>("a");
    firstLink?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
        menuToggleRef.current?.focus();
      }
      if (event.key === "Tab" && menuRef.current) {
        const links = [...menuRef.current.querySelectorAll<HTMLAnchorElement>("a[href]")];
        if (!links.length) return;
        const first = links[0];
        const last = links[links.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [menuOpen]);

  const closeMenu = () => setMenuOpen(false);

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden>
            <Image src="/favicon.ico" width={16} height={16} alt="" />
          </span>
          <strong className="brand-name">Scryglass</strong>
        </Link>
        <nav className="site-nav site-nav-flat site-nav-desktop" aria-label="Primary">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              aria-current={isCurrent(pathname, link.href) ? "page" : undefined}
              className="nav-link"
            >
              {link.label}
            </Link>
          ))}
        </nav>
        <span
          className="data-freshness"
          aria-label={updated ? `Data updated ${updated}` : "Data freshness unavailable"}
        >
          <i aria-hidden />
          <span>{updated ? `Updated ${updated}` : "Data status"}</span>
        </span>
        <button
          type="button"
          className="menu-toggle"
          ref={menuToggleRef}
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
          aria-controls="mobile-primary-menu"
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? <XIcon size={17} aria-hidden /> : <ListIcon size={17} aria-hidden />}
          <span>{menuOpen ? "Close" : "Menu"}</span>
        </button>
        <ThemeToggle />
      </div>
      {menuOpen && (
        <>
          <button type="button" className="site-menu-scrim" aria-label="Close navigation" onClick={closeMenu} />
          <nav
            id="mobile-primary-menu"
            ref={menuRef}
            className="site-menu"
            aria-label="Primary mobile navigation"
          >
            {LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                aria-current={isCurrent(pathname, link.href) ? "page" : undefined}
                className="site-menu-link"
                onClick={closeMenu}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </>
      )}
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div>
        <strong>Scryglass</strong>
        <span>Independent League of Legends research by koi.</span>
      </div>
      <div>
        <span>OE baseline + GRID recent pro rows</span>
        <Link href="/reproduce" className="row-link">
          Data &amp; reproduction <ArrowUpRightIcon size={13} aria-hidden />
        </Link>
        <span className="font-mono">2025–2026</span>
      </div>
    </footer>
  );
}
