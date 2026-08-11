"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ThemeToggle } from "./ThemeToggle";

const LINKS = [
  { href: "/elo", label: "Ratings" },
  { href: "/matches", label: "Matches" },
  { href: "/tiers", label: "Tier lists" },
  { href: "/methodology", label: "Method" },
];

function isCurrent(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function SiteHeader() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuToggleRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLElement>(null);

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
          <Image
            className="brand-mark"
            src="/scryglass-signal-mark.png"
            width={40}
            height={40}
            alt=""
            priority
          />
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
        <span className="data-freshness" aria-label="Ratings refresh every six hours">
          <i aria-hidden />
          <span>6h refresh</span>
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
        <span>League of Legends ratings by koi.</span>
      </div>
      <div>
        <span>Completed professional games</span>
        <Link href="/methodology" className="row-link">Method</Link>
      </div>
    </footer>
  );
}
