"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ThemeToggle } from "./ThemeToggle";

const GROUPS = [
  {
    label: "Read",
    links: [{ href: "/articles", label: "Articles" }],
  },
  {
    label: "Explore",
    links: [
      { href: "/elo", label: "Ratings" },
      { href: "/browse", label: "Matches" },
      { href: "/browse/head-to-head", label: "H2H" },
      { href: "/sandbox", label: "Sandbox" },
    ],
  },
  {
    label: "Verify",
    links: [
      { href: "/methodology", label: "Method" },
      { href: "/reproduce", label: "Reproduce" },
    ],
  },
];

function isCurrent(pathname: string, href: string): boolean {
  if (href === "/articles") return pathname === href || pathname.startsWith("/articles/");
  if (href === "/browse") return pathname === href || pathname.startsWith("/browse/match");
  return pathname === href || pathname.startsWith(`${href}/`);
}

function relativeFreshness(
  raw: string | undefined,
  now = Date.now(),
): string | null {
  if (!raw) return null;
  const time = Date.parse(raw);
  if (!Number.isFinite(time)) return null;
  const minutes = Math.max(0, Math.round((now - time) / 60_000));
  if (minutes < 2) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export type PublicPackStatus = {
  pack_id?: string;
  filters?: {
    years?: unknown[];
  };
  source?: "remote" | "bundled";
  degraded?: boolean;
  clocks?: {
    publication?: { value?: string | null };
    data_through?: { value?: string | null; status?: string };
  };
  source_provenance?: {
    sources?: Array<{ source?: string }>;
  };
};

export function packClockLabels(
  manifest: PublicPackStatus | null,
  now = Date.now(),
): {
  published: string | null;
  dataThrough: string | null;
  degraded: boolean;
} {
  return {
    published: relativeFreshness(
      manifest?.clocks?.publication?.value ?? undefined,
      now,
    ),
    dataThrough: relativeFreshness(
      manifest?.clocks?.data_through?.value ?? undefined,
      now,
    ),
    degraded: Boolean(manifest?.degraded),
  };
}

export function packSourceLabel(manifest: PublicPackStatus | null): string {
  const declared = (manifest?.source_provenance?.sources ?? [])
    .map(({ source }) => source?.trim().toLowerCase())
    .filter((source): source is string => Boolean(source))
    .map((source) => {
      if (source === "oe") return "Oracle’s Elixir";
      if (source === "grid") return "GRID";
      return source.toUpperCase();
    });
  return declared.length > 0
    ? declared.join(" + ")
    : "Source provenance unavailable";
}

export function packYearsLabel(
  manifest: PublicPackStatus | null,
): string | null {
  const years = [
    ...new Set(
      (manifest?.filters?.years ?? []).filter(
        (year): year is number =>
          typeof year === "number" &&
          Number.isInteger(year) &&
          year >= 2000 &&
          year <= 3000,
      ),
    ),
  ].sort((a, b) => a - b);
  if (!years.length) return null;
  return years.length === 1
    ? String(years[0])
    : `${years[0]}–${years[years.length - 1]}`;
}

export function SiteHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [published, setPublished] = useState<string | null>(null);
  const [dataThrough, setDataThrough] = useState<string | null>(null);
  const [degraded, setDegraded] = useState(false);
  const latestPackRef = useRef<string | null>(null);
  const menuToggleRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      fetch("/api/pack-manifest", { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .then((manifest: PublicPackStatus | null) => {
          if (cancelled) return;
          const labels = packClockLabels(manifest);
          setPublished(labels.published);
          setDataThrough(labels.dataThrough);
          setDegraded(labels.degraded);
          const nextPack = manifest?.pack_id ?? null;
          if (latestPackRef.current && nextPack && latestPackRef.current !== nextPack) {
            router.refresh();
          }
          latestPackRef.current = nextPack;
        })
        .catch(() => {
          if (!cancelled) {
            setPublished(null);
            setDataThrough(null);
            setDegraded(false);
          }
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
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [menuOpen]);

  const closeMenu = () => setMenuOpen(false);

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden />
          <span className="brand-name">Scryglass</span>
        </Link>
        <nav className="site-nav site-nav-desktop" aria-label="Primary">
          {GROUPS.map((group, groupIndex) => (
            <div className="nav-group" key={group.label}>
              {groupIndex > 0 && <span className="nav-divider" aria-hidden />}
              <span className="nav-group-label">{group.label}</span>
              {group.links.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  aria-current={isCurrent(pathname, link.href) ? "page" : undefined}
                  className="nav-link"
                >
                  {link.label}
                </Link>
              ))}
            </div>
          ))}
        </nav>
        <span
          className="data-freshness"
          aria-label={[
            published ? `Pack published ${published}` : "Pack publication time unavailable",
            dataThrough ? `data through ${dataThrough}` : "data-through time unavailable",
            degraded ? "bundled fallback in use" : null,
          ]
            .filter(Boolean)
            .join("; ")}
        >
          {published ? `Pack ${published}` : "Pack —"} ·{" "}
          {dataThrough ? `Data through ${dataThrough}` : "Data through —"}
          {degraded ? " · bundled fallback" : ""}
        </span>
        <button
          type="button"
          className="menu-toggle"
          ref={menuToggleRef}
          aria-expanded={menuOpen}
          aria-controls="mobile-primary-menu"
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? "Close" : "Menu"}
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
            {GROUPS.map((group) => (
              <div className="site-menu-group" key={group.label}>
                <span className="nav-group-label">{group.label}</span>
                {group.links.map((link) => (
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
              </div>
            ))}
          </nav>
        </>
      )}
    </header>
  );
}

export function SiteFooter() {
  const [sources, setSources] = useState<string[] | null>(null);
  const [years, setYears] = useState<string | null | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/pack-manifest", { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((manifest: PublicPackStatus | null) => {
        if (cancelled) return;
        const label = packSourceLabel(manifest);
        setSources(
          label === "Source provenance unavailable" ? [] : label.split(" + "),
        );
        setYears(packYearsLabel(manifest));
      })
      .catch(() => {
        if (!cancelled) {
          setSources([]);
          setYears(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const sourceLabel =
    sources == null
      ? "Source provenance loading"
      : sources.length > 0
        ? sources.join(" + ")
        : "Source provenance unavailable";

  return (
    <footer className="site-footer">
    Independent LoL research by koi · {sourceLabel} ·{" "}
      <Link href="/reproduce" className="row-link">
        Data &amp; reproduction
      </Link>{" "}
      · Pack years{" "}
      <span className="font-mono">
        {years === undefined ? "loading" : years ?? "unavailable"}
      </span>
    </footer>
  );
}
