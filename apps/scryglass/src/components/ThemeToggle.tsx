"use client";

import { useEffect, useRef, useState } from "react";
import { useTheme, type ThemeChoice } from "./ThemeProvider";

const OPTIONS: { id: ThemeChoice; label: string }[] = [
  { id: "light", label: "Light" },
  { id: "dark", label: "Dark" },
];

export function ThemeToggle() {
  const { choice, setChoice } = useTheme();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        btnRef.current?.focus();
        return;
      }
      if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      const items = itemRefs.current.filter(Boolean) as HTMLButtonElement[];
      if (!items.length) return;
      const idx = items.findIndex((el) => el === document.activeElement);
      let next = idx;
      if (e.key === "ArrowDown") next = idx < 0 ? 0 : (idx + 1) % items.length;
      if (e.key === "ArrowUp") next = idx < 0 ? items.length - 1 : (idx - 1 + items.length) % items.length;
      if (e.key === "Home") next = 0;
      if (e.key === "End") next = items.length - 1;
      items[next]?.focus();
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    // Focus active option when opened
    const activeIdx = Math.max(
      0,
      OPTIONS.findIndex((o) => o.id === choice),
    );
    queueMicrotask(() => itemRefs.current[activeIdx]?.focus());
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, choice]);

  const label = OPTIONS.find((o) => o.id === choice)?.label ?? "Light";

  return (
    <div className="theme-toggle" ref={root}>
      <button
        ref={btnRef}
        type="button"
        className="theme-toggle-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Theme"
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" && !open) {
            e.preventDefault();
            setOpen(true);
          }
        }}
      >
        {label}
      </button>
      {open && (
        <ul className="theme-menu" role="menu">
          {OPTIONS.map((o, i) => (
            <li key={o.id} role="none">
              <button
                ref={(el) => {
                  itemRefs.current[i] = el;
                }}
                type="button"
                role="menuitemradio"
                aria-checked={choice === o.id}
                className={choice === o.id ? "is-active" : undefined}
                onClick={() => {
                  setChoice(o.id);
                  setOpen(false);
                  btnRef.current?.focus();
                }}
              >
                {o.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
