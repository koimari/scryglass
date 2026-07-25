"use client";

import { useEffect, useRef, useState } from "react";
import { useTheme, type ThemeChoice } from "./ThemeProvider";

const OPTIONS: { id: ThemeChoice; label: string }[] = [
  { id: "system", label: "System" },
  { id: "light", label: "Light" },
  { id: "dark", label: "Dark" },
];

export function ThemeToggle() {
  const { choice, setChoice } = useTheme();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const label = OPTIONS.find((o) => o.id === choice)?.label ?? "System";

  return (
    <div className="theme-toggle" ref={root}>
      <button
        type="button"
        className="theme-toggle-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Theme"
        onClick={() => setOpen((v) => !v)}
      >
        {label}
      </button>
      {open && (
        <ul className="theme-menu" role="menu">
          {OPTIONS.map((o) => (
            <li key={o.id} role="none">
              <button
                type="button"
                role="menuitemradio"
                aria-checked={choice === o.id}
                className={choice === o.id ? "is-active" : undefined}
                onClick={() => {
                  setChoice(o.id);
                  setOpen(false);
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
