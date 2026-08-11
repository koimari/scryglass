"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type ThemeChoice = "light" | "dark";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "scryglass-theme";

type ThemeContextValue = {
  choice: ThemeChoice;
  resolved: ResolvedTheme;
  setChoice: (c: ThemeChoice) => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

function applyDom(resolved: ResolvedTheme) {
  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoiceState] = useState<ThemeChoice>("light");
  const [resolved, setResolved] = useState<ResolvedTheme>("light");

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as ThemeChoice | null;
    const initial: ThemeChoice = stored === "dark" ? "dark" : "light";
    applyDom(initial);
    queueMicrotask(() => {
      setChoiceState(initial);
      setResolved(initial);
    });
  }, []);

  const setChoice = useCallback((c: ThemeChoice) => {
    setChoiceState(c);
    localStorage.setItem(STORAGE_KEY, c);
    setResolved(c);
    applyDom(c);
  }, []);

  const value = useMemo(
    () => ({ choice, resolved, setChoice }),
    [choice, resolved, setChoice],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme requires ThemeProvider");
  return ctx;
}
