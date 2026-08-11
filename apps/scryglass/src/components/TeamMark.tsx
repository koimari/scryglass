import { teamMarkUrl } from "@/lib/teamMarks";
import styles from "./TeamMark.module.css";

type Size = "small" | "medium" | "large";

export function TeamMark({ team, size = "small" }: { team: string | null | undefined; size?: Size }) {
  const src = teamMarkUrl(team);
  if (!src || !team) return null;
  return (
    <span className={`${styles.mark} ${styles[size]}`}>
      {/* Marks are cached local assets with sources recorded beside the files. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={`${team} mark`} loading="lazy" />
    </span>
  );
}
