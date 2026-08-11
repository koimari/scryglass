import { teamInitials, teamMarkUrl } from "@/lib/teamMarks";
import styles from "./TeamMark.module.css";

type Size = "small" | "medium" | "large";

export function TeamMark({ team, size = "small" }: { team: string | null | undefined; size?: Size }) {
  const src = teamMarkUrl(team);
  if (!team) return null;
  return (
    <span className={`${styles.mark} ${styles[size]} ${src ? "" : styles.fallback}`}>
      {src ? (
        <>
          {/* Leaguepedia's original transparent PNG, without a white tile. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt={`${team} mark`} loading="lazy" referrerPolicy="no-referrer" />
        </>
      ) : <span aria-label={`${team} lettermark`}>{teamInitials(team)}</span>}
    </span>
  );
}
