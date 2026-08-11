"use client";

import { useState } from "react";
import type { PlayerVisualIdentity } from "@/lib/playerPortraits";
import { teamMarkUrl } from "@/lib/teamMarks";
import { TeamMark } from "./TeamMark";
import styles from "./PlayerPortrait.module.css";

export function PlayerPortrait({
  player,
  team,
  portrait,
}: {
  player: string;
  team?: string | null;
  portrait?: PlayerVisualIdentity | null;
}) {
  const [failed, setFailed] = useState(false);
  const src = portrait?.src;
  const source = portrait?.source;
  const hasPortrait = Boolean(src && !failed);
  const hasTeamMark = Boolean(teamMarkUrl(team));

  return (
    <figure className={styles.frame}>
      {hasPortrait ? (
        <a className={styles.photoLink} href={source ?? undefined} target="_blank" rel="noreferrer" title={`${player} photo source`}>
          {/* Reviewed remote image. Failure keeps the profile usable. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className={styles.photo}
            src={src ?? undefined}
            alt={`${player} portrait`}
            referrerPolicy="no-referrer"
            onError={() => setFailed(true)}
          />
        </a>
      ) : (
        <div className={styles.fallback} aria-label={`${player} portrait unavailable`}>
          {hasTeamMark ? <TeamMark team={team} size="large" /> : <span aria-hidden>{player.slice(0, 1).toUpperCase()}</span>}
        </div>
      )}
      {hasPortrait && hasTeamMark ? <span className={styles.teamBadge}><TeamMark team={team} size="small" /></span> : null}
    </figure>
  );
}
