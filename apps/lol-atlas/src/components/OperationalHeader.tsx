import type { ReactNode } from "react";
import styles from "./OperationalHeader.module.css";

type OperationalHeaderProps = {
  title: string;
  description?: ReactNode;
  meta?: ReactNode;
};

export function OperationalHeader({
  title,
  description,
  meta,
}: OperationalHeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.copy}>
        <h1>{title}</h1>
        {description ? <p>{description}</p> : null}
      </div>
      {meta ? <div className={styles.meta}>{meta}</div> : null}
    </header>
  );
}
