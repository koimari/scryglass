import Link from "next/link";
import type { CSSProperties, ReactNode } from "react";
import styles from "./DataBars.module.css";

export type DataBarRow = {
  id: string;
  label: string;
  href?: string;
  value: number;
  valueLabel: string;
  detail: string;
  mark?: ReactNode;
  tone?: "positive" | "negative" | "neutral";
};

type Domain = {
  min: number;
  max: number;
};

type DataBarsProps = {
  title: string;
  description: string;
  rows: DataBarRow[];
  domain?: Domain;
  baseline?: number;
  baselineLabel?: string;
  axisMiddle?: string;
  axisLeft: string;
  axisRight: string;
  limit?: number;
  className?: string;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

function scaleValue(value: number, domain: Domain): number {
  if (domain.max <= domain.min) return 50;
  return clamp(((value - domain.min) / (domain.max - domain.min)) * 100, 0, 100);
}

function formatDomain(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

export function DataBars({
  title,
  description,
  rows,
  domain,
  baseline,
  baselineLabel,
  axisMiddle,
  axisLeft,
  axisRight,
  limit = 8,
  className,
}: DataBarsProps) {
  const visibleRows = rows.slice(0, limit).filter((row) => Number.isFinite(row.value));
  if (!visibleRows.length) return null;

  const values = visibleRows.map((row) => row.value);
  const resolvedDomain = domain ?? {
    min: Math.floor((Math.min(...values) - 25) / 50) * 50,
    max: Math.ceil((Math.max(...values) + 25) / 50) * 50,
  };
  const hasBaseline = baseline != null && baseline >= resolvedDomain.min && baseline <= resolvedDomain.max;
  const baselinePosition = hasBaseline ? scaleValue(baseline, resolvedDomain) : null;
  const classNames = [styles.panel, className].filter(Boolean).join(" ");

  return (
    <section className={classNames} aria-label={title}>
      <header className={styles.header}>
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        <span>{visibleRows.length} shown</span>
      </header>
      <div className={styles.axis} aria-hidden="true">
        <span>{axisLeft}</span>
        <span>{axisMiddle ?? (baselineLabel && hasBaseline ? baselineLabel : formatDomain((resolvedDomain.min + resolvedDomain.max) / 2))}</span>
        <span>{axisRight}</span>
      </div>
      <ol className={styles.rows}>
        {visibleRows.map((row, index) => {
          const position = scaleValue(row.value, resolvedDomain);
          const distanceFromBaseline = hasBaseline ? Math.abs(position - (baselinePosition ?? 0)) : position;
          const left = hasBaseline && baselinePosition != null && position < baselinePosition ? position : baselinePosition ?? 0;
          const tone = row.tone ?? (hasBaseline && baseline != null && row.value < baseline ? "negative" : "positive");
          const style = {
            "--bar-left": `${left}%`,
            "--bar-width": `${hasBaseline ? distanceFromBaseline : position}%`,
            "--bar-position": `${position}%`,
          } as CSSProperties;
          const identity = (
            <span className={styles.identity}>
              {row.mark ? <span className={styles.mark}>{row.mark}</span> : null}
              <span className={styles.identityText}>
                <strong>{row.label}</strong>
                <small>{row.detail}</small>
              </span>
            </span>
          );
          return (
            <li key={row.id} className={styles.row}>
              <span className={styles.rank}>{String(index + 1).padStart(2, "0")}</span>
              {row.href ? <Link href={row.href} className={styles.link}>{identity}</Link> : identity}
              <span className={styles.track} style={style} aria-hidden="true">
                {hasBaseline ? <i className={styles.baseline} /> : null}
                <i className={`${styles.fill} ${styles[tone]}`} />
              </span>
              <strong className={styles.value}>{row.valueLabel}</strong>
            </li>
          );
        })}
      </ol>
      <p className={styles.note}>Bars show the published ordering metric. The detailed evidence remains available below.</p>
    </section>
  );
}
