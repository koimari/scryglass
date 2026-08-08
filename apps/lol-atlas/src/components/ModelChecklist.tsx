"use client";

import type { QueryRow } from "@/lib/duck";

type Props = {
  map: QueryRow;
  players?: QueryRow[];
};

export function ModelChecklist({}: Props) {
  return (
    <div className="model-checklist">
      <h3>Validation status</h3>
      <p className="status-hint">
        Predictive model-versus-result comparisons are withheld in this public MVP.
      </p>
    </div>
  );
}
