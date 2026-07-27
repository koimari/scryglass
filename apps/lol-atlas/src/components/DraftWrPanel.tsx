"use client";

import { useEffect, useState } from "react";
import type { QueryRow } from "@/lib/duck";

export type DraftStrengthInput = {
  teamEloDiff?: number | null;
  playerEloDiff?: number | null;
  source?: string | null;
};

type Props = {
  map: QueryRow;
  players: QueryRow[];
  strength?: DraftStrengthInput;
};

const WITHHELD =
  "Withheld · the current composition model did not beat the chronological base-rate benchmark.";

type GateSummary = {
  gateId: string;
  finalTestMaps: number;
  finalTestEnd: string;
};

function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function validatedGateSummary(payload: unknown): GateSummary | null {
  const root = record(payload);
  const evidence = record(root?.evidence);
  const gateId =
    typeof root?.gate_id === "string" ? root.gate_id.trim() : "";
  const finalTestMaps = evidence?.final_test_maps;
  const finalTestEnd =
    typeof evidence?.final_test_end === "string"
      ? evidence.final_test_end
      : "";
  if (
    root?.status !== "withheld_failed_chronological_gate" ||
    root?.evidence_status !== "verified_immutable_pack_artifact" ||
    evidence?.decision !== "withheld" ||
    !/^[a-z0-9][a-z0-9-]{2,100}$/.test(gateId) ||
    typeof finalTestMaps !== "number" ||
    !Number.isInteger(finalTestMaps) ||
    finalTestMaps <= 0 ||
    !Number.isFinite(Date.parse(finalTestEnd))
  ) {
    return null;
  }
  return { gateId, finalTestMaps, finalTestEnd };
}

/**
 * Historical match surfaces fail closed while the composition probability
 * pipeline is outside its promotion gate. The interactive Sandbox exposes a
 * separately labelled experimental policy value and must not be substituted
 * here.
 */
export function DraftWrPanel(_props: Props) {
  void _props;
  const [evidence, setEvidence] = useState<GateSummary | null | "loading">(
    "loading",
  );

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/draft-wr", {
      cache: "no-store",
      signal: controller.signal,
    })
      .then((response) => response.json())
      .then((payload: unknown) => {
        setEvidence(validatedGateSummary(payload));
      })
      .catch(() => {
        if (!controller.signal.aborted) setEvidence(null);
      });
    return () => controller.abort();
  }, []);

  return (
    <div className="check-row draft-wr-row">
      <span>Draft forecast:</span>
      <span className="text-[var(--ink-muted)]">{WITHHELD}</span>
      <span className="w-full text-xs text-[var(--ink-muted)]">
        {evidence === "loading"
          ? "Loading immutable gate evidence"
          : evidence
            ? `Gate ${evidence.gateId} · ${evidence.finalTestMaps.toLocaleString(
                "en-US",
              )} maps · final test through ${evidence.finalTestEnd.slice(0, 10)}`
            : "Immutable gate evidence unavailable"}{" "}
        ·{" "}
        <a href="/methodology">methods and transfer limits</a>
      </span>
      <span className="check-mark na">—</span>
    </div>
  );
}
