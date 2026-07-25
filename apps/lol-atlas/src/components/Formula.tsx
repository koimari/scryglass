"use client";

import katex from "katex";
import "katex/dist/katex.min.css";

type MathProps = {
  tex: string;
  className?: string;
};

function render(tex: string, displayMode: boolean): string {
  return katex.renderToString(tex, {
    displayMode,
    throwOnError: false,
    strict: "ignore",
    output: "html",
  });
}

/** Display (block) equation. */
export function BlockMath({ tex, className }: MathProps) {
  return (
    <div
      className={
        className ??
        "my-2 overflow-x-auto text-inherit [&_.katex]:text-[1.05em] [&_.katex]:text-[color:inherit]"
      }
      dangerouslySetInnerHTML={{ __html: render(tex, true) }}
    />
  );
}

/** Inline math fragment. */
export function InlineMath({ tex, className }: MathProps) {
  return (
    <span
      className={
        className ?? "text-inherit [&_.katex]:text-[1em] [&_.katex]:text-[color:inherit]"
      }
      dangerouslySetInnerHTML={{ __html: render(tex, false) }}
    />
  );
}
