import katex from "katex";

/** Server-side KaTeX HTML (import CSS where used: grubs page / layout). */
export function blockMathHtml(tex: string): string {
  return katex.renderToString(tex, {
    displayMode: true,
    throwOnError: false,
    strict: "ignore",
    output: "html",
  });
}
