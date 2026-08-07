---
name: "pdf"
description: "Use when tasks involve reading, creating, or reviewing PDF files; prefer `pdf-inspector` for fast classification and Markdown extraction, render pages with Poppler when layout matters, and use Python PDF tools for generation or fallback extraction."
---


# PDF Skill

## When to use
- Read or review PDF content where layout and visuals matter.
- Create PDFs programmatically with reliable formatting.
- Validate final rendering before delivery.

## Workflow
1. Run the global PDF inspector first for fast classification and structured text extraction.
   - Classify with `pdf-inspector detect "$INPUT_PDF" --json`.
   - Extract Markdown with `pdf-inspector "$INPUT_PDF"` or structured output with `pdf-inspector "$INPUT_PDF" --json`.
   - Use `--pages 1,3,5` when only selected pages are needed.
   - Treat `Scanned`, `ImageBased`, `Mixed`, non-empty `pagesNeedingOcr`, or unreliable/garbled text as an OCR or visual-review signal. The inspector does not perform OCR.
2. Prefer visual review when layout or appearance matters: render PDF pages to PNGs and inspect them.
   - Use `pdftoppm` if available.
   - If unavailable, install Poppler or ask the user to review the output locally.
3. Use `reportlab` to generate PDFs when creating new documents.
4. Use `pdfplumber` (or `pypdf`) as a fallback or for Python-specific extraction tasks; do not rely on text extraction alone for layout fidelity.
5. After each meaningful update, re-render pages and verify alignment, spacing, and legibility.

## Global PDF inspector
- Command: `/opt/homebrew/bin/pdf-inspector` (normally available as `pdf-inspector` on `PATH`).
- Source clone: `/Users/river/Documents/Codex/tools/pdf-inspector`.
- Installed package: `@firecrawl/pdf-inspector`.
- The tool is local-only and requires no API key or external PDF upload.

## Temp and output conventions
- Use `tmp/pdfs/` for intermediate files; delete when done.
- Write final artifacts under `output/pdf/` when working in this repo.
- Keep filenames stable and descriptive.

## Dependencies (install if missing)
Prefer `uv` for dependency management.

Python packages:
```
uv pip install reportlab pdfplumber pypdf
```
If `uv` is unavailable:
```
python3 -m pip install reportlab pdfplumber pypdf
```
System tools (for rendering):
```
# macOS (Homebrew)
brew install poppler

# Ubuntu/Debian
sudo apt-get install -y poppler-utils
```

If installation isn't possible in this environment, tell the user which dependency is missing and how to install it locally.

## Environment
No required environment variables.

## Rendering command
```
pdftoppm -png $INPUT_PDF $OUTPUT_PREFIX
```

## Quality expectations
- Maintain polished visual design: consistent typography, spacing, margins, and section hierarchy.
- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
- Charts, tables, and images must be sharp, aligned, and clearly labeled.
- Use ASCII hyphens only. Avoid U+2011 (non-breaking hyphen) and other Unicode dashes.
- Citations and references must be human-readable; never leave tool tokens or placeholder strings.

## Final checks
- Do not deliver until the latest PNG inspection shows zero visual or formatting defects.
- Confirm headers/footers, page numbering, and section transitions look polished.
- Keep intermediate files organized or remove them after final approval.
