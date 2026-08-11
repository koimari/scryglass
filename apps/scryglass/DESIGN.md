# Scryglass application design

The application follows the research-tool system documented in the repository root `DESIGN.md`.

## Application contract

- Core palette: warm gray canvas, solid white or charcoal working surfaces, and near-black or warm-white
  text. Strong borders and secondary text keep both themes readable. Champion and team artwork supplies color.
- Typeface: Atkinson Hyperlegible Next for interface copy, Atkinson Hyperlegible Mono for numbers,
  timestamps, IDs, and model notation, and Instrument Serif for large editorial headings.
- Shell: black navigation and footer; near-full-width working surface.
- Data flow: operational pages put the requested table, board, or result in the first viewport.
- Signature: a red proofing line marks an active route, selected filter, or
  meaningful change. It is never a decorative ranking treatment.
- Theme: system, light, and dark remain supported.
- Geometry: soft 8–14 px corners, one-pixel rules, and flat grouped surfaces. Shadows appear only on
  floating menus and hovered interactive cards.
- Motion: short exponential scroll inertia connects long data surfaces. The header condenses into a
  floating lens after scrolling begins. Touch, nested data grids, and reduced-motion preferences keep
  direct browser scrolling.
- Disclosure: ratings can show league context. Tier lists expose patch and role only.
- Voice: literal, impersonal, and concise. Product pages name the object and the measured quantity
  without “desk,” “ledger,” “board,” or other invented institutional framing.
- Responsive: structural one-column changes at tablet and mobile widths; no tiny desktop tables squeezed
  into mobile.
- Accessibility: visible focus rings, semantic control names, keyboard-operable navigation and tables,
  color-independent labels, reduced-motion support.
