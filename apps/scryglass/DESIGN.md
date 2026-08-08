# Scryglass application design

The application follows the research-tool system documented in the repository root `DESIGN.md`.

## Application contract

- Core palette: paper `#F2ECDF`, Superbet red `#FF0000`, ink `#121212`.
- Typeface: IBM Plex Sans for interface copy, IBM Plex Mono only for numbers, timestamps, IDs, and
  model notation.
- Shell: black navigation and footer; near-full-width working surface.
- Data flow: operational pages put the requested table, board, or result in the first viewport.
- Signature: a red proofing line marks an active route, current draft seat, selected filter, or
  meaningful change. It is never a decorative ranking treatment.
- Theme: system, light, and dark remain supported.
- Geometry: square controls, one-pixel rules, no shadows, gradients, glass, or ornamental cards.
- Disclosure: competitive level stays visible; secondary league, event, and methodology controls open
  only when requested.
- Voice: literal, impersonal, and concise. Product pages name the object and the measured quantity
  without “desk,” “ledger,” “board,” or other invented institutional framing.
- Responsive: structural one-column changes at tablet and mobile widths; no tiny desktop tables squeezed
  into mobile.
- Accessibility: visible focus rings, semantic control names, keyboard-operable navigation and tables,
  color-independent labels, reduced-motion support.
