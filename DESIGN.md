# Design System

## Visual Theme

**Mood:** daylight desk paper, cool ink, lab report. Physical scene: reading a printed working paper under cool ambient light; no dashboard glow.

**Color strategy:** Restrained. Cool-tinted neutrals + one steel accent ≤10% (negative EV / warnings only).

**Theme:** Light. Forced by print/PDF reading in daylight. Not dark-mode analytics.

## Colors

| Role | OKLCH | Hex approx | Use |
|------|-------|------------|-----|
| Paper | oklch(0.97 0.005 250) | #F4F5F7 | Page ground |
| Ink | oklch(0.22 0.01 250) | #1C1E24 | Body text |
| Muted | oklch(0.45 0.01 250) | #5C606A | Captions, meta |
| Rule | oklch(0.82 0.008 250) | #C8CAD0 | Hairlines |
| Accent | oklch(0.42 0.04 250) | #3D4A5C | Steel: figure emphasis, negative edge |
| Danger ink | oklch(0.40 0.08 25) | #6B3A32 | Avoid / negative edge only |

Never pure #000 / #fff. No purple. No warm cream+terracotta. No neon.

## Typography

- **Display / title:** system academic stack for PDF: Times-Bold (reportlab) or embedded Libertinus/Source Serif if available. Not Inter, Playfair, Fraunces, Space Grotesk.
- **Body:** Times-Roman, 10-11pt, leading 14-15pt, measure ~65-75ch.
- **Meta / tables / captions:** Helvetica, 8-9pt, cool muted ink.
- Hierarchy via size + weight only. No gradient text. No italic display affectation as brand costume.

## Components (print)

- Title block: title, version line, estimand one-liner. No pill chips.
- Tables: full-width, hairline rules, numeric columns right-aligned, pp to 2 decimals.
- Figures: grayscale or steel single series; caption below; table duplicate when decision-critical.
- Callouts: full-border or background tint, never left side-stripe >1px.
- No cards. No hero-metric blocks.

## Layout

- Letter / A4, 0.75-1.0in margins.
- Single column body. Section numbers as in the working paper.
- Page numbers footer center, muted.
- Vertical rhythm: tighter within sections, larger gap before H1/H2.

## Do / Don't

**Do:** cold precision, ASCII hyphens, estimand stated early, charts + tables together.  
**Don't:** SaaS metrics, emoji, motivational headers, dark glass, purple accents.
