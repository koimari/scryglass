# Design System

## Visual Theme

**Mood:** a quiet research tool used under time pressure. The interface should feel direct, auditable,
and composed without performing seriousness.

**Visual thesis:** Scryglass puts the requested evidence before its framing. Paper is the ordinary
working surface, black belongs to the shell and genuinely live analysis, and red marks a state change.
The data establishes authority. Decorative publishing conventions do not.

**Signature:** a red proofing line marks the current page, active decision, or selected row. It is an
information state, never decoration.

**Theme:** system, light, and dark are supported. Light uses a beige working field; dark inverts that
field. Neither theme turns every data surface into a black panel.

## Core Colors

| Role | Value | Use |
|------|-------|-----|
| Paper | `#F2ECDF` | Primary reading and working surface |
| Raised paper | `#F8F3E9` | Inputs and bounded context panels |
| Paper rule | `#C9C0B1` | Hairlines and table structure |
| Ink | `#121212` | Text, app frame, evidence surfaces |
| Superbet red | `#FF0000` | Active state, current decision, primary action |
| Red text | `#D90000` light / `#FF3434` dark | Small labels and signed deltas |

Do not add orange, lime, cyan, purple, gradients, glow, or translucent glass. Red is restrained to
actions and state, normally below 12% of the screen.

## Typography

- UI and headings: IBM Plex Sans, 400-700.
- Numbers, timestamps, pack IDs, and model notation: IBM Plex Mono, 400-500.
- Headings use weight and scale, not a decorative display family.
- Editorial display: `clamp(3.5rem, 7vw, 7rem)`.
- Operational page title: fixed `2.25rem` desktop and `1.8rem` mobile.
- Section title: `clamp(1.5rem, 2.4vw, 2.4rem)`.
- Body copy remains 16-17px with a 62-70ch reading measure and 1.6-1.7 line height.
- Data text remains 14-15px. Mobile interface text remains at least 14px.
- Tables use tabular numerals and right-align quantitative columns.
- Do not pair giant sans headings with tiny uppercase mono kickers on operational pages.

## Layout

- The persistent shell is black. Working pages sit inside a near-full-width paper field.
- Editorial pages use a real first-viewport thesis: one dominant claim, one evidence visual, one action.
- Operational pages start with the working surface. Title, provenance, primary controls, and the first
  result belong in the first viewport.
- Profile pages pair identity directly with the primary rating; secondary evidence forms a compact rail.
- Match pages make the result the hero. IDs and provenance remain subordinate.
- Methodology uses a persistent section index and a 62ch reading column, not an accordion wall.
- Data-heavy results use ordinary paper rows. Black is reserved for active analysis or match state.
- Mobile layouts become one column structurally; typography does not use fluid novelty.
- Cards are reserved for interactive selections. Static information uses rules, rows, and type.

## Content Plan

1. Orientation: the entity, match, question, or estimand currently being inspected.
2. Primary surface: the board, ranking, table, or research claim needed to act.
3. Evidence: uncertainty, provenance, decomposition, and supporting detail.
4. Next action: compare, branch, inspect, reproduce, or continue reading.

## Interaction Thesis

- Keep the current entity, series, scope, or draft state visible while evidence scrolls.
- Use progressive disclosure for secondary scopes and model detail. Do not expose the full query
  vocabulary before the result.
- Use 160-220ms transitions for selected scopes, expanded series, and disclosure states.
- Reveal provenance and decomposition on focus/hover without moving the surrounding layout.
- Preserve reduced-motion preferences and URL-addressable analysis state.

## Components

- Square controls and one-pixel borders. No decorative radii or shadows.
- Active navigation uses a three-pixel red baseline.
- Selected table rows use a restrained state treatment, never decorative podium styling.
- Callouts use a full border or black header band.
- Data graphics use labeled horizontal bars for rankings and percentages. Each bar keeps its exact value, sample size, axis, and baseline visible. Detail rows remain available for audit.
- Loading uses skeleton rows; empty and error states explain the next useful action.
- Phosphor supplies interface icons. Do not draw replacement icons with CSS or text glyphs.

## Motion

- 150-220ms state transitions only.
- No page-load choreography, bounce, parallax, glow, or decorative loops.
- Reduced-motion settings remove all non-essential transitions.

## Do / Don't

**Do:** state the estimand, separate raw draft EV from roster context, keep filters legible, and preserve
two-decimal discipline where the model reports percentage-point effects. Name pages and actions
literally.

**Don't:** build SaaS metric cards, broadsheet pastiche, generic esports HUDs, oversized marketing
heroes, or visually imply certainty the model does not have. Do not invent institutional nouns such as
“desk,” “ledger,” or “board” when the interface can name the actual object.
