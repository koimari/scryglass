# Signal Field design QA

## Target

- Reference: `/Users/river/Projects/scryglass/.impeccable/sketches/signal-field.webp`
- Routes: `/elo`, `/matches`, player profiles, team profiles, and match profiles
- Desktop comparison viewport: 1000 x 625
- Mobile check: 390 x 844 override

## Comparison history

### Pass 1

- P2, visual signal: Rating bars looked almost equal. The scale included zero, which compressed the valid rating range.
  - Fix: Scale each signal against its visible minimum and maximum.
- P2, content: The focus footer repeated the word `confidence`.
  - Fix: Use the public evidence label without a suffix.
- P2, performance: The match gallery mounted 100 full cards on its first render.
  - Fix: Show 31 maps first, add an explicit show-all control, and skip off-screen card rendering.

### Pass 2

- Typography follows the sharp display and instrument-label hierarchy from the reference.
- Spacing uses wide fields, hairline divisions, and square controls.
- Color stays black and white. Champion portraits supply the main color accents.
- Ratings use a signal field, leader rail, focused entity, movers, and a responsive gallery.
- Matches use gallery, timeline, and tournament views with a signal header and champion strips.
- Desktop and mobile layouts have no horizontal document overflow.
- Mobile navigation opens and closes. Controls retain usable labels and focus states.
- Profile and match links resolve. Recent player maps include champion portraits, KDA, and grades.
- Browser warnings and errors: none.
- Lint, 34 tests, and production build: passed.

### User feedback pass

- Removed the large rating waveform, compact rail waveforms, card waveforms, match waveform, and map-grade bars.
- Replaced them with one compact summary row and one selected entity as the visual focus.
- Kept the warm black-and-white palette and champion portraits.
- Removed repeated headings and duplicate match totals.

### Identity hierarchy pass

- Added reviewed local team marks to rating leaders, movers, focused profiles, rating cards, and match cards.
- Enlarged champion portraits in focused profiles and featured matches.
- Added compact champion groups to rating cards, so champion color identifies recent form at a glance.
- Teams without a reviewed mark keep the text layout. The interface does not create a generic placeholder.
- Desktop and mobile layouts keep their document width. Images load without broken assets.
- Team-name aliases resolve to the same reviewed mark.
- Team profile headers now carry the reviewed team mark.
- Player profile headers use reviewed portraits for prominent rated players, with a team-mark or initial fallback.
- Portrait lookup stays explicit because duplicate player handles can identify different people.

## Final result

passed
