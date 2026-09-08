# Layout Structure Checks

> **Purpose**: the mechanical half of `/layout-structure` — every grep and
> classification rule. The reasoning behind each one lives in
> [`layout-philosophy.md`](layout-philosophy.md).
>
> **Who reads this**: Step 2 (scan) runs these commands verbatim; the Step 3
> classify agent uses the classification rules below alongside the
> philosophy doc.

**Web-only, scoped to `web/src`** (Svelte 5 + Vite, plain CSS — no
Tailwind, no JSX/TSX). Never touch generated files or `node_modules/`.
`*.test.ts` and `lib/__fixtures__/` are out of scope by default — fixture
data is frequently deliberately minimal and not representative of real
screens. `lib/icons/` (SVG icon wrapper components) has no structural
content worth auditing.

Every check below was originally written for a React 19 + Tailwind v4
codebase and is adapted here for Svelte + plain CSS: `--include="*.tsx"`
becomes `--include="*.svelte"` (a `.svelte` file holds markup, script, and
scoped `<style>` together — there is no separate template/stylesheet
split), Tailwind utility-class patterns (`order-2`, `w-[300px]`, `sm:flex`)
become the equivalent plain-CSS property (`order: 2`, `width: 300px`,
`@media (min-width: ...)`), and `.map()` becomes Svelte's `{#each}` block —
Svelte's actual list-rendering construct, not a JS array method in a JSX
return.

## Which principles this file can actually find

Not every principle is greppable, and pretending otherwise wastes a run.

| | Principles | How they surface |
|---|---|---|
| **Scanned** | L2 (partial), L3 (partial), L4, L5 (partial), L10, L12 (partial), L13, L14, L15, L18 (partial) | a grep in §1–2 targets them directly |
| **Judgment-only** | L1, L6, L7, L8, L9, L11, L16, L17 | no grep targets them; they surface only when reading a file by hand, or as the reasoning behind classifying a scanned hit |

"Judgment-only" doesn't mean unfindable — L8 (flex vs. grid) and L11
(primitive reuse) routinely show up once a human or agent is already reading
a component for another reason. It means no scan is *aimed* at them, so a
run that only reads grep output will never surface them; the Step 3 agent
must actually open files, not just classify hit lists.

---

## 1. Markup scan

Run from the repo root, scoped to `web/src`.

```bash
# Sectioning content without an accessible name (L4) — a <section> that
# isn't a labeled region is functionally a styling-only div
grep -rn "<section" web/src --include="*.svelte" \
  | grep -vE "aria-label|aria-labelledby"

# Heading usage — collect every heading to check rank sequencing by hand (L3).
# This only gathers candidates; skipped-rank detection needs the surrounding
# file read, not a regex, since ranks are sequential across sibling elements.
grep -rn "<h[1-6]" web/src --include="*.svelte"

# Tabular-looking data rendered as a div grid instead of <table> (L5) --
# no Tailwind grid-cols-* utility here, so look for the plain-CSS property
# instead, scoped to a component's own <style> block.
grep -rln "display:\s*grid" web/src --include="*.svelte" \
  | xargs grep -liE "price|size|stock|quantity|column|row" 2>/dev/null

# Long-collection markup that isn't a list element (L5) -- Svelte's
# {#each ...} is this codebase's list-rendering construct (there is no
# .map() in a JSX return to grep for). This is a weak proxy, not a precise
# filter: it surfaces every {#each} block with a few lines of context so a
# human/agent can check what's actually rendered inside it -- read the
# full block body before judging <ul>/<ol>/<li> vs. a bare wrapper.
grep -rn "{#each" web/src --include="*.svelte" -A 3

# Skip links / bypass blocks (L15) — presence check across the whole app,
# not per-file; expect at most a handful of hits total (likely zero --
# this is a single-page app with no repeated nav to bypass, worth
# confirming rather than assuming)
grep -rn "skip.to.main\|skip-link\|#main-content" web/src --include="*.svelte"

# Multiple <nav> elements needing distinguishing labels (L2)
grep -rln "<nav" web/src --include="*.svelte"
```

For the heading grep, read each file's full heading sequence in document
order — the rank-skip rule (`h2` not directly followed by `h4`) is a
sequence property, not a per-line one.

## 2. CSS scan

No Tailwind here — every pattern below targets a plain-CSS property inside
a `.svelte` file's `<style>` block or `web/src/app.css`, not a utility
class name.

```bash
# Source-order divergence — DOM order vs. visual order (L14).
# \b is load-bearing: an unanchored "order:" also matches inside "border:"
# (border: 1px solid ... satisfies "order:\s*-?[0-9]+" as a substring) --
# the exact false-positive class the original Tailwind version of this grep
# warned about, just surfacing through a different word here.
grep -rnE "\border:\s*-?[0-9]+|flex-direction:\s*(row|column)-reverse" web/src \
  --include="*.svelte" --include="*.css"

# Explicit grid placement that could reorder visually from DOM order (L14)
grep -rnE "grid-area|grid-row-start|grid-column-start" web/src \
  --include="*.svelte" --include="*.css"

# Device-width breakpoints on a shared lib/ component (L10) — a shared
# component keyed to viewport width breaks the moment it's reused in a
# different-sized slot. As of the layout redesign, App.svelte's own
# single narrow-viewport @media query is a page-level decision (correct
# use, per L10's "media queries remain correct for page-level layout");
# the finding to look for is a @media query inside one of the individual
# lib/ components (Scoreboard.svelte, ActionPanel.svelte, etc.), which
# would be keyed to the viewport instead of the component's own slot.
grep -rn "@media" web/src/lib --include="*.svelte"
grep -rn "@container\|container-type" web/src --include="*.svelte" --include="*.css"

# Ad-hoc / escalating z-index values with no shared scale (L13) -- app.css
# has no --z-* custom property yet, so any z-index at all is worth reading
# in context, not just an escalating ladder.
grep -rnE "z-index:\s*[0-9]+" web/src --include="*.svelte" --include="*.css"

# Truncation that may be silently inert without min-width: 0 (L12) --
# gather candidates, then check by hand whether the truncating element sits
# inside a flex/grid item with no min-width: 0 sibling declaration
grep -rn "text-overflow:\s*ellipsis\|overflow:\s*hidden" web/src --include="*.svelte"

# Fixed pixel widths on layout containers, candidate for intrinsic sizing (L9)
grep -rnE "width:\s*[0-9]+px|min-width:\s*[0-9]+px" web/src \
  --include="*.svelte" --include="*.css" | grep -v "test"

# Arbitrary spacing values breaking app.css's own --space-* scale (L18) --
# app.css's :root defines --space-1 (0.25rem/4px) through --space-6 (2rem/
# 32px) specifically so nothing after it hardcodes a spacing value; any
# margin/padding/gap literal that isn't var(--space-*) is the candidate.
grep -rnE "(margin|padding|gap):\s*[0-9.]+(rem|px)" web/src \
  --include="*.svelte" --include="*.css" | grep -v "var(--space"
```

For the z-index and container-query greps, run them together — a codebase
with zero `@container`/`container-type` hits and a `@media` query inside a
reusable `lib/` component is the strongest L10 signal; report the absence,
not just the presence, per the Report format's `Correct-as-is` convention.

---

## 3. Classification rules

Read enough surrounding markup/CSS to understand what the element is doing
before classifying — a grep hit alone rarely settles which principle
applies, and never settles severity.

### DEFER table — kept as a reference for what these categories are, even with no sibling skill to route to

In the original (`fitted`) project this table routed a candidate to a
named sibling skill. **None of those skills exist in this project** — so a
candidate matching a row below is neither judged under this skill's own
rules nor silently dropped: report it under the "Uncovered — no sibling
skill in this project" section instead. Keeping the table (rather than
deleting it) is what prevents this skill from quietly expanding its own
scope into territory it was never built to judge carefully.

| Pattern | Would-be owner | Why it's not this skill's to judge |
|---|---|---|
| Missing/redundant `aria-label`, `alt`, or any accessible-name gap on an interactive/content element | accessible-name review | a dedicated accessibility pass owns accessible-name findings outright |
| A plain layout `<div>` with no role | accessible-name review | not a defect at all — never flag it, in this skill or any other |
| Positive `tabIndex`, focus traps, dialog focus management | keyboard/focus review | a dedicated a11y pass owns keyboard/focus findings |
| WCAG success-criterion tagging for landmarks/heading skips | a11y compliance review | this skill owns the underlying markup/outline technique (L2, L3); a dedicated pass would own the SC tagging |
| Specific proximity px thresholds, density ceiling, F/Z-pattern placement, grouping-count thresholds | UX/visual-design review | these are perceptual-design judgment calls, not structural-markup ones |
| `<div onClick>` standing in for a native control, navigation-as-button, design-system primitive reuse | component-composition review | owns leaf interactive elements and widget reuse; this skill owns containers/structure only |

### FIX-shaped — a straightforward finding, usually one-line remediation

- `<section>` with no accessible name and no outline-worthy content → `div` (L4)
- `<h1>` count ≠ 1 for a page/route, or a skipped rank in sequence → renumber (L3)
- Tabular data rendered as a div grid → `<table>` with scoped `<th>` (L5)
- `order`/`flex-direction: row-reverse`/explicit `grid-area` diverging from a
  DOM sequence that carries meaning (a form, a step flow, a title-then-price
  pairing) → reorder the markup instead of the CSS (L14)
- A truncating element (`text-overflow: ellipsis` + `overflow: hidden`) on a
  flex/grid item with no `min-width: 0` on that item or an ancestor → add it (L12)
- A margin/padding/gap literal that isn't `var(--space-*)`, with no reason it
  can't be the nearest scale value → replace with the token (L18)
- Missing skip link on a page with a nontrivial header/nav before `<main>` (L15)
- Unlabeled second `<nav>`/duplicate landmark on the same page → add
  `aria-label` (L2)

### JUDGMENT — needs more context before it's a finding at all

- A `<section>` *does* have an accessible name — confirm the content is
  actually outline-worthy (L4) before treating the label as sufficient;
  a labeled `<section>` wrapping trivial content is still a soft L4 finding
- Nested flex simulating a grid (L8) — only a finding if column alignment
  visibly depends on content length; a genuinely single-axis layout that
  happens to wrap is not a defect
- Fixed pixel width (L9) — only a finding if the value is a guess rather
  than a deliberate constraint (Board.svelte's SVG intrinsic sizing, an
  icon's fixed size) is not a finding
- Device-width `@media` query on a shared `lib/` component (L10) — confirm
  the component is actually reused in more than one visual context before
  flagging; App.svelte's own page-level `@media` breakpoint (added in the
  layout redesign) is correct use per L10, not a finding — the pattern to
  flag is a `@media` query *inside* an individual `lib/*.svelte` component
- Repeated bespoke vertical-rhythm CSS across multiple components (L11) —
  only a finding once the same spacing pattern is duplicated in ≥2 places
  (the shared `.panel` class already absorbed the five-way card-style
  duplication in the layout redesign; look for anything new re-duplicating
  it going forward); a single occurrence is not yet "soup"
- Multiple wrapper `<div>`s (L6) — only a finding when a single element
  (or one fewer wrapper) could achieve the identical visual result; a
  wrapper needed for a distinct styling hook or layout context is not soup
- z-index value (L13) — only a finding when it's part of an escalating,
  undocumented ladder; a single deliberate value with no competing context
  is not a defect (there is no `--z-*` scale yet, so any hit is worth a
  first read even without an escalating ladder — a single one may still be
  worth naming a scale for once a second one appears)

### Not a finding at all

- A plain `<div>` used as a styling hook with no semantic candidate that fits
  better (L1/L4) — this is the correct default, not a compromise
- A `<header>`/`<footer>` nested inside `<article>`/`<section>` with no
  landmark role — this is spec-correct behavior, not a bug (L2)
- Any pattern listed under `layout-philosophy.md`'s "Concepts we document
  but do not check"

Be conservative on JUDGMENT items — when genuinely unsure whether reuse
context or intent justifies the pattern, read one more call site before
opening the finding.

---

## 4. Quick audit checklist

When reviewing a single file or component by hand, scan in this order:

1. ❌ `<div>`/`<span>` where the content's actual meaning has a better
   semantic element (L1)
2. ❌ Missing or duplicate `<main>`; a second `<nav>` with no distinguishing
   label (L2)
3. ❌ More than one `<h1>`, or a skipped heading rank in document order (L3)
4. ❌ `<section>` with no accessible name; `<section>` used as a pure
   styling hook (L4)
5. ❌ Tabular data as a div grid; an `{#each}`-rendered collection not
   wrapped in `<ul>`/`<ol>`/`<li>` (L5)
6. ❌ Three or more nested wrapper elements achieving one visual effect (L6)
7. ❌ `flex`/`grid` applied where normal flow already does the job (L7)
8. ❌ Nested flex hand-simulating a two-axis grid (L8)
9. ❌ A fixed pixel width/height standing in for intrinsic sizing
   (`min-content`, `clamp()`, `minmax()`) with no deliberate reason (L9)
10. ❌ A device-width `@media` query on a shared `lib/` component instead of
    a container query (L10)
11. ❌ Bespoke vertical-rhythm/spacing CSS duplicated across ≥2 components
    instead of the shared `.panel` class or a named primitive (L11)
12. ❌ A truncating/`overflow: hidden` element inside a flex/grid item with
    no `min-width: 0` (L12)
13. ❌ An escalating, undocumented z-index ladder (L13)
14. ❌ `order`/`flex-direction: row-reverse`/explicit grid placement
    diverging from a DOM sequence that carries meaning (L14)
15. ❌ No skip link ahead of a nontrivial header/nav on a page (L15)
16. ❌ A heading whose key word is buried at the end rather than front-loaded (L16)
17. ❌ Elements visually grouped by spacing alone with no structural home
    (`<fieldset>`, a wrapping container, `aria-labelledby`) (L17)
18. ❌ Misaligned edges between adjacent elements; an arbitrary spacing value
    off the repo's 4pt grid (L18)
