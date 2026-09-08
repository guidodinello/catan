# Layout & Structure Philosophy

> **Purpose**: the *why* behind every finding the `/layout-structure` skill
> reports. The mechanical patterns (greps, classification rules) live in
> [`layout-checks.md`](layout-checks.md); this file is what lets a classifier
> make a judgment call on a finding that no grep can decide — and what a
> human reaches for when designing a new screen from scratch.
>
> **Who reads this**: the Step 3 classify agent (alongside
> `layout-checks.md`), and anyone building a new page or component who wants
> the reasoning, not just the rule.

The goal is structure that carries meaning — for sighted users scanning
visually, for assistive-technology users navigating the DOM, and for anyone
reading the markup six months from now. A well-structured page produces few
findings. This skill is **web-only**, scoped to `web/src` (Svelte 5 + Vite,
plain CSS).

**This is not a complete layout/a11y/UX skill on its own — but in this
project, nothing else fills the gap either.** The sibling skills referenced
below by name (`accessibility-review`, `ui-ux-review`,
`react-composition-audit`) exist in the project this skill was adapted
from, not in this one. See `SKILL.md`'s `## Boundaries` section: a
candidate matching one of their categories is reported as **Uncovered**,
not judged here and not silently dropped. Read that section before treating
any finding below as this skill's to make.

## Lineage

Nothing here is original. The normative claims come from the WHATWG and W3C
specs; the mechanics from MDN and web.dev; the compositional philosophy from
Every Layout, CUBE CSS, and the "browser's mentor" line; the perceptual
claims from Nielsen Norman Group eye-tracking research. Layer matters here
more than in a pure-code skill: don't blur "the spec says" with "eye-tracking
suggests" — they carry different kinds of authority.

| Principle | Layer | Origin |
|---|---|---|
| L1 — markup before CSS | Practice | general semantic-HTML discipline |
| L2 — landmarks, one `main` | Normative + Practice | W3C WAI Page Structure tutorial |
| L3 — the outline is flat and explicit | Normative | WHATWG HTML §4.3.11; MDN blog on UA style removal |
| L4 — `section` vs `article` vs `div` | Normative | WHATWG HTML §4.3 (`section`, `article`) |
| L5 — structure-carrying elements | Practice | MDN *Structuring documents* |
| L6 — wrapper budget / div soup | Practice | MDN *Structuring documents* ("divs so convenient... clutter") |
| L7 — flow first | Philosophy | buildexcellentwebsit.es; Every Layout *Axioms* |
| L8 — 1D → flex, 2D → grid | Practice | web.dev *Learn CSS: Layout* |
| L9 — intrinsic over fixed | Philosophy | Every Layout *Axioms* (measure as exemplar) |
| L10 — content-out breakpoints, container queries | Practice + Philosophy | MDN *Container queries*; Every Layout *Composition* |
| L11 — layout primitives & composition | Philosophy | Every Layout *Composition*; CUBE CSS |
| L12 — overflow / `min-width:0` | Practice | common flex/grid implementation trap, cross-checked against MDN box-sizing model |
| L13 — positioning & stacking context | Practice | MDN structuring/positioning fundamentals |
| L14 — source order is reading order and tab order | Normative | WCAG 2.2 Understanding 1.3.2; Léonie Watson, *Flexbox and the keyboard navigation disconnect* |
| L15 — skip links / bypass blocks | Normative | WCAG 2.2 SC 2.4.1 (Bypass Blocks) |
| L16 — structural hierarchy first | Empirical | NN/g *The Layer-Cake Pattern*, *F-Shaped Pattern* |
| L17 — grouping in markup, not only spacing | Empirical | NN/g *The Principle of Proximity* |
| L18 — alignment and scale consistency | Empirical + repo convention | NN/g *Visual Hierarchy*; `web/src/app.css`'s `--space-*` scale |

---

## L1 — Structure Is the Meaning; Markup Before CSS

**The idea.** Pick the element by what the content *is*, then style it.
Reaching for `<div>` + CSS first and asking "what should this mean?" second
produces a page whose only structure lives in a stylesheet — invisible to a
screen reader, a search engine, or the next developer reading the JSX
without the CSS open. MDN's structuring-documents guidance is blunt about
the failure mode: divs are "so convenient to use that it's easy to use them
too much. As they carry no semantic value, they just clutter your HTML
code."

**Bad** — a card that only *looks* like a card:

```html
<div class="card">
  <div class="card-title">Cotton overshirt, washed clay</div>
  <div class="card-price">$148</div>
</div>
```

**Good** — the same visual result, structure-first:

```html
<article class="card">
  <h3 class="card-title">Cotton overshirt, washed clay</h3>
  <p class="card-price">$148</p>
</article>
```

Nothing here changes a single pixel. The only difference is that a screen
reader, a browser's reader mode, and a search engine can now tell what the
block *is*.

**Priority: High** — this is the premise the other seventeen principles sit
on top of. A codebase that gets L1 wrong produces findings in L2–L6 by
construction.

---

## L2 — Landmarks: One `main`, Name the Repeats

**The idea.** `<header>`, `<nav>`, `<main>`, `<aside>`, `<footer>` map to
ARIA landmark roles (`banner`, `navigation`, `main`, `complementary`,
`contentinfo`) — but only under specific conditions the spec is explicit
about. The W3C WAI Page Structure tutorial states it exactly:

> "If the `<header>` element is used inside `<article>` and `<section>`
> elements, it is not associated to those elements. It does not get the
> WAI-ARIA `banner` role and does not have special behavior in assistive
> technologies."

The identical sentence applies to `<footer>` / `contentinfo`. So a
`<header>` inside a card component is just a grouping wrapper — it is not a
landmark, and treating it as one (e.g. expecting a screen reader's
landmark-navigation shortcut to reach it) is a mistake in either direction:
don't rely on it as a landmark, and don't worry that a card's internal
`<header>` is "polluting" page-level navigation. It isn't.

Two structural rules that follow:

- Exactly one `<main>` per page/route, a direct child of `<body>` (or the
  route's root element) — never nested inside another landmark.
- A second `<nav>` on the same page (e.g. primary nav + a table-of-contents
  nav) needs a distinguishing accessible name — `aria-label="Primary"` /
  `aria-label="On this page"` — or assistive-tech users get two identical,
  unlabeled "navigation" entries with no way to tell them apart.

**Bad** — two navs, indistinguishable to a screen reader:

```html
<nav>...</nav>
<!-- ...page content... -->
<nav>...</nav>
```

**Good**:

```html
<nav aria-label="Primary">...</nav>
<!-- ...page content... -->
<nav aria-label="On this page">...</nav>
```

**Boundary.** The WCAG success-criterion tagging for a missing/duplicated
landmark would be a dedicated accessibility skill's territory, if this
project had one (see `SKILL.md`'s Boundaries — report it as Uncovered
rather than tagging it here). This skill owns *which* landmark to reach for
and the page-level `<main>`/nesting rule regardless.

**Priority: High** — landmark navigation is how many screen-reader users
skip the entire page in one keystroke; getting it wrong doesn't degrade the
experience, it removes the shortcut entirely.

---

## L3 — The Outline Is Flat and Explicit

**The idea.** This is the correction most engineers get wrong, so it's worth
being precise about what actually changed and when.

There were, historically, two different ideas bundled under "HTML5
outline," and only one of them was ever removed:

1. **The sectioning-content-implies-heading-rank outline** — the idea that
   nesting an `<h1>` inside `<section>`, `<article>`, `<aside>`, or `<nav>`
   implicitly demoted it to act like an `<h2>`, and nesting it two levels
   deep made it act like an `<h3>`, and so on. **No browser ever implemented
   this as a document-structure feature** (it never changed what a screen
   reader announced, and it never generated an actual table of contents).
   The *only* place it ever manifested was visually, via a UA stylesheet
   rule that shrank an `<h1>`'s font-size the deeper it was nested:

   ```css
   /* the old default UA rule, applied where x is :is(article, aside, nav, section) */
   x h1 { font-size: 1.50em; }
   x x h1 { font-size: 1.17em; }
   x x x h1 { font-size: 1.00em; }
   ```

   This visual-only effect is what's being removed now: Firefox 140, Chrome
   (deprecation warnings since 136, removed mid-2025), and Safari 26.2 have
   all dropped these UA rules. A page whose heading hierarchy was only ever
   "correct" because a nested `<h1>` happened to render at `<h2>` size will,
   as of these browser versions, render every `<h1>` at full size regardless
   of nesting depth.

2. **The spec's current outline concept** (WHATWG HTML §4.3.11, "Headings
   and outlines") is a different, narrower thing: it still exists, still
   describes an algorithm, and its stated purpose is explicit — "the outline
   should be used for generating document outlines, for example when
   generating tables of contents." That is a tool concept (a documentation
   generator building a table of contents), not a claim that browsers apply
   it to change what gets rendered or announced. The spec also defines a
   distinct, current `headingoffset` content attribute ("allows authors to
   offset heading levels for descendants," valued 0–8) — an explicit,
   opt-in mechanism for a tool or author to declare an offset, not an
   implicit one inferred from DOM nesting.

The practical rule survives both facts intact, and doesn't depend on which
version of "outline" you mean:

- **One `<h1>` per page** (or per route, in an SPA).
- **No skipped ranks** — an `<h2>` should not be followed directly by an
  `<h4>` (skipping is acceptable only when *closing* a subsection: an `<h4>`
  can be followed by a new `<h2>`).
- **Style headings explicitly; never rely on default browser sizing to
  convey rank** — this was already good practice, and the 2025 UA-style
  removal makes it load-bearing. If a design relies on `<h1>` looking
  smaller inside a card, that's a CSS rule the codebase must write, not an
  emergent effect of nesting:

  ```css
  :where(h1) { font-size: 2em; margin-block: 0.67em; }
  ```

**Bad** — heading rank chosen by desired size, relying on the (now-removed)
UA default:

```html
<section>
  <section>
    <h1>Cotton overshirt</h1> <!-- "worked" only because nested h1 rendered small -->
  </section>
</section>
```

**Good** — rank chosen by document position, size set explicitly:

```html
<article>
  <h2 class="text-[19px] font-medium">Cotton overshirt</h2>
</article>
```

**Boundary.** A dedicated accessibility skill would flag "heading levels
that skip (h1→h3)" as a WCAG 1.3.1/4.1.2 finding, if one existed here — this
skill owns the *technique* (rendering/asserting the outline, explaining why
nesting never should have mattered) regardless of whether the compliance
tag ever gets applied.

**Priority: High** — an incorrect heading structure breaks screen-reader
document navigation (which jumps by heading), and after the 2025 browser
changes it now also breaks the *visual* hierarchy that many teams were
accidentally relying on the outline algorithm to produce.

---

## L4 — `section` vs `article` vs `div`

**The idea.** Three container elements, three different jobs, and the spec
states the decision rule directly rather than leaving it to taste.

`div` is the correct *default* — not a fallback to feel guilty about:

> "The `section` element is not a generic container element. When an element
> is needed only for styling purposes or as a convenience for scripting,
> authors are encouraged to use the `div` element instead."

`section` earns its place only when the content would belong in a document
outline/table of contents on its own:

> "A general rule is that the `section` element is appropriate only if the
> element's contents would be listed explicitly in the document's outline."

And even when `section` is the right choice structurally, it only becomes a
`region` landmark — reachable via a screen reader's landmark navigation — if
it has an accessible name (Scott O'Hara's summary of this: "To expose a
`<section>` element's implicit `region` role, it must be given an accessible
name"). An unlabeled `<section>` is inert as a landmark; it's just a `div`
with extra steps.

`article` is reserved for content that is independently distributable or
reusable on its own — a blog post, a forum comment, a single feed item that
would make sense syndicated out of context.

**Bad** — `section` used as a styling hook, no name, no outline-worthy content:

```html
<section class="flex items-center gap-2">
  <Icon name="chevron-right" />
  <span>See all</span>
</section>
```

**Good** — this is a styling/grouping wrapper, so `div` is correct:

```html
<div class="flex items-center gap-2">
  <Icon name="chevron-right" />
  <span>See all</span>
</div>
```

**Good** — `section` earning its place, with a name:

```html
<section aria-labelledby="recs-heading">
  <h2 id="recs-heading">Picked for you</h2>
  <!-- ...feed items... -->
</section>
```

**Good** — `article` for a self-contained, reusable unit:

```html
<article class="card">
  <h3>Cotton overshirt, washed clay</h3>
  <p>$148</p>
</article>
```

**Priority: High** — no other skill in this project greps for `<section>`
or `<article>` at all (there is no accessibility skill here to have that
gap in the first place). This skill is the only place this rule lives for
`web/src`.

---

## L5 — Structure-Carrying Elements

**The idea.** Some HTML elements don't just group content — they encode a
relationship that a `<div>`-based equivalent throws away. A list conveys
"these are members of a set, in this order, and there are N of them" to a
screen reader (which announces "list, 5 items"). A `<table>` conveys row/
column relationships that let a screen reader announce a cell's row and
column headers when navigating it. A `<div>`-based data grid conveys none of
that unless every one of those semantics is manually re-implemented with
ARIA — and it usually isn't.

Rules of thumb:

- Genuinely tabular data (rows *and* columns both carry meaning — a
  size/price grid, an order history table) → `<table>` with `<th scope="col">`
  / `<th scope="row">`, not a CSS grid of `<div>`s.
- A repeated, homogeneous collection (a feed, a list of tags, a nav's link
  list) → `<ul>`/`<ol>`/`<li>`, not a `<div>` per item. If the visual design
  requires removing list bullets (`list-style: none`), the semantics survive
  the styling — Safari's older bug that dropped the implicit `role="list"`
  under `list-style:none` is exactly why an explicit `role="list"` is
  sometimes added back defensively, but verify current browser behavior
  before assuming it's still needed.
- An image with a caption, or any self-contained media + its explanation →
  `<figure>` + `<figcaption>`, which keeps the two associated for assistive
  tech even when the DOM is later reordered.

**Bad** — a price table as a div grid:

```html
<div class="grid grid-cols-3">
  <div>Size</div><div>Stock</div><div>Price</div>
  <div>S</div><div>12</div><div>$148</div>
</div>
```

**Good**:

```html
<table>
  <thead>
    <tr><th scope="col">Size</th><th scope="col">Stock</th><th scope="col">Price</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">S</th><td>12</td><td>$148</td></tr>
  </tbody>
</table>
```

**Priority: Medium** — most feed/list UIs in this codebase already use
`<ul>`/`<li>` correctly; the higher-value catch is a hand-rolled data grid
standing in for a real `<table>`.

---

## L6 — Wrapper Budget / Div Soup

**The idea.** Every wrapper element should earn its place: a styling hook a
utility class needs to attach to, a layout context (a flex/grid container),
or genuine semantics. A component that nests five `<div>`s to achieve one
visual effect isn't wrong in the way L1–L5 describe — none of those divs
need to be a different element — but it's a maintenance cost with no payoff,
and it's worth counting.

**Boundary.** This is a different axis from the (uncovered, in this
project) "don't flag a plain layout `<div>` for missing a role" question —
that's about *whether* a div needs a name; L6 is about *how many* divs a
given visual effect actually requires. A div that exists purely to apply
`class="flex gap-2"`-equivalent styling isn't a naming concern at all, but
three of them nested to achieve what one flex container with `flex-wrap`
could do is an L6 finding regardless.

**Bad** — three wrappers where one would do:

```html
<div class="row-wrapper">
  <div class="row-inner">
    <div class="row-content">
      <span>Cotton overshirt</span>
    </div>
  </div>
</div>
```

**Good**:

```html
<div class="row-content">
  <span>Cotton overshirt</span>
</div>
```

**Priority: Low** — a code-quality and maintainability concern, not a
correctness one. Flag it, don't block on it.

---

## L7 — Flow First; Be the Browser's Mentor, Not Its Micromanager

**The idea.** Normal document flow already solves most layout problems for
free — text wraps, blocks stack, inline elements flow. Reach for `flex` or
`grid` when flow genuinely can't express the relationship (aligning items
across an axis, distributing space), not by default on every container. The
summary formulation from buildexcellentwebsit.es: "Be the browser's mentor,
not its micromanager." Every Layout frames the same idea as designing
*algorithmically* rather than pixel-by-pixel — "relinquish control to the
algorithms (like text wrapping) browsers use to lay out web pages
automatically," because (their phrase) "designing for the web is designing
without seeing:" you cannot enumerate every viewport, font-size setting, and
zoom level a real visitor will combine.

**Bad** — flex reaching for a job flow already does:

```css
.paragraph-wrapper { display: flex; flex-direction: column; }
```

**Good** — a `<p>` already stacks and wraps without help:

```css
.paragraph-wrapper { /* no display override needed */ }
```

**Priority: Low** — a style/cleanliness signal, and one of the few in this
file that's more a smell than a defect; report it, don't block on it.

---

## L8 — One Dimension → Flex, Two Dimensions → Grid

**The idea.** Choose the layout module by the shape of the relationship
you're expressing, not habit. Flexbox is for a single axis — "layout across
a single axis, either horizontally or vertically" (web.dev) — with items
free to grow, shrink, and wrap along that one line. Grid is for genuinely
two-dimensional placement — rows *and* columns need independent control at
once. A common tell that flex is being pushed past its natural job: nested
flex containers simulating a grid (a flex-row of flex-columns, hand-aligning
things that should share a grid's column tracks).

**Bad** — a card grid built from nested flex, columns never actually align:

```css
.grid { display: flex; flex-wrap: wrap; }
.grid > * { flex: 1 1 240px; }
```

**Good** — the two-dimensional relationship, stated directly:

```css
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
}
```

**Priority: Medium** — the flex-wrap version usually *looks* fine until
content lengths vary, at which point columns stop aligning and the "grid"
was never actually one.

---

## L9 — Intrinsic Over Fixed

**The idea.** Prefer sizing that lets the browser compute the right answer
from content and available space over sizing that hard-codes a guess.
`ch` units for measure (line length in characters, independent of font),
`min-content`/`max-content`/`fit-content()` for "as small/large as the
content needs," `clamp(min, preferred, max)` for fluid values that still
have a floor and ceiling, `minmax()` inside `auto-fit`/`auto-fill` grid
tracks for a card grid that reflows without a breakpoint. Every Layout's
framing: measure is "an innately algorithmic approach... because the
outcome is predicated on a calculation you permit the browser to make,"
rather than a number you had to guess and will be wrong for some viewport.

**Bad** — a fixed pixel width that's either too wide or too narrow depending
on content and viewport:

```css
.card-grid { width: 960px; }
.card { width: 300px; }
```

**Good**:

```css
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 18rem), 1fr));
  gap: 1rem;
}
```

**Priority: Medium** — fixed sizing is the most common source of the
overflow bugs L12 describes; catching it here is cheaper than debugging the
symptom later.

---

## L10 — Content-Out Breakpoints; Container Queries for Components

**The idea.** Break where the *content* breaks (where a line gets
uncomfortably long, where a card's internal layout stops fitting), not at
fixed device widths chosen in the abstract. And for anything that's a
reusable component rather than a page-level region, prefer a **container
query** over a media query — a card placed in a narrow sidebar and the same
card placed in a wide main column need different internal layouts *based on
their own slot*, not on the viewport, which knows nothing about where the
component landed:

```css
.card { container-type: inline-size; }

@container (width > 24rem) {
  .card { grid-template-columns: auto 1fr; }
}
```

Media queries remain correct for page-level layout decisions (does the
whole page get a sidebar) — Every Layout's point isn't "never use media
queries," it's that per-component breakpoints belong to the component, and a
device-width media query baked into a shared component silently breaks the
moment that component is reused somewhere else on the page.

**Bad** — a card component's internal layout keyed to viewport width:

```css
.card { display: block; }
@media (min-width: 768px) { .card { display: grid; grid-template-columns: auto 1fr; } }
```

**Good** — keyed to the card's own available space:

```css
.card { container-type: inline-size; display: block; }
@container (width > 24rem) { .card { display: grid; grid-template-columns: auto 1fr; } }
```

**Boundary.** This skill owns the responsive *mechanism* (container query
vs. media query, and at what level each belongs). *Where* key content
should land once laid out (a scanning-pattern/F-Z-pattern heuristic) is a
perceptual-design judgment this project has no dedicated skill for — report
it as Uncovered rather than guessing at it here.

**Priority: Medium** — container queries are a genuine gap; a device-width
media query hard-coded into a reusable `lib/` component is exactly the kind
of finding that won't show up until the component gets reused in a
different-sized slot.

---

## L11 — Layout Primitives & Composition

**The idea.** Rather than writing a bespoke page-level grid for every
screen, compose layouts from a small set of single-purpose, content-agnostic
primitives — the Every Layout vocabulary: **Stack** (consistent vertical
rhythm between children), **Cluster** (wrapping horizontal groups with
consistent gaps — a tag list, a button row), **Sidebar** (one fixed-width
column beside one flexible one that wraps to stacked below a threshold),
**Switcher** (children that lay out horizontally until a container
threshold, then stack), **Cover** (centers one piece of content vertically
within a container, with optional header/footer pinned). Every Layout's
framing: a primitive is "meaningful only through composition" — like a
boolean `true` carries no meaning until used in a larger expression — and
should not encode what content it holds or which page it appears on.

**Boundary.** This is distinct from leaf-component reuse (widgets like
buttons, inputs, badges, and whether a call site should reuse one instead of
hand-rolling it) — this project has no dedicated skill for that, so a
candidate matching it is Uncovered. L11 is about *layout* wrappers
specifically — the Stack/Cluster/Sidebar family, not design-system leaf
components.

**Bad** — a bespoke, one-off page grid reinvented per screen:

```css
.feed-screen-layout { display: flex; flex-direction: column; }
.feed-screen-layout > * + * { margin-top: 16px; }
.browse-screen-layout { display: flex; flex-direction: column; }
.browse-screen-layout > * + * { margin-top: 12px; } /* same idea, different number, no shared name */
```

**Good** — one named, reusable primitive:

```css
.stack { display: flex; flex-direction: column; }
.stack > * + * { margin-top: var(--stack-space, 1rem); }
```

```html
<div class="stack" style="--stack-space: 1rem">
  <FeedRow />
  <FeedRow />
</div>
```

**Priority: Low** — an architectural improvement, not a correctness defect;
worth a `low`/`nit` finding when a screen reinvents vertical rhythm from
scratch rather than reusing an existing pattern.

---

## L12 — Overflow and the `min-width: 0` Trap

**The idea.** Flex and grid items default to `min-width: auto` (and
`min-height: auto`), which means an item won't shrink smaller than its
content's intrinsic minimum size — even if you told it to `flex: 1` or gave
it `overflow: hidden` with `text-overflow: ellipsis`. A long, unbreakable
string (a long item name, an email address) inside a flex/grid item will
blow out the layout instead of truncating, because the truncation CSS never
gets a chance to apply — the box refuses to shrink below the text's natural
width in the first place. The fix is almost always one declaration on the
item that needs to shrink: `min-width: 0` (or `min-height: 0` for a column
axis).

**Bad** — `truncate` present, but silently inert:

```html
<div class="flex gap-2">
  <span class="flex-1 truncate">A very long store name that should truncate</span>
  <span>$148</span>
</div>
```

**Good**:

```html
<div class="flex gap-2">
  <span class="min-w-0 flex-1 truncate">A very long store name that should truncate</span>
  <span>$148</span>
</div>
```

**Priority: Medium** — this is a real bug class (content silently
overflowing its container), not a style preference, and it's invisible until
someone's data happens to be long enough to trigger it — which real product
data (a long store or item name) reliably will be.

---

## L13 — Positioning and Stacking-Context Discipline

**The idea.** `position: absolute`/`fixed` removes an element from normal
flow — its neighbors act as if it isn't there, which is exactly the L14
concern from the opposite direction (visual placement diverging from DOM
placement). `z-index` only orders elements *within the same stacking
context*; a `z-index: 9999` on a deeply nested element does not guarantee it
appears above a sibling of an ancestor if that ancestor itself created a new
stacking context (any element with `position` + `z-index`, `opacity < 1`,
a `transform`, or `filter` starts one). An escalating ladder of
ever-larger `z-index` values across a codebase is usually a symptom of
fighting stacking contexts rather than solving the actual overlap.

**Bad** — an arbitrary, ever-escalating z-index with no context:

```css
.dropdown { z-index: 9999; }
.modal { z-index: 99999; }
.toast { z-index: 999999; }
```

**Good** — a small, deliberate scale tied to a documented stacking order:

```css
:root { --z-dropdown: 10; --z-modal: 20; --z-toast: 30; }
.dropdown { z-index: var(--z-dropdown); }
.modal { z-index: var(--z-modal); }
.toast { z-index: var(--z-toast); }
```

**Priority: Medium** — usually surfaces as "why is this element hidden
behind that one even though its z-index is higher," a debugging cost that a
small documented scale avoids entirely.

---

## L14 — Source Order Is Reading Order and Tab Order

**The idea.** This is the seam where CSS layout meets accessibility most
directly, and it's a gap none of the sibling skills currently cover.
`order`, `flex-direction: row-reverse`, explicit `grid-area`/`grid-column`/
`grid-row` placement, and absolute positioning can all make an element
*appear* somewhere different from where it sits in the DOM. Tab order and
screen-reader reading order both follow the DOM, not the visual position —
so a visual reorder that isn't also a DOM reorder creates a page where
sighted keyboard users tab through elements in an order that doesn't match
what they see, and screen-reader users hear content in a sequence that
doesn't match the visual story.

WCAG 2.2's Understanding for **1.3.2 Meaningful Sequence** states the
success criterion directly: "When the sequence in which content is
presented affects its meaning, a correct reading sequence can be
programmatically determined" — and names exactly this failure mode: "CSS is
used to position a navigation bar, the main story on a page, and a side
story. The visual presentation of the sections does not match the
programmatically determined order" is only acceptable when "the meaning of
the page does not depend on the order of the sections." The moment order
*does* carry meaning — a form's fields, a step-by-step flow, a card's
title-then-price relationship — a CSS-only reorder is a Bypass Blocks/
Meaningful Sequence violation, formally cataloged as failure **F1**: "Failure
... due to changing the meaning of content by positioning information with
CSS."

Léonie Watson's practitioner framing of the same problem, from the
accessibility side: "With flexbox it is possible to display content in a
different visual order, without changing the DOM order," and when a
keyboard user tabs through it, "there is a disconnect between the visual
order and the keyboard navigation (DOM) order" — ranging from "mildly
awkward" to "horribly unusable" depending on how far the reorder goes.

**Bad** — a form whose visual field order is reversed from its DOM order:

```html
<form class="flex flex-col">
  <input name="email" class="order-2" placeholder="Email" />
  <input name="name" class="order-1" placeholder="Name" />
</form>
<!-- Tab order visits email, then name — opposite of what's on screen -->
```

**Good** — DOM order and visual order agree; reorder the markup, not the CSS:

```html
<form class="flex flex-col">
  <input name="name" placeholder="Name" />
  <input name="email" placeholder="Email" />
</form>
```

**Boundary.** Neither `1.3.2 Meaningful Sequence` nor `2.4.3 Focus Order`
has any other skill in this project claiming them — this skill owns the
finding outright, and should say so explicitly in its report rather than
silently ignoring the gap.

**Priority: High** — this is the skill's headline finding class: a defect
that's invisible from a screenshot, invisible to a mouse user, and only
surfaces for keyboard and screen-reader users — exactly the population most
likely to be missed by a purely visual review.

---

## L15 — Skip Links / Bypass Blocks

**The idea.** A keyboard or screen-reader user landing on any page has to
traverse every element that comes before the main content on every single
page load — the full nav, any promotional banner, any secondary chrome —
unless the page offers a mechanism to bypass repeated blocks. WCAG 2.2's
**2.4.1 Bypass Blocks** exists precisely for this. The common, low-cost
implementation is a visually-hidden "Skip to main content" link as the very
first focusable element, which becomes visible on focus and jumps straight
to `<main>`:

```html
<a href="#main-content" class="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:p-2">
  Skip to main content
</a>
<!-- ...header, nav... -->
<main id="main-content">...</main>
```

**Boundary.** This is ambiguous territory — arguably a dedicated
accessibility skill's, if one existed here — but since no such skill exists
in this project, this skill claims it and should say so in its report so
the gap doesn't silently persist under an assumption that "someone else
already checks this."

**Priority: Medium** — a real WCAG 2.4.1 failure mode, but lower-severity
than L14: it's an efficiency cost for repeat navigation, not a blocked task.

---

## L16 — Structural Hierarchy First, Visual Hierarchy Second

**The idea.** Headings aren't just an accessibility requirement — they are
where sighted users' eyes actually land first when scanning a page.
Nielsen Norman Group's eye-tracking research describes the efficient pattern
directly: the **layer-cake pattern** — "fixations made mostly on the page's
headings and subheadings, with deliberate occasional fixations on the (body)
text in between" — is "by far the most effective way to scan pages," because
"it ensures that users will find the information they are looking for (if it
is present on the page)." The failure mode it contrasts against is the
**F-shaped pattern**: users sweep the first couple of lines, then trail down
the left edge, reading less and less — which NN/g frames not as a user
preference but as "the default pattern when there are no strong cues to
attract the eyes towards meaningful information." A page that produces an
F-pattern isn't failing its users; it's exposing that its heading structure
never gave them anything better to do.

The practical implication is that getting the *document* structure right
(L1–L4) is not a separate concern from getting the page *scannable* — a
correct heading hierarchy is simultaneously an accessibility requirement and
the single strongest lever for visual scannability. Front-load: put the
important word first in each heading, since that's what a scanning eye reads
before deciding whether to keep going.

**Boundary.** *Where* on the page key content should sit (a scan-pattern
placement judgment) has no dedicated skill in this project — Uncovered if
it comes up. L16 owns only the underlying heading-structure mechanism that
makes any scan pattern possible in the first place, not spatial placement
of already-present content.

**Priority: Medium** — a page can pass every WCAG heading check and still
scan poorly if headings don't front-load their key word; this is a quality
signal, not a hard defect, unless it coincides with an L3 finding.

---

## L17 — Grouping Must Exist in the Markup, Not Only in the Spacing

**The idea.** Gestalt's principle of proximity predicts, reliably, that
users perceive elements placed close together as belonging to one group —
NN/g's summary: "Items close together are likely to be perceived as part of
the same group... Proximity can even override other visual cues like color
or shape similarity." That perceptual grouping is a purely *visual* effect —
spacing, not markup. If the DOM structure disagrees with what spacing
implies (the "grouped" elements aren't actually nested together, or a
screen-reader user encounters them in an order that doesn't reflect the
grouping a sighted user perceives), assistive-technology users get a
functionally different page: the visual grouping that guides a sighted
user's understanding of "these three things go together" simply isn't
available to them.

The fix isn't more spacing tricks — it's making the grouping structural: a
`<fieldset>`/`<legend>` for a group of related form controls, a wrapping
`<li>` or `<div>` that actually contains the elements being visually
grouped rather than three CSS-margin-adjacent siblings, or `aria-labelledby`
tying a heading to the region it labels.

**Bad** — three form fields visually grouped by spacing alone, no
structural grouping:

```html
<form class="flex flex-col gap-6">
  <div class="flex flex-col gap-1">
    <label>Street</label><input />
  </div>
  <div class="flex flex-col gap-1">
    <label>City</label><input />
  </div>
  <!-- visually close, but "shipping address" as a concept exists nowhere in the DOM -->
</form>
```

**Good**:

```html
<form>
  <fieldset class="flex flex-col gap-1">
    <legend>Shipping address</legend>
    <label>Street</label><input />
    <label>City</label><input />
  </fieldset>
</form>
```

**Boundary.** The concrete px thresholds for how much spacing implies
grouping (`≤8px` label-value, `≥16px` section-to-section) are a UX-judgment
question this project has no dedicated skill for — don't invent a specific
number here if it comes up; note it as Uncovered instead. This principle
owns only the claim that grouping needs a structural home, not the specific
numbers.

**Priority: Medium** — most relevant on forms and settings screens, where
"which fields belong together" is exactly the information a sighted user
gets for free from spacing and a screen-reader user needs a `<fieldset>` or
equivalent to get at all.

---

## L18 — Alignment, and Consistency With the Existing Spacing Scale

**The idea.** Two genuinely unowned gaps, deliberately kept narrow:

**Alignment.** No skill in this project has alignment guidance. Misaligned
edges — a card's internal padding that doesn't line up with its neighbor's,
text baselines that don't match across an inline group — are a low-level
but persistent visual-quality signal worth this skill owning outright:
edges of related elements (card padding, icon + label baselines, form
field left edges) should align to a shared line, not approximately.

**Spacing — consistency only, not a new scale.** `web/src/app.css` already
defines the spacing convention, established during the layout redesign
that unified the five sidebar panels' previously copy-pasted CSS: a
4px-base scale as CSS custom properties, `--space-1: 0.25rem` (4px) through
`--space-6: 2rem` (32px), with the comment "the only spacing values used
anywhere after this." This skill's job is to flag spacing that breaks that
existing scale (a hardcoded `margin: 0.8rem` sitting next to a codebase
that otherwise uses `var(--space-4)`), **not** to propose a different or
additional scale. Inventing a new spacing system here would conflict
directly with the tokens the layout redesign just established — own
consistency with what exists, not authorship of what should exist.

**Bad** — a hardcoded value bypassing the token scale:

```css
.foo { margin-top: 13px; display: flex; flex-direction: column; gap: 7px; }
```

**Good** — same visual intent, on-scale:

```css
.foo { margin-top: var(--space-3); display: flex; flex-direction: column; gap: var(--space-2); }
```

**Boundary.** The Gestalt proximity px thresholds (`≤8px`/`≥16px`) are a
UX-judgment question this project has no dedicated skill for — Uncovered
if it comes up. This principle is about scale-consistency, not about how
much space implies grouping (that's L17's question).

**Priority: Low** — a polish signal. Flag it; it rarely blocks a user, but
an arbitrary value breaking an otherwise-consistent 4pt grid is a real,
cheap-to-fix inconsistency.

---

## Concepts We Document but Do Not Check

These are real, and worth knowing about, but outside what this skill's
greps or judgment passes can reliably catch — background reading, not
findings. Never open a finding against any of these; if one comes up, note
it as out of scope rather than guessing.

### Subgrid

`grid-template-columns: subgrid` lets a nested grid inherit its parent's
track sizing — solves the classic "columns in nested cards don't line up"
problem — but the codebase would need a specific case in front of it to
judge whether subgrid vs. an alternative is the right call; too contextual
for a general rule.

### CSS anchor positioning (`anchor-name` / `position-anchor`)

A newer mechanism for tethering a positioned element (a tooltip, a
popover) to another element without JS measurement code. Worth knowing
exists; too new and too implementation-specific to audit generically here.

### CSS masonry layout

A `display: masonry` / `grid-template-rows: masonry` proposal for
Pinterest-style staggered grids. Not yet stable enough across browsers to
build a rule of thumb around.

### Print layout (`@media print`, `break-inside`, page margins)

A distinct discipline (page-based rather than viewport-based) that this
web-app skill doesn't scope into — flag only if a `web/src` component has
an explicit print requirement, which none currently do.

### `@scope`

Scoped CSS blocks (`@scope (.card) { ... }`) as an alternative to naming
conventions for style encapsulation — a styling-architecture decision
orthogonal to structure/layout; belongs with the project's CSS
methodology decisions, not this skill.

### View transitions

The View Transitions API (cross-document and same-document) governs
*animated* transitions between states/pages, not static structure — a
motion concern this skill doesn't scope into.

---

## Priority Summary

| Priority | Principles |
|---|---|
| **High** | L1, L2, L3, L4, L14 |
| **Medium** | L5, L8, L9, L10, L12, L13, L15, L16, L17 |
| **Low** | L6, L7, L11, L18 |

Work top-down when triaging a large report: the High tier is where a
defect is either invisible to assistive technology entirely (L2, L3, L14) or
undermines every principle that depends on correct structure existing at
all (L1, L4).

---

*Based on: the WHATWG HTML Living Standard §4.3 (Sections); the W3C WAI
Page Structure tutorial; WCAG 2.2 Understanding documents for 1.3.2
(Meaningful Sequence) and 2.4.1 (Bypass Blocks); MDN's "Document and website
structure" guide and its blog post "Default styles for h1 elements are
changing"; MDN's Container Queries guide; web.dev's "Learn CSS: Layout";
Scott O'Hara's writing on the `<section>` element; Léonie Watson's "Flexbox
and the keyboard navigation disconnect"; Every Layout's *Axioms* and
*Composition* (Heydon Pickering & Andy Bell); CUBE CSS (cube.fyi, Andy
Bell); buildexcellentwebsit.es; and Nielsen Norman Group's "The Layer-Cake
Pattern for Scanning Content," "F-Shaped Pattern of Reading on the Web,"
"The Power of Proximity: Using Gestalt Psychology in UX Design," and
"Visual Hierarchy in UX Design."*
