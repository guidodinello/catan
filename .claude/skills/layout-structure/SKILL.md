---
name: layout-structure
allowed-tools: Bash Read Grep Glob Agent
description: >
  Design and audit web page structure — document skeleton, semantic markup,
  heading hierarchy, and CSS layout mechanics (flow, flex, grid, intrinsic
  sizing, container queries). Use when building a new screen or component,
  when markup has drifted into div soup, or as a periodic structural audit
  of web/src. Does NOT modify code — findings only.
---

## Goal

Catch structural defects that linters and typecheckers can't: markup chosen
for its CSS hooks instead of its meaning, a heading hierarchy that no longer
means anything, layout mechanics fighting the browser instead of working
with it, and — the highest-value class — visual order silently diverging
from DOM order, which breaks tab order and screen-reader reading order
without ever showing up in a screenshot. A well-structured page produces few
or no blocker/high findings — don't manufacture severity to fill out a
report.

This skill is **web-only**, scoped to `web/src` (Svelte 5 + Vite, plain CSS
— no Tailwind, no JSX/TSX; markup, script, and scoped `<style>` all live in
one `.svelte` file per component, plus the global `web/src/app.css`).

**Adapted from a sibling project's `layout-structure` skill** (originally
written for a React 19 + Tailwind v4 codebase with three adjacent skills:
`accessibility-review`, `ui-ux-review`, `react-composition-audit`). The 18
principles below are HTML/CSS/WCAG-level and framework-agnostic, so they
carry over unchanged; the greps and the sibling-skill boundaries do not, and
are adapted below.

---

## Boundaries

**None of `accessibility-review`, `ui-ux-review`, or `react-composition-audit`
exist in this project.** The categories they would have owned (accessible
names, focus/keyboard management, WCAG SC tagging, proximity thresholds,
density/scan-pattern judgment, native-control-vs-`<div onClick>`, primitive
reuse) are genuinely **uncovered** here, not silently handled elsewhere.

This skill still does **not** expand to cover them — scope creep here means
sloppy, under-researched findings in territory this skill was never built
to judge. Instead: when a candidate matches a row below, note it in the
report's own **"Uncovered — no sibling skill in this project"** section
(see `references/layout-checks.md` §3's DEFER table, kept as a reference
for *what these categories are* even with no skill to route to) rather than
silently dropping it or half-judging it under this skill's rules.

This skill's own claimed territory, unchanged from the original: **container/
structural elements and their CSS layout behavior** — semantic markup choice,
heading hierarchy, flow/flex/grid mechanics, intrinsic sizing, container
queries — plus two things that would otherwise fall through every skill's
cracks: **source order vs. visual/tab order** (WCAG 1.3.2/2.4.3 — see
[`layout-philosophy.md`](references/layout-philosophy.md) L14) and **skip
links / bypass blocks** (WCAG 2.4.1 — L15).

---

## References

| File | Contents | Read it in |
|------|----------|------------|
| [`references/layout-checks.md`](references/layout-checks.md) | Every grep, the DEFER routing table, classification rules, quick checklist | Steps 2, 3 |
| [`references/layout-philosophy.md`](references/layout-philosophy.md) | Principles L1–L18 as ideas, plus concepts documented but deliberately not checked | Step 3 |

---

## Invocation

Accepts one of:

- **A component file path** — read the source directly.
- **Nothing specific** — full sweep across `web/src`.

## Step 1 — Scope

Identify whether this is a **quick check** (single component, e.g. during
feature review) or a **full audit** (periodic sweep). `web/src` is small
(one `App.svelte`, ~9 components under `lib/`, plus `lib/icons/` — purely
presentational SVG wrappers, low-value to audit) — **no per-directory agent
spawning is needed for a full audit here**; the whole tree fits comfortably
in one pass. Reserve the "spawn one agent per candidate agent" pattern in
Step 3 for judging, not for scope-splitting.

Exclude `*.test.ts`, `lib/__fixtures__/`, and `lib/icons/` (icon components
have no structural content to judge) unless the user asks for them; note
the scope applied in the final report.

## Step 2 — Scan for candidates

Run the greps from [`references/layout-checks.md`](references/layout-checks.md)
§1–2 verbatim, across `web/src` (all `*.svelte` files plus `web/src/app.css`).

## Step 3 — Judge each candidate (agent)

Spawn a **single** fresh agent (general-purpose, needs Read and Bash) with
the Step 2 grep output and these instructions — one agent is enough given
`web/src`'s size; only split further if a future sweep finds the tree has
grown substantially:

```
You are auditing web page structure and CSS layout in a Svelte 5 + Vite
codebase (web/src), plain CSS, no Tailwind. This skill owns container/
structural elements and their layout behavior. Three sibling skills that
would normally own adjacent ground (accessible names/focus management,
proximity/density/scan-pattern judgment, native-control-vs-div reuse) do
NOT exist in this project -- note candidates matching those categories in
an "Uncovered" section rather than silently dropping or half-judging them
under this skill's own rules.

Read BOTH of these first, in full:
  .claude/skills/layout-structure/references/layout-checks.md    (§3 classification rules, incl. the DEFER table)
  .claude/skills/layout-structure/references/layout-philosophy.md (L1–L18, the reasoning)

For EVERY hit, read enough surrounding markup/CSS (~20 lines, more if the
component is small) before classifying. Do not classify from the grep
string alone -- the heading-rank and truncation checks in particular require
reading the full element sequence, not a single line. Remember Svelte's
{#each ...} block is this codebase's list-rendering construct, not a .map()
call in a JSX return -- read what each {#each} actually renders.

First check the DEFER table (layout-checks.md §3) -- if a candidate matches
a row there, note it as Uncovered (no sibling skill exists to route it to)
and move on; do not judge it under this skill's own rules. Otherwise
classify per §3's FIX-shaped / JUDGMENT / Not-a-finding rules.

For each finding return:
- file:line
- pattern matched
- principle (L1–L18)
- category: FIX-shaped / JUDGMENT / Uncovered (name which sibling-skill
  category it would have matched)
- one-line reason citing what you read in the file
- proposed fix direction (for FIX-shaped and JUDGMENT findings that survive)

Be conservative on JUDGMENT items — when unsure whether reuse context or
intent justifies a pattern, read one more call site before opening it.
```

Wait for the agent to return before continuing.

## Step 4 — Rate severity

A standard five-level scale, calibrated for structural defects specifically
— weight by how much of the page's meaning or navigability is affected, not
by how many lines change:

- **blocker** — the structural defect makes a primary flow unusable for
  keyboard or screen-reader users specifically (source order diverging from
  visual order on a form or step flow — L14; a page with zero landmarks or
  zero headings at all).
- **high** — a load-bearing structural defect that degrades but doesn't
  fully block (missing `<h1>` or duplicate `<h1>`s on a content-heavy
  screen — L3; a `<section>` masquerading as the only structural element on
  a page that badly needs one — L4; a real overflow bug from a missing
  `min-width: 0` — L12).
- **medium** — noticeable but self-contained (a hand-simulated grid via
  nested flex — L8; a device-width breakpoint on a reused primitive that
  should be a container query — L10; missing skip link — L15).
- **low** — architectural/polish (div soup — L6; flex reaching for a job
  flow already does — L7; a bespoke layout primitive duplicated across
  screens — L11; misaligned edges — L18).
- **nit** — cosmetic (an arbitrary spacing value off-grid with no visible
  impact).

## Step 5 — Report

Group findings by component so the reviewer can act file-by-file. Within a
component, list findings most-severe first, each tagged with its principle.

### Report format

```
## Layout Structure Review — <scope>

### <ComponentName> (<file path>)
- **[blocker]** [L14] <file:line> — <what's wrong> → <fix>
- **[medium]** [L10] <file:line> — <what's wrong> → <fix>

### <NextComponent> (<file path>)
- ...

### Correct-as-is (checked, no defect)
- <file:line> — <pattern considered> — <why it already passes>

### Uncovered — no sibling skill in this project
- <file:line> — <pattern> — <category it would have matched, e.g. "accessible name">

<N> findings across <M> components. <K> uncovered (no sibling skill exists
to judge them — see the skill's Boundaries section).
```

If no findings survive Step 4, report `No layout defects found` plus which
categories and directories were checked — don't leave the reviewer guessing
what was covered.

---

## Appendix A — Quick audit ordering checklist

For a fast pass on a single file, work through
[`references/layout-checks.md`](references/layout-checks.md) §4 in order —
it's ordered from highest-value (missing/broken structural skeleton) to
lowest (spacing polish), so stopping early after time pressure still
surfaces the highest-priority findings first.
