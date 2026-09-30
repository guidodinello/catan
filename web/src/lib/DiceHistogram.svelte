<script lang="ts">
  import type { DiceRollRecord } from "./api";
  import { PLAYER_COLOR } from "./playerColor";
  import { DICE_TOTALS, countTotals, countsByPlayer, expectedCounts } from "./diceStats";

  interface Props {
    rolls: DiceRollRecord[];
    numPlayers: number;
    // False when the server restored this game from a snapshot that predates
    // roll tracking -- the earlier rolls are unrecoverable.
    complete: boolean;
    // "compact" = single-colour totals for the sidebar; "full" = bars stacked
    // by roller, with a per-player legend, for the game-over panel.
    variant?: "compact" | "full";
  }

  const { rolls, numPlayers, complete, variant = "compact" }: Props = $props();

  const full = $derived(variant === "full");
  const counts = $derived(countTotals(rolls));
  const expected = $derived(expectedCounts(rolls.length));
  const byPlayer = $derived(countsByPlayer(rolls, numPlayers));
  const sevens = $derived(byPlayer.map((row) => row[5]));

  const W = $derived(full ? 440 : 240);
  const H = $derived(full ? 200 : 110);
  const LABEL_H = 16; // axis labels under the bars
  const COUNT_H = 14; // headroom for the count printed above the tallest bar
  const slot = $derived(W / DICE_TOTALS.length);
  const barW = $derived(slot * 0.62);
  const plotH = $derived(H - LABEL_H - COUNT_H);
  // Scale to whichever is taller, actual or expected, so neither clips.
  const yMax = $derived(Math.max(1, ...counts, ...expected));
  const scale = $derived(plotH / yMax);
  const baseline = $derived(COUNT_H + plotH);

  // The first roll's turn, for the "counting since" note on a partial game.
  const firstTurn = $derived(rolls.length > 0 ? rolls[0].turn : null);
</script>

<figure class="dice-histogram" class:full>
  {#if rolls.length === 0}
    <p class="empty">No rolls yet.</p>
  {:else}
    <svg
      viewBox="0 0 {W} {H}"
      role="img"
      aria-label="Dice rolls: how often each total from 2 to 12 was rolled, with the fair-dice expectation"
    >
      {#each DICE_TOTALS as total, i (total)}
        {@const x = i * slot + (slot - barW) / 2}
        {#if full}
          {#each byPlayer as row, p (p)}
            {@const below = byPlayer.slice(0, p).reduce((sum, r) => sum + r[i], 0)}
            {#if row[i] > 0}
              <rect
                {x}
                y={baseline - (below + row[i]) * scale}
                width={barW}
                height={row[i] * scale}
                fill={PLAYER_COLOR[p]}
              />
            {/if}
          {/each}
        {:else if counts[i] > 0}
          <rect
            {x}
            y={baseline - counts[i] * scale}
            width={barW}
            height={counts[i] * scale}
            class="bar"
          />
        {/if}
        <line
          class="expected"
          x1={x - barW * 0.1}
          x2={x + barW * 1.1}
          y1={baseline - expected[i] * scale}
          y2={baseline - expected[i] * scale}
        />
        {#if counts[i] > 0}
          <!-- above the taller of bar and expected tick, so the two never overlap -->
          <text
            class="count"
            x={x + barW / 2}
            y={baseline - Math.max(counts[i], expected[i]) * scale - 3}
          >
            {counts[i]}
          </text>
        {/if}
        <text class="axis" x={x + barW / 2} y={H - 3}>{total}</text>
      {/each}
      <line class="base" x1="0" x2={W} y1={baseline} y2={baseline} />
    </svg>

    <p class="key">
      <span class="expected-swatch"></span> expected for {rolls.length} roll{rolls.length === 1
        ? ""
        : "s"} of fair dice
    </p>

    {#if full}
      <ul class="legend">
        {#each byPlayer as row, p (p)}
          <li>
            <span class="swatch" style="background: {PLAYER_COLOR[p]}"></span>
            Player {p}: {row.reduce((a, b) => a + b, 0)} rolls, {sevens[p]}
            {sevens[p] === 1 ? "seven" : "sevens"}
          </li>
        {/each}
      </ul>
    {/if}

    <table class="visually-hidden">
      <caption>Dice roll counts by total</caption>
      <thead><tr><th>Total</th><th>Rolled</th><th>Expected</th></tr></thead>
      <tbody>
        {#each DICE_TOTALS as total, i (total)}
          <tr><td>{total}</td><td>{counts[i]}</td><td>{expected[i].toFixed(1)}</td></tr>
        {/each}
      </tbody>
    </table>
  {/if}

  {#if !complete}
    <p class="partial">
      Counting since turn {firstTurn ?? "now"} -- this game was already under way when roll
      tracking began, so earlier rolls are missing.
    </p>
  {/if}
</figure>

<style>
  .dice-histogram {
    margin: 0;
  }

  svg {
    display: block;
    width: 100%;
    height: auto;
    max-width: 32rem;
  }

  .bar {
    fill: var(--accent);
  }

  .expected {
    stroke: var(--text-strong);
    stroke-width: 1.5;
    stroke-dasharray: 3 2;
  }

  .base {
    stroke: var(--border-strong);
    stroke-width: 1;
  }

  text {
    fill: var(--text);
    font-size: 9px;
    text-anchor: middle;
  }

  .count {
    fill: var(--text-strong);
    font-weight: 600;
  }

  .axis {
    fill: var(--text-muted);
  }

  .empty,
  .key,
  .partial {
    margin: var(--space-1) 0 0;
    font-size: var(--fs-sm);
    color: var(--text-muted);
  }

  .expected-swatch {
    display: inline-block;
    width: 1.1em;
    border-top: 2px dashed var(--text-strong);
    vertical-align: middle;
  }

  .legend {
    list-style: none;
    margin: var(--space-2) 0 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    font-size: var(--fs-sm);
  }

  .legend li {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
</style>
