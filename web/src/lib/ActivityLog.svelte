<script lang="ts">
  import type { TrailEntry } from "./api";
  import { groupActivityEntries } from "./actionLog";
  import { PLAYER_COLOR } from "./playerColor";

  interface Props {
    // Already-revealed entries only (paced or not) -- App.svelte owns the
    // pending/reveal queue; this component just renders what's landed so
    // far, oldest first.
    entries: TrailEntry[];
  }

  const { entries }: Props = $props();

  // A run of consecutive RejectTrade entries that killed the trade outright
  // collapses into one "everyone rejected" line -- see groupActivityEntries.
  const lines = $derived(groupActivityEntries(entries));

  let listEl: HTMLUListElement | undefined = $state();

  // Scroll to the newest (bottom) entry whenever the list changes -- entries
  // is a new array each time App.svelte reveals one (see pushToLog), so this
  // re-runs on every reveal, not just once.
  $effect(() => {
    entries;
    listEl?.scrollTo({ top: listEl.scrollHeight });
  });
</script>

<div class="activity-log panel">
  <p class="activity-log-label">Recent activity</p>
  {#if lines.length === 0}
    <p class="empty">Nothing yet.</p>
  {:else}
    <ul bind:this={listEl}>
      {#each lines as line (line.key)}
        <li>
          <span class="swatches">
            {#each line.playerIds as playerId (playerId)}
              <span class="swatch" style="background: {PLAYER_COLOR[playerId]}"></span>
            {/each}
          </span>
          {line.text}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .activity-log {
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex: 1;
  }

  .activity-log-label {
    margin: 0 0 var(--space-2);
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    flex: none;
  }

  .empty {
    margin: 0;
    color: var(--text-muted);
    font-size: var(--fs-sm);
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  li {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--fs-sm);
  }

  .swatches {
    flex: none;
    display: inline-flex;
    gap: 0.15rem;
  }

  .swatch {
    --swatch-size: 0.8em;
  }
</style>
