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

<div class="activity-log">
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
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
  }

  .activity-log-label {
    margin: 0 0 0.5rem;
    font-weight: bold;
  }

  .empty {
    margin: 0;
    color: #666;
    font-size: 0.85rem;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    max-height: 180px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  li {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.9rem;
  }

  .swatches {
    flex: none;
    display: inline-flex;
    gap: 0.15rem;
  }

  .swatch {
    flex: none;
    display: inline-block;
    width: 0.8em;
    height: 0.8em;
    border-radius: 50%;
    border: 1px solid #000;
  }
</style>
