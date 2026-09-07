<script lang="ts">
  import type { TrailEntry } from "./api";
  import { describeTrailEntry } from "./actionLog";
  import { PLAYER_COLOR } from "./playerColor";

  interface Props {
    // Already-revealed entries only (paced or not) -- App.svelte owns the
    // pending/reveal queue; this component just renders what's landed so
    // far, oldest first.
    entries: TrailEntry[];
  }

  const { entries }: Props = $props();
</script>

<div class="activity-log">
  <p class="activity-log-label">Recent activity</p>
  {#if entries.length === 0}
    <p class="empty">Nothing yet.</p>
  {:else}
    <ul>
      {#each entries as entry, i (i)}
        <li>
          <span class="swatch" style="background: {PLAYER_COLOR[entry.player_id]}"></span>
          {describeTrailEntry(entry)}
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

  .swatch {
    flex: none;
    display: inline-block;
    width: 0.8em;
    height: 0.8em;
    border-radius: 50%;
    border: 1px solid #000;
  }
</style>
