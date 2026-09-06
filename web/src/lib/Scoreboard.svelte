<script lang="ts">
  import type { GameStateView } from "./api";
  import { PLAYER_COLOR } from "./playerColor";

  interface Props {
    state: GameStateView;
    viewer?: number;
  }

  const { state, viewer }: Props = $props();
</script>

<div class="scoreboard">
  <p class="scoreboard-label">Victory points</p>
  <ul>
    {#each state.players as player (player.player_id)}
      <li>
        <span class="swatch" style="background: {PLAYER_COLOR[player.player_id]}"></span>
        Player {player.player_id}{player.player_id === viewer ? " (you)" : ""}:
        <strong>{player.victory_points}</strong>
        {#if state.longest_road_owner === player.player_id}
          <span class="badge">Longest Road</span>
        {/if}
        {#if state.largest_army_owner === player.player_id}
          <span class="badge">Largest Army</span>
        {/if}
      </li>
    {/each}
  </ul>
</div>

<style>
  .scoreboard {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
  }

  .scoreboard-label {
    margin: 0 0 0.5rem;
    font-weight: bold;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .swatch {
    display: inline-block;
    width: 0.9em;
    height: 0.9em;
    border-radius: 50%;
    border: 1px solid #000;
    vertical-align: middle;
  }

  .badge {
    font-size: 0.75em;
    border: 1px solid currentColor;
    border-radius: 999px;
    padding: 0.05rem 0.4rem;
    margin-left: 0.25rem;
  }
</style>
