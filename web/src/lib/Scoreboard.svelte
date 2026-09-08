<script lang="ts">
  import type { GameStateView } from "./api";
  import { PLAYER_COLOR } from "./playerColor";
  import DevCardStrip from "./DevCardStrip.svelte";

  interface Props {
    state: GameStateView;
    viewer?: number;
  }

  const { state, viewer }: Props = $props();

  // player.victory_points (engine.game.victory_points) is deliberately the
  // *public* tally -- settlements/cities/road/army bonuses plus only
  // revealed_vp_cards, never unrevealed cards still sitting in dev_hand
  // (that's engine.game.true_victory_points, the real win condition -- kept
  // engine-internal and never serialized). An unplayed VICTORY_POINT dev
  // card is hidden information a player knows about their own hand but the
  // rest of the board doesn't -- so only the viewer's own row adds their own
  // unrevealed VP cards on top of the public number; every other row stays
  // at the public tally, same as a real board where you can't see what's in
  // someone else's hand. On a win, the engine reveals the winner's VP cards
  // (dev_hand -> revealed_vp_cards) in the same step it ends the game, so
  // the winner's public victory_points is already 10 and hidden here reads 0
  // -- no double-count.
  function hiddenVpCount(player: GameStateView["players"][number]): number {
    return player.dev_hand?.filter((c) => c.card_type === "VICTORY_POINT").length ?? 0;
  }
</script>

<div class="scoreboard panel">
  <h2 class="panel-label">Victory points</h2>
  <ul>
    {#each state.players as player (player.player_id)}
      {@const hidden = player.player_id === viewer ? hiddenVpCount(player) : 0}
      <li>
        <span class="swatch" style="background: {PLAYER_COLOR[player.player_id]}"></span>
        Player {player.player_id}{player.player_id === viewer ? " (you)" : ""}:
        <strong>{player.victory_points + hidden}</strong>
        {#if hidden > 0}
          <span class="badge" title="Not visible to other players">
            includes {hidden} unrevealed VP card{hidden > 1 ? "s" : ""}
          </span>
        {/if}
        {#if state.longest_road_owner === player.player_id}
          <span class="badge">Longest Road</span>
        {/if}
        {#if state.largest_army_owner === player.player_id}
          <span class="badge">Largest Army</span>
        {/if}
        <DevCardStrip {player} />
      </li>
    {/each}
  </ul>
</div>

<style>
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .badge {
    font-size: 0.75em;
    border: 1px solid currentColor;
    border-radius: 999px;
    padding: var(--space-1) 0.4rem;
    margin-left: var(--space-1);
  }
</style>
