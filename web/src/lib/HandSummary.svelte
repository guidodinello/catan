<script lang="ts">
  import { RESOURCES } from "./resources";
  import { RESOURCE_ICON } from "./icons";
  import type { TrailEntry } from "./api";
  import { PLAYER_COLOR } from "./playerColor";

  interface Props {
    diceRoll: [number, number] | null;
    // Recent RollDice trail entries (see lib/actionLog.ts's recentRolls),
    // already capped to the last few by the caller -- folded into the same
    // action-trail mechanism as "visible, paced bot turns" rather than a
    // separate history buffer, so it also captures bot rolls within a
    // synchronous batch that a naive poll-based buffer would miss.
    recentRolls?: TrailEntry[];
    // Undefined when this seat's hand isn't revealed to the viewer (e.g.
    // an all-bot spectator game) -- PlayerView.resources is itself optional
    // for exactly that reason.
    resources?: Record<string, number>;
  }

  const { diceRoll, recentRolls = [], resources }: Props = $props();
</script>

<div class="hand-summary">
  <p class="roll">
    {#if diceRoll}
      Last roll: {diceRoll[0]} + {diceRoll[1]} = <strong>{diceRoll[0] + diceRoll[1]}</strong>
    {:else}
      Last roll: --
    {/if}
  </p>
  {#if recentRolls.length > 0}
    <p class="recent-rolls">
      Recent rolls:
      {#each recentRolls as entry, i (i)}
        {@const [d1, d2] = entry.dice_roll ?? [0, 0]}
        <span class="roll-chip">
          <span class="swatch" style="background: {PLAYER_COLOR[entry.player_id]}"></span>
          {d1 + d2}
        </span>
      {/each}
    </p>
  {/if}
  {#if resources}
    <p class="hand-label">Your hand</p>
    <ul class="hand">
      {#each RESOURCES as resource (resource)}
        {@const Icon = RESOURCE_ICON[resource]}
        <li title={resource}>
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
            <Icon />
          </svg>
          <span class="visually-hidden">{resource}</span>
          {resources[resource] ?? 0}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .hand-summary {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
  }

  .roll {
    margin: 0 0 0.5rem;
  }

  .recent-rolls {
    margin: 0 0 0.5rem;
    font-size: 0.85rem;
    color: #444;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .roll-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
  }

  .swatch {
    display: inline-block;
    width: 0.7em;
    height: 0.7em;
    border-radius: 50%;
    border: 1px solid #000;
  }

  .hand-label {
    margin: 0 0 0.25rem;
    font-weight: bold;
  }

  .hand {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
  }

  .hand li {
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
