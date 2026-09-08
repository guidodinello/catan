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

<div class="hand-summary panel">
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
  .roll {
    margin: 0 0 var(--space-2);
  }

  .recent-rolls {
    margin: 0 0 var(--space-2);
    font-size: var(--fs-sm);
    color: var(--text-muted);
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2);
  }

  .roll-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }

  .roll-chip .swatch {
    --swatch-size: 0.7em;
  }

  .hand-label {
    margin: 0 0 var(--space-1);
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .hand {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-3);
  }

  .hand li {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
</style>
