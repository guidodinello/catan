<script lang="ts">
  import { RESOURCES } from "./resources";
  import { RESOURCE_ICON } from "./icons";

  interface Props {
    diceRoll: [number, number] | null;
    // Undefined when this seat's hand isn't revealed to the viewer (e.g.
    // an all-bot spectator game) -- PlayerView.resources is itself optional
    // for exactly that reason.
    resources?: Record<string, number>;
  }

  const { diceRoll, resources }: Props = $props();
</script>

<div class="hand-summary">
  <p class="roll">
    {#if diceRoll}
      Last roll: {diceRoll[0]} + {diceRoll[1]} = <strong>{diceRoll[0] + diceRoll[1]}</strong>
    {:else}
      Last roll: --
    {/if}
  </p>
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
