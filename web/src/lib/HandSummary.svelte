<script lang="ts">
  // engine/board.py's Resource enum, in its declared order (matches
  // TradeForm.svelte's RESOURCES so a card's position stays consistent
  // across both places it's shown).
  const RESOURCES = ["LUMBER", "WOOL", "GRAIN", "BRICK", "ORE"] as const;

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
        <li>{resource}: {resources[resource] ?? 0}</li>
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
</style>
