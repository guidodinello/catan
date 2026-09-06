<script lang="ts">
  // engine/board.py's Resource enum, in its declared order.
  const RESOURCES = ["LUMBER", "WOOL", "GRAIN", "BRICK", "ORE"] as const;

  // engine/game.py's MAX_TRADE_OFFER_SIDE. A UI convenience cap only --
  // mirrors agents/human.py's _prompt_resource_bundle, exactly as the plan
  // doc specifies; apply_action is the real, authoritative validator, so a
  // client that guesses wrong still just gets a rejected-action error, same
  // as any other illegal action.
  const MAX_TRADE_OFFER_SIDE = 4;

  interface Props {
    // The human proposer's own hand, capping the give side. Optional
    // because PlayerView only reveals `resources` for the viewer's own seat.
    humanResources?: Record<string, number>;
    onSubmit: (give: Record<string, number>, receive: Record<string, number>) => void;
    onCancel: () => void;
  }

  const { humanResources = {}, onSubmit, onCancel }: Props = $props();

  function emptyBundle(): Record<string, number> {
    return Object.fromEntries(RESOURCES.map((r) => [r, 0]));
  }

  let give: Record<string, number> = $state(emptyBundle());
  let receive: Record<string, number> = $state(emptyBundle());

  const giveTotal = $derived(RESOURCES.reduce((sum, r) => sum + (give[r] ?? 0), 0));
  const receiveTotal = $derived(
    RESOURCES.reduce((sum, r) => sum + (receive[r] ?? 0), 0),
  );
  const overlappingResource = $derived(
    RESOURCES.some((r) => (give[r] ?? 0) > 0 && (receive[r] ?? 0) > 0),
  );
  const isValid = $derived(giveTotal > 0 && receiveTotal > 0 && !overlappingResource);

  function nonZero(bundle: Record<string, number>): Record<string, number> {
    return Object.fromEntries(
      Object.entries(bundle).filter(([, count]) => count > 0),
    );
  }

  function submit() {
    if (!isValid) return;
    onSubmit(nonZero(give), nonZero(receive));
  }
</script>

<div class="trade-form">
  <h3>Propose a trade</h3>
  <p>Up to {MAX_TRADE_OFFER_SIDE} cards per side, no shared resource type.</p>
  <div class="columns">
    <div>
      <h4>You give</h4>
      {#each RESOURCES as r (r)}
        {@const cap = Math.min(humanResources[r] ?? 0, MAX_TRADE_OFFER_SIDE)}
        <label>
          {r} (have {humanResources[r] ?? 0})
          <input type="number" min="0" max={cap} bind:value={give[r]} />
        </label>
      {/each}
    </div>
    <div>
      <h4>You receive</h4>
      {#each RESOURCES as r (r)}
        <label>
          {r}
          <input type="number" min="0" max={MAX_TRADE_OFFER_SIDE} bind:value={receive[r]} />
        </label>
      {/each}
    </div>
  </div>
  {#if overlappingResource}
    <p class="error">Cannot trade the same resource on both sides.</p>
  {/if}
  <button onclick={submit} disabled={!isValid}>Propose</button>
  <button onclick={onCancel}>Cancel</button>
</div>

<style>
  .trade-form {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
  }

  .columns {
    display: flex;
    gap: 2rem;
  }

  label {
    display: block;
    margin-bottom: 0.3rem;
  }

  input {
    width: 3.5rem;
    margin-left: 0.4rem;
  }

  .error {
    color: #d90429;
  }
</style>
