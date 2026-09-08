<script lang="ts">
  import { RESOURCES } from "./resources";
  import { RESOURCE_ICON } from "./icons";

  // engine/game.py's MAX_TRADE_OFFER_SIDE. A UI convenience cap only --
  // mirrors agents/human.py's _prompt_resource_bundle, exactly as the plan
  // doc specifies; apply_action is the real, authoritative validator, so a
  // client that guesses wrong still just gets a rejected-action error, same
  // as any other illegal action.
  const MAX_TRADE_OFFER_SIDE = 4;

  interface Props {
    // The human proposer's (or counterer's) own hand, capping the give
    // side. Optional because PlayerView only reveals `resources` for the
    // viewer's own seat.
    humanResources?: Record<string, number>;
    // "propose" (default) for a fresh ProposeTrade, "counter" when
    // responding to someone else's offer with a CounterTrade -- only the
    // heading/button wording differs, the bundle-building UI is identical.
    mode?: "propose" | "counter";
    onSubmit: (give: Record<string, number>, receive: Record<string, number>) => void;
    onCancel: () => void;
  }

  const { humanResources = {}, mode = "propose", onSubmit, onCancel }: Props = $props();

  const heading = $derived(mode === "counter" ? "Counter-offer" : "Propose a trade");
  const submitLabel = $derived(mode === "counter" ? "Counter" : "Propose");

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

<div class="trade-form panel">
  <h3>{heading}</h3>
  <p>Up to {MAX_TRADE_OFFER_SIDE} cards per side, no shared resource type.</p>
  <div class="columns">
    <div>
      <h4>You give</h4>
      {#each RESOURCES as r (r)}
        {@const cap = Math.min(humanResources[r] ?? 0, MAX_TRADE_OFFER_SIDE)}
        {@const Icon = RESOURCE_ICON[r]}
        <label>
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
            <Icon />
          </svg>
          {r} (have {humanResources[r] ?? 0})
          <input type="number" min="0" max={cap} bind:value={give[r]} />
        </label>
      {/each}
    </div>
    <div>
      <h4>You receive</h4>
      {#each RESOURCES as r (r)}
        {@const Icon = RESOURCE_ICON[r]}
        <label>
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
            <Icon />
          </svg>
          {r}
          <input type="number" min="0" max={MAX_TRADE_OFFER_SIDE} bind:value={receive[r]} />
        </label>
      {/each}
    </div>
  </div>
  {#if overlappingResource}
    <p class="error">Cannot trade the same resource on both sides.</p>
  {/if}
  <button onclick={submit} disabled={!isValid}>{submitLabel}</button>
  <button onclick={onCancel}>Cancel</button>
</div>

<style>
  .columns {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-5);
  }

  label {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
  }

  input {
    width: 3.5rem;
    margin-left: var(--space-2);
    background: var(--surface-raised);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
  }

  .error {
    color: var(--danger);
  }
</style>
