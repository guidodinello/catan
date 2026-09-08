<script lang="ts">
  import type { TradeOfferView } from "./api";
  import { RESOURCES, type Resource } from "./resources";
  import { RESOURCE_ICON } from "./icons";

  interface Props {
    offer: TradeOfferView;
    viewer?: number;
  }

  const { offer, viewer }: Props = $props();

  // A domestic trade offer is announced to every player by the real rules
  // (README decision 5) -- unlike a hand, it's never hidden information, so
  // there's no redaction to worry about here, just phrasing it unambiguously
  // for whoever's looking (the proposer waiting on responses, or a
  // responder deciding whether to accept). A counter-offer is exactly as
  // public, but must read as a counter (offer.counter_of set) rather than a
  // fresh offer, so the original proposer can tell the two apart.
  function nonZero(bundle: Record<string, number>): [Resource, number][] {
    return RESOURCES.map((r) => [r, bundle[r] ?? 0] as [Resource, number]).filter(
      ([, count]) => count > 0,
    );
  }

  const label = $derived(
    offer.counter_of !== null
      ? offer.proposer === viewer
        ? "Your counter-offer"
        : `Player ${offer.proposer} countered`
      : offer.proposer === viewer
        ? "Your trade offer"
        : `Player ${offer.proposer} offers a trade`,
  );
</script>

<div class="trade-offer banner-warn">
  <p class="trade-offer-label">{label}</p>
  <div class="trade-offer-bundle">
    <span>Gives:</span>
    <ul class="card-list">
      {#each nonZero(offer.give) as [resource, count] (resource)}
        {@const Icon = RESOURCE_ICON[resource]}
        <li class="card">
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><Icon /></svg>
          {count}
        </li>
      {/each}
    </ul>
  </div>
  <div class="trade-offer-bundle">
    <span>Wants:</span>
    <ul class="card-list">
      {#each nonZero(offer.receive) as [resource, count] (resource)}
        {@const Icon = RESOURCE_ICON[resource]}
        <li class="card">
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><Icon /></svg>
          {count}
        </li>
      {/each}
    </ul>
  </div>
</div>

<style>
  .trade-offer-label {
    margin: 0 0 var(--space-1);
    font-weight: bold;
  }

  .trade-offer-bundle {
    margin: var(--space-1) 0;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }

  .card-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: inline-flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }

  .card {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
</style>
