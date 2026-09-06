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
  // responder deciding whether to accept).
  function nonZero(bundle: Record<string, number>): [Resource, number][] {
    return RESOURCES.map((r) => [r, bundle[r] ?? 0] as [Resource, number]).filter(
      ([, count]) => count > 0,
    );
  }
</script>

<div class="trade-offer">
  <p class="trade-offer-label">
    {offer.proposer === viewer ? "Your trade offer" : `Player ${offer.proposer} offers a trade`}
  </p>
  <p class="trade-offer-bundle">
    Gives:
    {#each nonZero(offer.give) as [resource, count] (resource)}
      {@const Icon = RESOURCE_ICON[resource]}
      <span class="card">
        <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><Icon /></svg>
        {count}
      </span>
    {/each}
  </p>
  <p class="trade-offer-bundle">
    Wants:
    {#each nonZero(offer.receive) as [resource, count] (resource)}
      {@const Icon = RESOURCE_ICON[resource]}
      <span class="card">
        <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><Icon /></svg>
        {count}
      </span>
    {/each}
  </p>
</div>

<style>
  .trade-offer {
    border: 1px solid #ffd60a;
    background: #fff3cd;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.75rem;
  }

  .trade-offer-label {
    margin: 0 0 0.35rem;
    font-weight: bold;
  }

  .trade-offer-bundle {
    margin: 0.15rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .card {
    display: inline-flex;
    align-items: center;
    gap: 0.15rem;
  }
</style>
