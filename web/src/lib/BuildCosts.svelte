<script lang="ts">
  import { RESOURCES } from "./resources";
  import { RESOURCE_ICON } from "./icons";
  import type { BuildCosts } from "./api";

  interface Props {
    costs: BuildCosts;
  }

  const { costs }: Props = $props();

  // Display order -- the wire payload is a plain object, so its key order
  // isn't a contract; this is the order this component chooses to render.
  const KIND_LABEL: Record<string, string> = {
    ROAD: "Road",
    SETTLEMENT: "Settlement",
    CITY: "City",
    DEV_CARD: "Dev card",
  };
  const KIND_ORDER = ["ROAD", "SETTLEMENT", "CITY", "DEV_CARD"];
</script>

<div class="build-costs panel">
  <p class="label">Build costs</p>
  <ul class="costs">
    {#each KIND_ORDER as kind (kind)}
      {@const cost = costs[kind] ?? {}}
      <li>
        <span class="kind">{KIND_LABEL[kind]}</span>
        <span class="amounts">
          {#each RESOURCES as resource (resource)}
            {#if cost[resource]}
              {@const Icon = RESOURCE_ICON[resource]}
              <span class="amount" title={resource}>
                <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                  <Icon />
                </svg>
                <span class="visually-hidden">{resource}</span>
                {cost[resource]}
              </span>
            {/if}
          {/each}
        </span>
      </li>
    {/each}
  </ul>
</div>

<style>
  .label {
    margin: 0 0 var(--space-2);
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .costs {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .costs li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    font-size: var(--fs-sm);
  }

  /* This is a value (what a build costs), not a caption -- it must read
     at full body contrast, not the muted "greyed-out" look it had before
     tokens existed. */
  .kind {
    color: var(--text);
  }

  .amounts {
    display: flex;
    gap: var(--space-3);
  }

  .amount {
    display: flex;
    align-items: center;
    gap: 0.2rem;
    color: var(--text);
  }
</style>
