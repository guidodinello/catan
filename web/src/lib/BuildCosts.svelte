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
  <h2 class="panel-label">Build costs</h2>
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
    gap: var(--space-1);
    color: var(--text);
  }
</style>
