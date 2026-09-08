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

<div class="build-costs">
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
  .build-costs {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
  }

  .label {
    margin: 0 0 0.5rem;
    font-weight: bold;
  }

  .costs {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .costs li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    font-size: 0.9rem;
  }

  .kind {
    color: #444;
  }

  .amounts {
    display: flex;
    gap: 0.6rem;
  }

  .amount {
    display: flex;
    align-items: center;
    gap: 0.2rem;
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
