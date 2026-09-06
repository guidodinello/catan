<script lang="ts">
  import { untrack } from "svelte";
  import { RESOURCES, type Resource } from "./resources";
  import { RESOURCE_ICON } from "./icons";
  import type { LegalAction } from "./api";

  interface Props {
    title: string;
    // 1 for PlayMonopoly (engine/actions.py's `resource`), 2 for
    // PlayYearOfPlenty (`resource_1`/`resource_2`, repeats allowed).
    count: 1 | 2;
    // Pre-filtered to just this kind's instances (PlayMonopoly or
    // PlayYearOfPlenty) -- this component only looks up which already-legal
    // instance matches the picks, it never constructs an action itself.
    legalActions: LegalAction[];
    onSubmit: (index: number) => void;
    onCancel: () => void;
  }

  const { title, count, legalActions, onSubmit, onCancel }: Props = $props();

  // count is fixed for this component's lifetime (a fresh instance is
  // mounted each time the parent shows the form) -- untrack makes that
  // intentional one-time read explicit instead of tripping
  // state_referenced_locally.
  let picks: Resource[] = $state(
    untrack(() => (count === 1 ? ["LUMBER"] : ["LUMBER", "LUMBER"])),
  );

  // engine/game.py's _year_of_plenty_options only enumerates canonical
  // order (i <= j by Resource's declared enum order, i.e. RESOURCES' order)
  // -- sort both sides the same way before comparing so pick order doesn't
  // matter to the player.
  function sorted2(a: Resource, b: Resource): [Resource, Resource] {
    return RESOURCES.indexOf(a) <= RESOURCES.indexOf(b) ? [a, b] : [b, a];
  }

  const matchedIndex = $derived.by(() => {
    if (count === 1) {
      const match = legalActions.find((a) => a.resource === picks[0]);
      return match?.index ?? null;
    }
    const [pa, pb] = sorted2(picks[0], picks[1]);
    const match = legalActions.find((a) => {
      const [ra, rb] = sorted2(a.resource_1 as Resource, a.resource_2 as Resource);
      return ra === pa && rb === pb;
    });
    return match?.index ?? null;
  });

  function submit() {
    if (matchedIndex !== null) onSubmit(matchedIndex);
  }
</script>

<div class="dev-card-form">
  <h3>{title}</h3>
  {#each Array.from({ length: count }) as _unused, i (i)}
    {@const Icon = RESOURCE_ICON[picks[i]]}
    <label>
      {count > 1 ? `Resource ${i + 1}` : "Resource"}
      <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
        <Icon />
      </svg>
      <select bind:value={picks[i]}>
        {#each RESOURCES as r (r)}
          <option value={r}>{r}</option>
        {/each}
      </select>
    </label>
  {/each}
  <button onclick={submit} disabled={matchedIndex === null}>Confirm</button>
  <button onclick={onCancel}>Cancel</button>
</div>

<style>
  .dev-card-form {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.4rem;
  }
</style>
