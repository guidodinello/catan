<script lang="ts">
  import type { LegalAction } from "./api";

  // Handled spatially by Board.svelte instead, per the plan's hybrid
  // interaction model (docs/plans/gui-web-frontend.md) -- never listed here.
  const SPATIAL_KINDS = new Set([
    "PlaceSettlement",
    "PlaceCity",
    "PlaceRoad",
    "MoveRobber",
    "StealFrom",
  ]);

  interface Props {
    legalActions: LegalAction[];
    onSelect: (index: number) => void;
    onProposeTrade: () => void;
    onCounterTrade: () => void;
    onPlayRoadBuilding: () => void;
    onPlayYearOfPlenty: () => void;
    onPlayMonopoly: () => void;
  }

  const {
    legalActions,
    onSelect,
    onProposeTrade,
    onCounterTrade,
    onPlayRoadBuilding,
    onPlayYearOfPlenty,
    onPlayMonopoly,
  }: Props = $props();

  // Kinds routed to a dedicated form/board-picking flow instead of the
  // generic expandable instance list below -- ProposeTrade already
  // established this precedent; CounterTrade/PlayRoadBuilding/
  // PlayYearOfPlenty/PlayMonopoly extend it rather than growing
  // formatAction's raw text.
  const SPECIAL_KIND_HANDLERS: Record<string, () => void> = {
    ProposeTrade: () => onProposeTrade(),
    CounterTrade: () => onCounterTrade(),
    PlayRoadBuilding: () => onPlayRoadBuilding(),
    PlayYearOfPlenty: () => onPlayYearOfPlenty(),
    PlayMonopoly: () => onPlayMonopoly(),
  };

  const panelActions = $derived(
    legalActions.filter((a) => !SPATIAL_KINDS.has(a.kind)),
  );

  // Bucket by kind, mirroring agents/human.py's HumanAgent.choose_action:
  // group legal actions by type, list the kinds, and only ask for a
  // specific instance when a kind has more than one.
  const byKind = $derived.by(() => {
    const map = new Map<string, LegalAction[]>();
    for (const a of panelActions) {
      const list = map.get(a.kind) ?? [];
      list.push(a);
      map.set(a.kind, list);
    }
    return map;
  });

  const kinds = $derived([...byKind.keys()].sort());

  let expandedKind: string | null = $state(null);

  function formatAction(action: LegalAction): string {
    const fields = Object.entries(action).filter(
      ([key]) => key !== "index" && key !== "kind" && key !== "open_ended",
    );
    if (fields.length === 0) return action.kind;
    const rendered = fields
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(", ");
    return `${action.kind}(${rendered})`;
  }

  function chooseKind(kind: string) {
    const specialHandler = SPECIAL_KIND_HANDLERS[kind];
    if (specialHandler) {
      specialHandler();
      return;
    }
    const options = byKind.get(kind) ?? [];
    if (options.length === 1) {
      onSelect(options[0].index);
      return;
    }
    expandedKind = expandedKind === kind ? null : kind;
  }
</script>

<div class="action-panel panel">
  <h2 class="panel-label">Actions</h2>
  {#if panelActions.length === 0}
    <p class="empty">No actions available.</p>
  {/if}
  {#each kinds as kind (kind)}
    {@const options = byKind.get(kind) ?? []}
    <div class="action-group">
      <button class:primary={kind === "EndTurn"} onclick={() => chooseKind(kind)}>
        {kind}{options.length > 1 ? ` (${options.length} options)` : ""}
      </button>
      {#if expandedKind === kind && options.length > 1}
        <ul>
          {#each options as option (option.index)}
            <li>
              <button onclick={() => onSelect(option.index)}>
                {formatAction(option)}
              </button>
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  {/each}
</div>

<style>
  .empty {
    margin: 0;
    color: var(--text-muted);
    font-size: var(--fs-sm);
  }

  .action-group {
    margin-bottom: var(--space-2);
  }

  .action-group > button {
    width: 100%;
    text-align: left;
  }

  ul {
    list-style: none;
    margin: var(--space-1) 0 0;
    padding-left: var(--space-2);
    max-height: 220px;
    overflow-y: auto;
  }

  li {
    margin-bottom: var(--space-1);
  }

  li > button {
    width: 100%;
    text-align: left;
  }
</style>
