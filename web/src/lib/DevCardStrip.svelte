<script lang="ts">
  import type { PlayerView } from "./api";
  import { DEV_CARD_ICON, DevCardBackIcon, KnightIcon, ProgressCardIcon, VictoryPointIcon } from "./icons";

  interface Props {
    player: PlayerView;
  }

  const { player }: Props = $props();

  // Branches on field presence, not `player.player_id === viewer` -- the
  // server (server/serialize.py's player_view) already decided which one
  // to send, so this mirrors that redaction decision instead of
  // re-deriving it client-side.
  const heldGroups = $derived.by(() => {
    if (!player.dev_hand) return null; // redacted -- render the generic cover count instead
    const counts = new Map<string, number>();
    for (const card of player.dev_hand) {
      counts.set(card.card_type, (counts.get(card.card_type) ?? 0) + 1);
    }
    return [...counts.entries()];
  });

  const playedEntries = $derived(
    [
      { Icon: KnightIcon, label: "Knight", count: player.played_knights },
      { Icon: ProgressCardIcon, label: "Progress card", count: player.played_progress_count },
      { Icon: VictoryPointIcon, label: "Victory Point", count: player.revealed_vp_cards },
    ].filter((entry) => entry.count > 0),
  );
</script>

<div class="dev-card-strip">
  {#if (heldGroups && heldGroups.length > 0) || (player.dev_card_count ?? 0) > 0}
    <p class="dev-card-label">Held</p>
    <ul class="dev-card-list">
      {#if heldGroups}
        {#each heldGroups as [cardType, count] (cardType)}
          {@const Icon = DEV_CARD_ICON[cardType]}
          <li title={cardType}>
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <Icon />
            </svg>
            {count}
          </li>
        {/each}
      {:else if (player.dev_card_count ?? 0) > 0}
        <li title="Face-down (hidden)">
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
            <DevCardBackIcon />
          </svg>
          {player.dev_card_count}
        </li>
      {/if}
    </ul>
  {/if}
  {#if playedEntries.length > 0}
    <p class="dev-card-label">Played</p>
    <ul class="dev-card-list">
      {#each playedEntries as { Icon, label, count } (label)}
        <li title={label}>
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
            <Icon />
          </svg>
          {count}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .dev-card-strip {
    margin-top: 0.15rem;
  }

  .dev-card-label {
    margin: 0.15rem 0 0.1rem;
    font-size: 0.75em;
    color: var(--text-muted);
  }

  .dev-card-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }

  .dev-card-list li {
    display: flex;
    align-items: center;
    gap: 0.2rem;
    font-size: 0.85em;
  }
</style>
