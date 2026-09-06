<script lang="ts">
  import type { SeatKind } from "./api";

  const SEAT_KIND_OPTIONS: { value: SeatKind; label: string }[] = [
    { value: "human", label: "Human (this browser)" },
    { value: "random", label: "Bot: Random" },
    { value: "stratified_random", label: "Bot: Stratified Random" },
    { value: "heuristic", label: "Bot: Heuristic" },
  ];

  export interface NewGameConfig {
    num_players: number;
    seat_kinds: SeatKind[];
    seed: number | null;
  }

  interface Props {
    onCreate: (config: NewGameConfig) => void;
  }

  const { onCreate }: Props = $props();

  // engine/game.py's CatanGame fills non-human seats with a real opponent by
  // default (cli.py does the same with HeuristicAgent) -- default new bot
  // seats to a round-robin cycle rather than all the same kind, so a
  // default game has varied opponents (HeuristicAgent always rejects trade
  // offers -- see agents/heuristic.py's docstring -- so an all-heuristic
  // default meant every trade silently failed unless you noticed and
  // changed the dropdowns yourself).
  const DEFAULT_BOT_CYCLE: SeatKind[] = ["heuristic", "random", "stratified_random"];
  // Seat 0 is the assumed human slot for this cycle's purposes; bot seats
  // are everything after it, in order.
  function defaultBotKind(botOrdinal: number): SeatKind {
    return DEFAULT_BOT_CYCLE[botOrdinal % DEFAULT_BOT_CYCLE.length];
  }

  let numPlayers: number = $state(3);
  let seatKinds: SeatKind[] = $state([
    "human",
    defaultBotKind(0),
    defaultBotKind(1),
  ]);
  let seed: number | undefined = $state(1);

  // Keep seatKinds' length in sync with numPlayers, defaulting any newly
  // added seat to the next kind in the round-robin cycle and trimming from
  // the end when it shrinks.
  $effect(() => {
    if (seatKinds.length < numPlayers) {
      const added = Array.from({ length: numPlayers - seatKinds.length }, (_, i) =>
        defaultBotKind(seatKinds.length + i - 1),
      );
      seatKinds = [...seatKinds, ...added];
    } else if (seatKinds.length > numPlayers) {
      seatKinds = seatKinds.slice(0, numPlayers);
    }
  });

  function submit() {
    onCreate({
      num_players: numPlayers,
      seat_kinds: seatKinds,
      seed: seed ?? null,
    });
  }
</script>

<div class="game-setup">
  <h2>New game</h2>

  <label>
    Players:
    <select bind:value={numPlayers}>
      <option value={3}>3</option>
      <option value={4}>4</option>
    </select>
  </label>

  <div class="seats">
    {#each seatKinds as _seatKind, i (i)}
      <label>
        Seat {i}:
        <select bind:value={seatKinds[i]}>
          {#each SEAT_KIND_OPTIONS as opt (opt.value)}
            <option value={opt.value}>{opt.label}</option>
          {/each}
        </select>
      </label>
    {/each}
  </div>

  <p class="hint">
    At most one "Human" seat is playable from this browser tab -- the first
    one, if there are several. A lineup with no human seat runs entirely as
    a bot-vs-bot spectator game.
  </p>

  <label>
    Seed (optional):
    <input type="number" bind:value={seed} placeholder="random" />
  </label>

  <button onclick={submit}>Start game</button>
</div>

<style>
  .game-setup {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    max-width: 480px;
  }

  label {
    display: block;
    margin-bottom: 0.5rem;
  }

  .seats {
    margin: 0.75rem 0;
  }

  .hint {
    color: #666;
    font-size: 0.85rem;
  }
</style>
