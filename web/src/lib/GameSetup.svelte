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

  let numPlayers: number = $state(3);
  // engine/game.py's CatanGame fills non-human seats with a real opponent by
  // default (cli.py does the same with HeuristicAgent) -- default new seats
  // to "heuristic" rather than leaving them unset.
  let seatKinds: SeatKind[] = $state(["human", "heuristic", "heuristic"]);
  let seed: number | undefined = $state(1);

  // Keep seatKinds' length in sync with numPlayers, defaulting any newly
  // added seat to "heuristic" and trimming from the end when it shrinks.
  $effect(() => {
    if (seatKinds.length < numPlayers) {
      seatKinds = [
        ...seatKinds,
        ...Array(numPlayers - seatKinds.length).fill("heuristic"),
      ];
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
