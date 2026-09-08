<script lang="ts">
  import type { SeatKind } from "./api";
  import { loadLastGame } from "./lastGame";

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
    onResume: (gameId: string, viewer: number | undefined) => void;
  }

  const { onCreate, onResume }: Props = $props();

  // Prefilled from whatever was last persisted (see lib/lastGame.ts) so a
  // hard page reload can reconnect without retyping the game id.
  const lastGame = loadLastGame();
  let resumeGameId = $state(lastGame.gameId);
  let resumeViewer: number | undefined = $state(lastGame.viewer);

  function submitResume() {
    const trimmed = resumeGameId.trim();
    if (!trimmed) return;
    onResume(trimmed, resumeViewer);
  }

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

<div class="game-setup panel">
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
    Multiple "Human" seats are all playable from this browser tab -- a
    "Viewing as" switcher appears once the game starts, letting you hot-seat
    between them (pass-and-play style; nothing hides a seat's view from
    someone standing at the same screen). A lineup with no human seat runs
    entirely as a bot-vs-bot spectator game.
  </p>

  <label>
    Seed (optional):
    <input type="number" bind:value={seed} placeholder="random" />
  </label>

  <button onclick={submit}>Start game</button>

  <hr />

  <h3>Resume a game</h3>
  <label>
    Game ID:
    <input type="text" bind:value={resumeGameId} placeholder="e.g. AbC123-xyz" />
  </label>
  <label>
    Your seat (leave blank to spectate):
    <input
      type="number"
      min="0"
      value={resumeViewer ?? ""}
      oninput={(e) => {
        const v = e.currentTarget.value;
        resumeViewer = v === "" ? undefined : Number(v);
      }}
    />
  </label>
  <p class="hint">
    Reconnects to a game already running on the server (e.g. after a page
    reload) instead of starting a new one. Only works if this browser has
    created or resumed at least one game before, since resuming reuses this
    browser's cached board layout rather than fetching it again.
  </p>
  <button onclick={submitResume} disabled={!resumeGameId.trim()}>Resume game</button>
</div>

<style>
  .game-setup {
    max-width: 480px;
  }

  label {
    display: block;
    margin-bottom: var(--space-2);
  }

  .seats {
    margin: var(--space-3) 0;
  }

  .hint {
    color: var(--text-muted);
    font-size: var(--fs-sm);
  }

  input,
  select {
    background: var(--surface-raised);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    font: inherit;
    padding: 0.15rem var(--space-2);
  }
</style>
