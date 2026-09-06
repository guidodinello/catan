<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import {
    createGame,
    getLegalActions,
    getState,
    postAction,
    type Geometry,
    type GameStateView,
    type LegalAction,
  } from "./lib/api";
  import Board from "./lib/Board.svelte";
  import ActionPanel from "./lib/ActionPanel.svelte";
  import TradeForm from "./lib/TradeForm.svelte";

  // seat_kinds below always makes player 0 the human seat (seat_kinds[i]
  // fixes player i's kind -- see server/bots.py's build_agents -- it is not
  // about turn order, which the engine randomizes independently). A real
  // seat picker is step 8 (GameSetup.svelte) territory; hardcoded here.
  const VIEWER = 0;

  // Duplicated from Board.svelte's PLAYER_COLOR (4 entries -- not worth a
  // shared module for this repo's "extract only past 3 repeats" rule).
  const PLAYER_COLOR = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"];

  const POLL_INTERVAL_MS = 2000;

  let status: "loading" | "ok" | "error" = $state("loading");
  let gameId: string | null = $state(null);
  let geometry: Geometry | null = $state(null);
  let gameState: GameStateView | null = $state(null);
  let legalActions: LegalAction[] = $state([]);
  let errorMessage = $state("");
  let showTradeForm = $state(false);

  // Set only while an action HTTP request is in flight. server/app.py's
  // post_action calls server/bots.py's step_bots synchronously before
  // responding -- step_bots loops "while acting_player is a bot seat",
  // so by the time any response reaches this client, acting_player is
  // already back to the human viewer (or the game is over). There is no
  // server-observable "it's currently a bot's turn" state to poll for;
  // the one real window where bots are computing is this in-flight
  // request itself, which is what this flag actually represents.
  let isBusy = $state(false);

  const isGameOver = $derived.by(() => gameState !== null && gameState.phase === "GAME_OVER");

  let pollHandle: ReturnType<typeof setInterval> | null = null;

  function stopPolling() {
    if (pollHandle !== null) {
      clearInterval(pollHandle);
      pollHandle = null;
    }
  }

  async function refreshLegalActions() {
    if (!gameId || isGameOver) return;
    legalActions = await getLegalActions(gameId, VIEWER);
  }

  // Periodic background refresh, per the plan doc's documented design
  // (poll GET /state rather than a websocket, for a first cut). Given the
  // synchronous step_bots behavior above, this mostly guards against this
  // browser tab drifting from server state (e.g. a second tab / a future
  // multi-client viewer) rather than "watching" bot turns happen -- in
  // today's single-tab flow there is nothing new for it to observe between
  // one action's response and the next poll tick.
  function startPolling() {
    stopPolling();
    pollHandle = setInterval(async () => {
      if (!gameId || isBusy || isGameOver) return;
      try {
        gameState = await getState(gameId, VIEWER);
        await refreshLegalActions();
        if (gameState.phase === "GAME_OVER") stopPolling();
      } catch (err) {
        console.error("background poll failed", err);
      }
    }, POLL_INTERVAL_MS);
  }

  onMount(async () => {
    try {
      const created = await createGame({
        num_players: 3,
        seat_kinds: ["human", "heuristic", "heuristic"],
        seed: 1,
      });
      gameId = created.game_id;
      geometry = created.geometry;
      // createGame's own response is a spectator view (viewer=None) --
      // fetch the properly redacted view for our seat instead, so another
      // seat's hand is never briefly shown before the first action.
      gameState = await getState(gameId, VIEWER);
      status = "ok";
      await refreshLegalActions();
      if (!isGameOver) startPolling();
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
      status = "error";
    }
  });

  onDestroy(stopPolling);

  async function selectAction(
    index: number,
    give?: Record<string, number>,
    receive?: Record<string, number>,
  ) {
    if (!gameId || isBusy) return;
    isBusy = true;
    try {
      gameState = await postAction(gameId, { index, give, receive });
      errorMessage = "";
      showTradeForm = false;
      if (isGameOver) {
        stopPolling();
        legalActions = [];
      } else {
        await refreshLegalActions();
      }
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      isBusy = false;
    }
  }

  function submitTrade(give: Record<string, number>, receive: Record<string, number>) {
    const sentinel = legalActions.find((a) => a.kind === "ProposeTrade");
    if (!sentinel) return;
    void selectAction(sentinel.index, give, receive);
  }
</script>

<main>
  <h1>Catan</h1>

  {#if status === "loading"}
    <p>Loading...</p>
  {:else if status === "error"}
    <p style="color: red">Error: {errorMessage}</p>
  {:else if geometry && gameState}
    {#if isGameOver}
      <div class="game-over">
        <h2>Game over</h2>
        {#if gameState.winner !== null}
          <p>
            <span
              class="swatch"
              style="background: {PLAYER_COLOR[gameState.winner]}"
            ></span>
            Player {gameState.winner} wins!
          </p>
        {:else}
          <p>Game over.</p>
        {/if}
      </div>
      <Board {geometry} state={gameState} />
    {:else}
      <p>
        Phase: {gameState.phase}, acting player: {gameState.acting_player}
        {#if isBusy}
          <em>-- applying your move (any bot turns resolve automatically)...</em>
        {/if}
      </p>
      {#if errorMessage}
        <p style="color: red">{errorMessage}</p>
      {/if}
      <div class="layout" class:busy={isBusy}>
        <div class="board-column">
          <Board
            {geometry}
            state={gameState}
            legalActions={isBusy ? [] : legalActions}
            onSelect={(index) => selectAction(index)}
          />
        </div>
        <div class="panel-column">
          <ActionPanel
            legalActions={isBusy ? [] : legalActions}
            onSelect={(index) => selectAction(index)}
            onProposeTrade={() => (showTradeForm = true)}
          />
          {#if showTradeForm}
            <TradeForm
              humanResources={gameState.players[VIEWER].resources}
              onSubmit={submitTrade}
              onCancel={() => (showTradeForm = false)}
            />
          {/if}
        </div>
      </div>
    {/if}
  {/if}
</main>

<style>
  .layout {
    display: flex;
    gap: 1.5rem;
    align-items: flex-start;
  }

  .layout.busy {
    opacity: 0.6;
    pointer-events: none;
  }

  .board-column {
    flex: 1;
    min-width: 0;
  }

  .panel-column {
    flex: 0 0 auto;
  }

  .game-over {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
  }

  .swatch {
    display: inline-block;
    width: 0.9em;
    height: 0.9em;
    border-radius: 50%;
    border: 1px solid #000;
    vertical-align: middle;
  }
</style>
