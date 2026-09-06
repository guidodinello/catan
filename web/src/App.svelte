<script lang="ts">
  import { onDestroy } from "svelte";
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
  import GameSetup, { type NewGameConfig } from "./lib/GameSetup.svelte";
  import HandSummary from "./lib/HandSummary.svelte";

  // Duplicated from Board.svelte's PLAYER_COLOR (4 entries -- not worth a
  // shared module for this repo's "extract only past 3 repeats" rule).
  const PLAYER_COLOR = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"];

  const POLL_INTERVAL_MS = 2000;

  let status: "setup" | "loading" | "ok" | "error" = $state("setup");
  let gameId: string | null = $state(null);
  let geometry: Geometry | null = $state(null);
  let gameState: GameStateView | null = $state(null);
  let legalActions: LegalAction[] = $state([]);
  let errorMessage = $state("");
  let tradeResultMessage = $state("");
  let showTradeForm = $state(false);

  // Index of the one human seat this browser tab plays, per
  // GameSetup.svelte's chosen seat_kinds (server/bots.py's build_agents
  // indexes agents by player id, not by turn order). `undefined` -- no
  // "human" seat chosen -- means a bot-vs-bot spectator game: getState/
  // getLegalActions are called with no viewer (the spectator/full-reveal
  // view), and since step_bots never stops for a spectator, such a game
  // finishes entirely inside createGame's own response.
  let viewer: number | undefined = $state(undefined);

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
    legalActions = await getLegalActions(gameId, viewer);
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
        gameState = await getState(gameId, viewer);
        await refreshLegalActions();
        if (gameState.phase === "GAME_OVER") stopPolling();
      } catch (err) {
        console.error("background poll failed", err);
      }
    }, POLL_INTERVAL_MS);
  }

  onDestroy(stopPolling);

  async function startGame(config: NewGameConfig) {
    status = "loading";
    errorMessage = "";
    const humanSeat = config.seat_kinds.indexOf("human");
    viewer = humanSeat === -1 ? undefined : humanSeat;
    try {
      const created = await createGame(config);
      gameId = created.game_id;
      geometry = created.geometry;
      // createGame's own response is a spectator view (viewer=None) --
      // fetch the properly redacted view for our seat instead, so another
      // seat's hand is never briefly shown before the first action.
      gameState = await getState(gameId, viewer);
      status = "ok";
      await refreshLegalActions();
      if (!isGameOver) startPolling();
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
      status = "error";
    }
  }

  function playAgain() {
    stopPolling();
    status = "setup";
    gameId = null;
    geometry = null;
    gameState = null;
    legalActions = [];
    errorMessage = "";
    showTradeForm = false;
    viewer = undefined;
  }

  // Re-syncs state/legal_actions after a rejected action -- mirrors
  // cli.py:124's "print the error and keep going" pattern, but a race
  // (this tab's `legalActions` going stale against the real server state,
  // e.g. from a second tab or a future multi-client viewer) means the
  // index that was just rejected may no longer even be the same action by
  // the time the error is shown, so re-fetching is what actually lets the
  // human recover instead of clicking the same now-wrong control again.
  async function resyncAfterError() {
    if (!gameId) return;
    try {
      gameState = await getState(gameId, viewer);
      await refreshLegalActions();
    } catch (err) {
      console.error("failed to resync after a rejected action", err);
    }
  }

  async function selectAction(
    index: number,
    give?: Record<string, number>,
    receive?: Record<string, number>,
  ) {
    if (!gameId || isBusy) return;
    isBusy = true;
    // ProposeTrade resolves entirely inside step_bots before this response
    // comes back (engine/game.py's _trade_response_apply clears
    // trade_offer/trade_responders either way), so acceptance vs. every bot
    // rejecting looks identical in the returned state unless we diff the
    // viewer's own hand against what it was right before the request.
    const isTrade = give !== undefined && receive !== undefined;
    const priorResources =
      isTrade && viewer !== undefined ? { ...gameState?.players[viewer]?.resources } : undefined;
    try {
      gameState = await postAction(gameId, { index, give, receive });
      errorMessage = "";
      showTradeForm = false;
      tradeResultMessage = isTrade
        ? describeTradeOutcome(priorResources, give, receive)
        : "";
      if (isGameOver) {
        stopPolling();
        legalActions = [];
      } else {
        await refreshLegalActions();
      }
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
      await resyncAfterError();
    } finally {
      isBusy = false;
    }
  }

  function describeTradeOutcome(
    priorResources: Record<string, number> | undefined,
    give: Record<string, number>,
    receive: Record<string, number>,
  ): string {
    if (!priorResources || viewer === undefined || !gameState) {
      return "Trade proposed.";
    }
    const nowResources = gameState.players[viewer].resources ?? {};
    const expectedAfterAccept = { ...priorResources };
    for (const [r, count] of Object.entries(give)) {
      expectedAfterAccept[r] = (expectedAfterAccept[r] ?? 0) - count;
    }
    for (const [r, count] of Object.entries(receive)) {
      expectedAfterAccept[r] = (expectedAfterAccept[r] ?? 0) + count;
    }
    const accepted = Object.keys(expectedAfterAccept).every(
      (r) => (nowResources[r] ?? 0) === expectedAfterAccept[r],
    );
    return accepted ? "Trade accepted!" : "No one accepted your trade offer.";
  }

  function submitTrade(give: Record<string, number>, receive: Record<string, number>) {
    const sentinel = legalActions.find((a) => a.kind === "ProposeTrade");
    if (!sentinel) return;
    void selectAction(sentinel.index, give, receive);
  }
</script>

<main>
  <h1>Catan</h1>

  {#if status === "setup"}
    <GameSetup onCreate={startGame} />
  {:else if status === "loading"}
    <p>Loading...</p>
  {:else if status === "error"}
    <p class="error">
      Error: {errorMessage}
      <button onclick={playAgain}>Back to setup</button>
    </p>
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
        <button onclick={playAgain}>New game</button>
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
        <p class="error">
          {errorMessage}
          <button onclick={() => (errorMessage = "")}>&times;</button>
        </p>
      {/if}
      {#if tradeResultMessage}
        <p class="trade-result">
          {tradeResultMessage}
          <button onclick={() => (tradeResultMessage = "")}>&times;</button>
        </p>
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
          <HandSummary
            diceRoll={gameState.dice_roll}
            resources={viewer !== undefined ? gameState.players[viewer].resources : undefined}
          />
          <ActionPanel
            legalActions={isBusy ? [] : legalActions}
            onSelect={(index) => selectAction(index)}
            onProposeTrade={() => (showTradeForm = true)}
          />
          {#if showTradeForm}
            <TradeForm
              humanResources={viewer !== undefined
                ? gameState.players[viewer].resources
                : undefined}
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

  .error {
    color: #d90429;
  }

  .trade-result {
    color: #1d3557;
    font-weight: bold;
  }
</style>
