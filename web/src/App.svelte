<script lang="ts">
  import { onMount } from "svelte";
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
  // seat picker is step 7 territory; hardcoded here for step 6's wiring.
  const VIEWER = 0;

  let status: "loading" | "ok" | "error" = $state("loading");
  let gameId: string | null = $state(null);
  let geometry: Geometry | null = $state(null);
  let gameState: GameStateView | null = $state(null);
  let legalActions: LegalAction[] = $state([]);
  let errorMessage = $state("");
  let showTradeForm = $state(false);

  async function refreshLegalActions() {
    if (!gameId) return;
    legalActions = await getLegalActions(gameId, VIEWER);
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
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
      status = "error";
    }
  });

  async function selectAction(
    index: number,
    give?: Record<string, number>,
    receive?: Record<string, number>,
  ) {
    if (!gameId) return;
    try {
      gameState = await postAction(gameId, { index, give, receive });
      errorMessage = "";
      showTradeForm = false;
      await refreshLegalActions();
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
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
    <p>Phase: {gameState.phase}, acting player: {gameState.acting_player}</p>
    {#if errorMessage}
      <p style="color: red">{errorMessage}</p>
    {/if}
    <div class="layout">
      <div class="board-column">
        <Board
          {geometry}
          state={gameState}
          {legalActions}
          onSelect={(index) => selectAction(index)}
        />
      </div>
      <div class="panel-column">
        <ActionPanel
          {legalActions}
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
</main>

<style>
  .layout {
    display: flex;
    gap: 1.5rem;
    align-items: flex-start;
  }

  .board-column {
    flex: 1;
    min-width: 0;
  }

  .panel-column {
    flex: 0 0 auto;
  }
</style>
