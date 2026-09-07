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
    type TrailEntry,
  } from "./lib/api";
  import Board from "./lib/Board.svelte";
  import ActionPanel from "./lib/ActionPanel.svelte";
  import ActivityLog from "./lib/ActivityLog.svelte";
  import TradeForm from "./lib/TradeForm.svelte";
  import TradeOfferBanner from "./lib/TradeOfferBanner.svelte";
  import DevCardResourceForm from "./lib/DevCardResourceForm.svelte";
  import GameSetup, { type NewGameConfig } from "./lib/GameSetup.svelte";
  import HandSummary from "./lib/HandSummary.svelte";
  import Scoreboard from "./lib/Scoreboard.svelte";
  import { firstEdgeCandidates, secondEdgeCandidates } from "./lib/roadBuilding";
  import { recentRolls as recentRollsFrom } from "./lib/actionLog";
  import { PLAYER_COLOR } from "./lib/playerColor";
  import { loadCachedGeometry, saveCachedGeometry, saveLastGame } from "./lib/lastGame";

  const POLL_INTERVAL_MS = 2000;

  // "Visible, paced bot turns" + "dice roll history" (docs/backlog.md) --
  // both share one mechanism: server/app.py's action responses now carry
  // an action_trail (every action applied while producing that response,
  // human's own first, then any consecutive bot turns). Rather than
  // jumping straight from one gameState to the next, this trail is
  // revealed into `revealedLog` one entry at a time with a pause between
  // each, so a whole bot turn doesn't complete invisibly between one
  // click and the next. There are no intermediate GameState snapshots to
  // show (only intermediate actions), so the board/panel still jump
  // straight to the final state immediately -- only this text log is
  // paced. A fuller "animate the board turn by turn" version stays a
  // distinct, larger follow-up if ever wanted.
  const REPLAY_DELAY_MS = 900;
  const LOG_CAP = 30;

  let status: "setup" | "loading" | "ok" | "error" = $state("setup");
  let gameId: string | null = $state(null);
  let geometry: Geometry | null = $state(null);
  let gameState: GameStateView | null = $state(null);
  let legalActions: LegalAction[] = $state([]);
  let errorMessage = $state("");
  let tradeResultMessage = $state("");
  let showTradeForm = $state(false);
  let showMonopolyForm = $state(false);
  let showYearOfPlentyForm = $state(false);

  // PlayRoadBuilding's two-click board pick (see lib/roadBuilding.ts):
  // roadBuildingFirstEdge is null while awaiting the first click, set once
  // the first edge is chosen, and reset (along with roadBuildingActive)
  // whenever the flow completes, is cancelled, or an action successfully
  // posts.
  let roadBuildingActive = $state(false);
  let roadBuildingFirstEdge: number | null = $state(null);

  function cancelRoadBuilding() {
    roadBuildingActive = false;
    roadBuildingFirstEdge = null;
  }

  const edgePickHandlers = $derived.by(() => {
    if (!roadBuildingActive) return undefined;
    const map = new Map<number, () => void>();
    if (roadBuildingFirstEdge === null) {
      for (const edgeId of firstEdgeCandidates(legalActions)) {
        map.set(edgeId, () => (roadBuildingFirstEdge = edgeId));
      }
    } else {
      for (const [edgeId, index] of secondEdgeCandidates(legalActions, roadBuildingFirstEdge)) {
        map.set(edgeId, () => selectAction(index));
      }
    }
    return map;
  });

  // Index of the one human seat this browser tab plays, per
  // GameSetup.svelte's chosen seat_kinds (server/bots.py's build_agents
  // indexes agents by player id, not by turn order). `undefined` -- no
  // "human" seat chosen -- means a bot-vs-bot spectator game: getState/
  // getLegalActions are called with no viewer (the spectator/full-reveal
  // view), and since step_bots never stops for a spectator, such a game
  // finishes entirely inside createGame's own response.
  let viewer: number | undefined = $state(undefined);

  // Every seat this browser tab could switch to viewing, in seat order --
  // populated from GameSetup.svelte's chosen seat_kinds on create (there
  // can legitimately be more than one "human" seat: server/bots.py only
  // auto-steps bot seats, so multiple human seats already worked
  // server-side, just not from one browser tab's fixed `viewer` before
  // this). Left empty after a Resume, since a resumed game's seat_kinds
  // aren't known to the client -- see resumeGame.
  let humanSeats: number[] = $state([]);

  // Hot-seat switch: re-fetches this seat's own redacted view and clears
  // any UI state that was scoped to the *previous* viewer (open forms
  // reference gameState.players[viewer], which would otherwise silently
  // point at the wrong seat's hand).
  async function switchViewer(newViewer: number) {
    if (!gameId || newViewer === viewer || isBusy) return;
    viewer = newViewer;
    showTradeForm = false;
    showMonopolyForm = false;
    showYearOfPlentyForm = false;
    cancelRoadBuilding();
    cancelAutoReject();
    tradeResultMessage = "";
    errorMessage = "";
    try {
      gameState = await getState(gameId, viewer);
      await refreshLegalActions();
      saveLastGame(gameId, viewer);
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  // Set only while an action HTTP request is in flight. server/app.py's
  // post_action calls server/bots.py's step_bots synchronously before
  // responding -- step_bots loops "while acting_player is a bot seat",
  // so by the time any response reaches this client, acting_player is
  // already back to the human viewer (or the game is over). There is no
  // server-observable "it's currently a bot's turn" state to poll for;
  // the one real window where bots are computing is this in-flight
  // request itself, which is what this flag actually represents.
  let isBusy = $state(false);

  // The paced reveal queue -- see the top-of-file comment. `pendingTrail`
  // holds entries not yet revealed; `revealedLog` (capped) holds what's
  // been shown so far and feeds both ActivityLog and HandSummary's
  // recent-rolls strip. `isBusy` stays true for the whole reveal, not just
  // the network request, per the "reuse isBusy" design -- selectAction's
  // own finally only resets it when nothing is queued to replay.
  let pendingTrail: TrailEntry[] = $state([]);
  let revealedLog: TrailEntry[] = $state([]);
  let replayTimeout: ReturnType<typeof setTimeout> | null = null;

  function cancelReplay() {
    if (replayTimeout !== null) {
      clearTimeout(replayTimeout);
      replayTimeout = null;
    }
    pendingTrail = [];
  }

  function pushToLog(entry: TrailEntry) {
    revealedLog = [...revealedLog, entry].slice(-LOG_CAP);
  }

  function advanceReplay() {
    if (pendingTrail.length === 0) {
      replayTimeout = null;
      isBusy = false;
      return;
    }
    const [next, ...rest] = pendingTrail;
    pendingTrail = rest;
    pushToLog(next);
    replayTimeout = setTimeout(advanceReplay, REPLAY_DELAY_MS);
  }

  // Queues `trail` for paced reveal; isBusy is expected to already be true
  // (selectAction/startGame/resumeGame set it before calling this) and
  // stays true until the queue empties.
  function startReplay(trail: TrailEntry[]) {
    if (trail.length === 0) return;
    pendingTrail = [...pendingTrail, ...trail];
    if (replayTimeout === null) advanceReplay();
  }

  // Reveals every still-pending entry immediately, e.g. for a long bot
  // batch (an all-bot spectator game) someone doesn't want to sit through.
  function skipReplay() {
    if (replayTimeout !== null) {
      clearTimeout(replayTimeout);
      replayTimeout = null;
    }
    for (const entry of pendingTrail) pushToLog(entry);
    pendingTrail = [];
    isBusy = false;
  }

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
    humanSeats = config.seat_kinds.flatMap((kind, i) => (kind === "human" ? [i] : []));
    viewer = humanSeats.length > 0 ? humanSeats[0] : undefined;
    try {
      const created = await createGame(config);
      gameId = created.game_id;
      geometry = created.geometry;
      saveCachedGeometry(geometry);
      // createGame's own response is a spectator view (viewer=None) --
      // fetch the properly redacted view for our seat instead, so another
      // seat's hand is never briefly shown before the first action.
      gameState = await getState(gameId, viewer);
      // No board/panel is on screen yet during "loading" -- pacing this
      // reveal wouldn't be visible to anyone, so it's dumped straight into
      // the log rather than run through startReplay/isBusy.
      revealedLog = created.action_trail.slice(-LOG_CAP);
      status = "ok";
      await refreshLegalActions();
      if (!isGameOver) startPolling();
      saveLastGame(gameId, viewer);
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
      status = "error";
    }
  }

  // "Resume a game" (docs/backlog.md) -- reconnects to a game_id already
  // running server-side (e.g. after a hard page reload, which loses all of
  // this component's state) instead of creating a new one. server/app.py
  // has no endpoint that returns geometry alone (only POST /api/games
  // does), so this relies on a previously-cached copy (see lib/lastGame.ts
  // for why that's valid for any game_id, not just the one it came from).
  async function resumeGame(resumeGameId: string, resumeViewer: number | undefined) {
    status = "loading";
    errorMessage = "";
    const cachedGeometry = loadCachedGeometry();
    if (!cachedGeometry) {
      errorMessage =
        "Can't resume yet -- this browser has no cached board layout. " +
        "Create a game once first; after that, resuming (including after a reload) will work.";
      status = "error";
      return;
    }
    viewer = resumeViewer;
    humanSeats = resumeViewer !== undefined ? [resumeViewer] : [];
    geometry = cachedGeometry;
    try {
      gameId = resumeGameId;
      gameState = await getState(gameId, viewer);
      status = "ok";
      await refreshLegalActions();
      if (!isGameOver) startPolling();
      saveLastGame(gameId, viewer);
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
      status = "error";
      gameId = null;
      geometry = null;
    }
  }

  function playAgain() {
    stopPolling();
    cancelAutoReject();
    cancelReplay();
    revealedLog = [];
    isBusy = false;
    status = "setup";
    gameId = null;
    geometry = null;
    gameState = null;
    legalActions = [];
    errorMessage = "";
    showTradeForm = false;
    showMonopolyForm = false;
    showYearOfPlentyForm = false;
    cancelRoadBuilding();
    viewer = undefined;
    humanSeats = [];
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
      const response = await postAction(gameId, { index, give, receive });
      gameState = response;
      errorMessage = "";
      showTradeForm = false;
      showMonopolyForm = false;
      showYearOfPlentyForm = false;
      cancelRoadBuilding();
      tradeResultMessage = isTrade
        ? describeTradeOutcome(priorResources, give, receive)
        : "";
      if (isGameOver) {
        stopPolling();
        legalActions = [];
      } else {
        await refreshLegalActions();
      }
      startReplay(response.action_trail);
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
      await resyncAfterError();
    } finally {
      // Deferred to advanceReplay's own end-of-queue branch when a trail
      // was just queued (startReplay set pendingTrail synchronously above,
      // before this runs) -- isBusy stays true for the whole paced reveal,
      // not just this request.
      if (pendingTrail.length === 0) isBusy = false;
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

  // engine/game.py's _trade_response_legal only offers AcceptTrade when the
  // responder actually holds the requested resources -- so if RejectTrade
  // is the *only* legal action, there is no real decision on the table,
  // just a forced outcome. Requiring a click for that is pure friction (the
  // same reasoning step_bots already applies to bot turns), so auto-submit
  // it -- but only in that exact case, never a reject the viewer could have
  // turned down deliberately.
  // A short delay before actually submitting -- so the trade-offer banner
  // and the "auto-rejecting" notice are visible long enough to read,
  // instead of flashing and vanishing the instant this effect runs.
  const AUTO_REJECT_DELAY_MS = 3000;
  let autoRejectTimeout: ReturnType<typeof setTimeout> | null = null;

  function cancelAutoReject() {
    if (autoRejectTimeout !== null) {
      clearTimeout(autoRejectTimeout);
      autoRejectTimeout = null;
    }
  }

  $effect(() => {
    const canOnlyReject =
      !isBusy &&
      gameState?.phase === "AWAIT_TRADE_RESPONSE" &&
      legalActions.length === 1 &&
      legalActions[0].kind === "RejectTrade";
    if (!canOnlyReject) {
      cancelAutoReject();
      return;
    }
    if (autoRejectTimeout !== null) return; // already counting down
    const rejectIndex = legalActions[0].index;
    tradeResultMessage = "Auto-rejecting: you don't have the resources to accept...";
    autoRejectTimeout = setTimeout(() => {
      autoRejectTimeout = null;
      void selectAction(rejectIndex);
    }, AUTO_REJECT_DELAY_MS);
  });

  onDestroy(cancelAutoReject);
</script>

<main class:playing={status === "ok"}>
  <h1>Catan</h1>

  {#if status === "setup"}
    <GameSetup onCreate={startGame} onResume={resumeGame} />
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
      <p class="phase-line">
        Phase: {gameState.phase}, turn: Player {gameState.current_player}
        {#if gameState.acting_player !== gameState.current_player}
          <!-- Distinct from current_player while awaiting a domestic-trade
               response -- e.g. it's Player 2's turn, but Player 0 (you) must
               respond to their trade offer before anything else can happen. -->
          (Player {gameState.acting_player} must respond to a trade)
        {/if}
        {#if isBusy}
          <em>-- applying your move (any bot turns resolve automatically)...</em>
          {#if pendingTrail.length > 0}
            <button onclick={skipReplay}>Skip</button>
          {/if}
        {/if}
      </p>
      {#if humanSeats.length > 1}
        <p class="viewer-switch">
          Viewing as:
          <select
            value={viewer}
            disabled={isBusy || roadBuildingActive}
            onchange={(e) => switchViewer(Number(e.currentTarget.value))}
          >
            {#each humanSeats as seat (seat)}
              <option value={seat}>Player {seat}</option>
            {/each}
          </select>
          <span class="hint">(pass-and-play -- this doesn't hide the view from anyone else at this screen)</span>
        </p>
      {/if}
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
      {#if roadBuildingActive}
        <p class="road-building-banner">
          Play Road Building:
          {roadBuildingFirstEdge === null
            ? "pick the first road on the board."
            : "now pick the second road."}
          <button onclick={cancelRoadBuilding}>Cancel</button>
        </p>
      {/if}
      {#if gameState.trade_offer}
        <TradeOfferBanner offer={gameState.trade_offer} {viewer} />
      {/if}
      <div class="layout" class:busy={isBusy}>
        <div class="board-column">
          <Board
            {geometry}
            state={gameState}
            legalActions={isBusy || roadBuildingActive ? [] : legalActions}
            onSelect={(index) => selectAction(index)}
            {edgePickHandlers}
          />
        </div>
        <div class="panel-column" class:disabled={roadBuildingActive}>
          <Scoreboard state={gameState} {viewer} />
          <ActivityLog entries={revealedLog} />
          <HandSummary
            diceRoll={gameState.dice_roll}
            recentRolls={recentRollsFrom(revealedLog)}
            resources={viewer !== undefined ? gameState.players[viewer].resources : undefined}
          />
          <ActionPanel
            legalActions={isBusy || roadBuildingActive ? [] : legalActions}
            onSelect={(index) => selectAction(index)}
            onProposeTrade={() => (showTradeForm = true)}
            onPlayRoadBuilding={() => {
              roadBuildingActive = true;
              roadBuildingFirstEdge = null;
            }}
            onPlayYearOfPlenty={() => (showYearOfPlentyForm = true)}
            onPlayMonopoly={() => (showMonopolyForm = true)}
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
          {#if showMonopolyForm}
            <DevCardResourceForm
              title="Play Monopoly"
              count={1}
              legalActions={legalActions.filter((a) => a.kind === "PlayMonopoly")}
              onSubmit={(index) => selectAction(index)}
              onCancel={() => (showMonopolyForm = false)}
            />
          {/if}
          {#if showYearOfPlentyForm}
            <DevCardResourceForm
              title="Play Year of Plenty"
              count={2}
              legalActions={legalActions.filter((a) => a.kind === "PlayYearOfPlenty")}
              onSubmit={(index) => selectAction(index)}
              onCancel={() => (showYearOfPlentyForm = false)}
            />
          {/if}
        </div>
      </div>
    {/if}
  {/if}
</main>

<style>
  /* app.css caps <main> at 720px for the setup/error screens, where a
     narrow form reads better -- but that same cap starved the board of
     room once actually playing, since Board.svelte's svg is width: 100%
     of .board-column and scales down with whatever space it's given. */
  main.playing {
    max-width: 1400px;
  }

  main h1 {
    font-size: 3rem;
    margin-bottom: 0.25rem;
  }

  .phase-line {
    font-size: 1.2rem;
  }

  .layout {
    display: flex;
    gap: 1.5rem;
    /* stretch (not the default-overriding flex-start) so .panel-column
       matches .board-column's height instead of shrink-wrapping its own
       content and leaving empty space below it next to a taller board. */
    align-items: stretch;
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
    display: flex;
    flex-direction: column;
  }

  /* The last box (normally ActionPanel, or whichever form is currently
     open below it) grows to fill the remaining stretched height from
     .layout above, so the column reaches the board's bottom without
     spreading big empty gaps between every box (space-between's effect,
     tried first and rejected -- it looked disconnected, not "aligned"). */
  .panel-column > :global(:last-child) {
    flex: 1;
  }

  .panel-column.disabled {
    opacity: 0.6;
    pointer-events: none;
  }

  .road-building-banner {
    background: #fff3cd;
    color: #664d03;
    border: 1px solid #ffd60a;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.75rem;
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

  .viewer-switch {
    margin: 0 0 0.75rem;
  }

  .viewer-switch .hint {
    color: #666;
    font-size: 0.8rem;
  }
</style>
