<script lang="ts">
  import { createGame, type CreateGameResponse } from "./lib/api";

  let status: "idle" | "loading" | "ok" | "error" = "idle";
  let result: CreateGameResponse | null = null;
  let errorMessage = "";

  async function testCreateGame() {
    status = "loading";
    errorMessage = "";
    try {
      result = await createGame({
        num_players: 3,
        seat_kinds: ["human", "heuristic", "heuristic"],
        seed: 1,
      });
      status = "ok";
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
      status = "error";
    }
  }
</script>

<main>
  <h1>Catan</h1>
  <p>Step 4 skeleton -- board rendering lands in a later step.</p>

  <button on:click={testCreateGame} disabled={status === "loading"}>
    POST /api/games
  </button>

  {#if status === "loading"}
    <p>Loading...</p>
  {:else if status === "ok" && result}
    <p>Created game <code>{result.game_id}</code></p>
    <p>Phase: {result.state.phase}, acting player: {result.state.acting_player}</p>
  {:else if status === "error"}
    <p style="color: red">Error: {errorMessage}</p>
  {/if}
</main>
