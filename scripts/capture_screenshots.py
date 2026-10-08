"""Regenerate the README's web-UI screenshots (``docs/images/*.png``).

Reproducible by construction: both showcase games are built in-process from
fixed seeds (``ENGINE_SEED``/``DRIVER_SEED``) with ``HeuristicAgent`` in every
seat, registered in the server's session store, and then opened in a headless
browser through the UI's own "Resume a game" flow. Persistence is pointed at a
temp dir, so nothing is written into the checkout and no RL checkpoint is used.

Prerequisites (Playwright/Pillow are deliberately not project dependencies)::

    (cd web && npm ci && npm run build)
    uv run --with playwright==1.55.0 playwright install --only-shell chromium
    # --gif only: ffmpeg on PATH (e.g. /usr/bin/ffmpeg)

Run from the repo root::

    uv run --with playwright==1.55.0 --with pillow==12.3.0 \\
        python scripts/capture_screenshots.py [--gif]

``--gif`` additionally records ``docs/images/gameplay.gif``: seat 0 rolls and
builds on a seeded turn, then the bots' turns play out with the UI's paced
replay visible. It is a GIF (not mp4/webm) because GitHub READMEs render
repo-relative images but not repo-relative videos. The recording is
re-encoded with ffmpeg's two-pass palettegen/paletteuse; the content is
reproducible, the bytes are not.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import uvicorn
from PIL import Image
from playwright.sync_api import Browser, Locator, Page, sync_playwright

import server.persistence as persistence
from agents import HeuristicAgent
from engine.actions import PlaceCity, PlaceRoad, PlaceSettlement, RollDice
from engine.state import GameState, Phase, acting_player
from server.app import app
from server.bots import SeatKind, apply_to_session, build_agents, step_bots
from server.serialize import serialize_geometry
from server.sessions import GameSession, create_session

logger = logging.getLogger("capture_screenshots")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "images"

ENGINE_SEED = 7
DRIVER_SEED = 7
NUM_PLAYERS = 4
SEAT_KINDS: list[SeatKind] = ["human", "heuristic", "heuristic", "heuristic"]
HUMAN_SEAT = 0
# Seat 0 is played by a HeuristicAgent in-process up to this turn (a count of
# per-seat turns, ~4 per round), so the board already has cities, settlements
# and long roads when the browser takes over -- 7-8 VP for the leaders here.
SHOWCASE_TURN = 183
# The GIF game stops at the first turn from here on where seat 0's roll (not a
# 7) leaves a placement it can afford, so the clip always shows a build.
GIF_MIN_TURN = 130
GIF_NAME = "gameplay.gif"
GIF_WIDTH = 900
GIF_FPS = 10
# The app dims the whole board (`.layout.busy`, opacity 0.6) for as long as a
# move is being applied / the bots' paced replay is revealing, so that stretch
# is played back GIF_REPLAY_SPEEDUP x faster and the clip is mostly the
# undimmed human turn.
GIF_REPLAY_SPEEDUP = 2.0
GIF_TAIL_SECONDS = 1.5
GIF_LEAD_SECONDS = 0.5
GIF_MAX_BUILDS = 2
MAX_GIF_BYTES = 4 * 1024 * 1024

VIEWPORT = {"width": 1440, "height": 900}
TRADE_SCROLL_PX = 64
DEVICE_SCALE_FACTOR = 2
MAX_PNG_BYTES = 400 * 1024
PALETTE_COLORS = 256
SERVER_START_TIMEOUT_S = 15.0
REPLAY_TIMEOUT_MS = 90_000

# A sensible hand-picked trade for the trade-form shot: the viewer gives two of
# their most plentiful cards, wants one ore.
TRADE_RECEIVE = {"ORE": 1}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_session(
    should_stop: Callable[[GameSession], bool],
) -> tuple[str, GameSession]:
    """A seeded 4-seat game; seat 0 is driven by ``HeuristicAgent`` here.

    Stops with seat 0 about to roll on the first turn where ``should_stop``
    says so, or plays to game over if it never does. Seat 0 stays a human seat
    in the session, so the browser (viewer 0) can act for it afterwards.
    """
    agents = build_agents(SEAT_KINDS, DRIVER_SEED)
    game_id, session = create_session(NUM_PLAYERS, agents, seed=ENGINE_SEED)
    driver = HeuristicAgent(name="seat0")
    state = session.state
    step_bots(session)
    while not session.game.is_terminal(state):
        if state.phase is Phase.ROLL and should_stop(session):
            break
        actor = acting_player(state)
        assert actor == HUMAN_SEAT, f"step_bots stopped at bot seat {actor}"
        legal = session.game.legal_actions(state)
        apply_to_session(session, actor, driver.choose_action(state, legal, actor))
        step_bots(session)
    return game_id, session


def _at_turn(turn: int) -> Callable[[GameSession], bool]:
    return lambda session: session.turn_count >= turn


def _never(_: GameSession) -> bool:
    return False


def _roll_enables_build(session: GameSession) -> bool:
    """Whether seat 0's roll this turn is not a 7 and leaves a placement."""
    if session.turn_count < GIF_MIN_TURN:
        return False
    after = session.state.copy()  # copies the RNG too: the real roll is unchanged
    session.game.apply_action(after, RollDice())
    if after.phase is not Phase.MAIN:
        return False
    return any(
        isinstance(a, PlaceRoad | PlaceSettlement | PlaceCity)
        for a in session.game.legal_actions(after)
    )


def _start_server() -> uvicorn.Server:
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while not server.started:
        if time.monotonic() > deadline:
            raise TimeoutError("uvicorn did not start")
        time.sleep(0.05)
    return server


def _base_url(server: uvicorn.Server) -> str:
    port = server.servers[0].sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _settle(page: Page) -> None:
    """Wait out the paced bot-turn reveal (skipping it), then network idle."""
    skip = page.get_by_role("button", name="Skip")
    if skip.count():
        skip.first.click()
    page.wait_for_selector("em:has-text('applying your move')", state="detached")
    page.wait_for_load_state("networkidle")


def _scroll_panels(page: Page, *, to_bottom: bool) -> None:
    """Pin the side panel to its top or bottom and the window to a fixed
    offset, so a shot never depends on where the last click happened to
    scroll. The bottom variant nudges the window down by ``TRADE_SCROLL_PX``,
    which brings the trade form's Propose/Cancel buttons fully into frame.
    """
    page.evaluate(
        """([toBottom, windowOffset]) => {
            window.scrollTo(0, toBottom ? windowOffset : 0);
            const col = document.querySelector(".panel-column");
            col.scrollTop = toBottom ? col.scrollHeight : 0;
        }""",
        [to_bottom, TRADE_SCROLL_PX],
    )


def _snap_log_to_whole_lines(page: Page) -> None:
    """Show only whole lines in the activity log.

    The list is a fixed-height scroll area pinned to its newest entry, so
    whichever line sits at the top edge is normally cut mid-line -- ordinary
    scroll-container behaviour, not an app bug. Shrink the list (inline style,
    capture only) to the whole lines that fit and re-pin it to the bottom.
    """
    page.evaluate(
        """() => {
            const ul = document.querySelector(".activity-log ul");
            const gap = parseFloat(getComputedStyle(ul).rowGap) || 0;
            let used = 0;
            for (let i = ul.children.length - 1; i >= 0; i--) {
                const next = used + ul.children[i].offsetHeight + (used ? gap : 0);
                if (next > ul.clientHeight) break;
                used = next;
            }
            ul.style.maxHeight = `${used}px`;
            ul.scrollTop = ul.scrollHeight;
        }"""
    )


def _click_action(page: Page, kind: str) -> None:
    page.get_by_role("button", name=kind, exact=True).click()
    _settle(page)


def _save_png(target: Page | Locator, name: str) -> None:
    raw = target.screenshot()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image = image.quantize(colors=PALETTE_COLORS, dither=Image.Dither.NONE)
    path = OUT_DIR / name
    image.save(path, optimize=True)
    size = path.stat().st_size
    logger.info("wrote %s (%d KB)", path.relative_to(ROOT), size // 1024)
    if size > MAX_PNG_BYTES:
        raise RuntimeError(f"{name} is {size} bytes, over the {MAX_PNG_BYTES} cap")


def _resume_script(game_id: str) -> str:
    """localStorage the Resume form reads (``web/src/lib/lastGame.ts``)."""
    entries = {
        "catan:lastGameId": game_id,
        "catan:lastViewer": str(HUMAN_SEAT),
        "catan:cachedGeometry": json.dumps(serialize_geometry()),
    }
    return "".join(
        f"localStorage.setItem({json.dumps(k)}, {json.dumps(v)});"
        for k, v in entries.items()
    )


def _capture(
    base_url: str, mid_id: str, over_id: str, gif_id: str | None, tmp: Path
) -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            # Fresh browser: the new-game screen.
            setup_ctx = browser.new_context(
                viewport=VIEWPORT, device_scale_factor=DEVICE_SCALE_FACTOR
            )
            page = setup_ctx.new_page()
            page.goto(base_url)
            page.get_by_role("heading", name="New game").wait_for()
            page.wait_for_load_state("networkidle")
            _save_png(page, "setup.png")
            setup_ctx.close()

            ctx = browser.new_context(
                viewport=VIEWPORT, device_scale_factor=DEVICE_SCALE_FACTOR
            )
            ctx.add_init_script(_resume_script(mid_id))
            page = ctx.new_page()
            page.set_default_timeout(REPLAY_TIMEOUT_MS)
            page.goto(base_url)
            page.get_by_role("button", name="Resume game").click()
            page.get_by_role("heading", name="Victory points").wait_for()

            # One full turn through the UI so the activity log has content,
            # then roll again and stop in the post-roll main phase.
            _click_action(page, "RollDice")
            _click_action(page, "EndTurn")
            _click_action(page, "RollDice")
            _scroll_panels(page, to_bottom=False)
            _snap_log_to_whole_lines(page)
            _save_png(page, "board.png")

            page.locator("details.dice-panel summary").click()
            _save_png(page.locator(".dice-panel"), "dice.png")

            page.get_by_role("button", name="ProposeTrade").click()
            form = page.locator(".trade-form")
            form.wait_for()
            for resource, count in TRADE_RECEIVE.items():
                form.locator(
                    "div:has(> h4:text('You receive')) label", has_text=resource
                ).locator("input").fill(str(count))
            have = form.locator("div:has(> h4:text('You give')) label")
            for i in range(have.count()):
                label = have.nth(i)
                text = label.inner_text()
                if "(have 0)" not in text and "ORE" not in text:
                    label.locator("input").fill("1")
                    break
            _scroll_panels(page, to_bottom=True)
            page.mouse.move(0, 0)  # no stray hover state on a button
            _save_png(page, "trade.png")
            ctx.close()

            over_ctx = browser.new_context(
                viewport=VIEWPORT, device_scale_factor=DEVICE_SCALE_FACTOR
            )
            over_ctx.add_init_script(_resume_script(over_id))
            page = over_ctx.new_page()
            page.goto(base_url)
            page.get_by_role("button", name="Resume game").click()
            page.get_by_role("heading", name="Game over").wait_for()
            page.wait_for_load_state("networkidle")
            _save_png(page, "game-over.png")
            over_ctx.close()

            if gif_id is not None:
                _record_gif(browser, base_url, gif_id, tmp)
        finally:
            browser.close()


def _record_gif(browser: Browser, base_url: str, game_id: str, tmp: Path) -> None:
    """Roll and build with the board undimmed, then end the turn and let the
    bots' paced replay play out.
    """
    ctx = browser.new_context(
        viewport=VIEWPORT,
        record_video_dir=str(tmp / "video"),
        record_video_size={"width": GIF_WIDTH, "height": GIF_WIDTH * 5 // 8},
    )
    started = time.monotonic()
    ctx.add_init_script(_resume_script(game_id))
    page = ctx.new_page()
    page.set_default_timeout(REPLAY_TIMEOUT_MS)
    page.goto(base_url)
    page.get_by_role("button", name="Resume game").click()
    page.get_by_role("heading", name="Victory points").wait_for()
    start_s = time.monotonic() - started + GIF_LEAD_SECONDS
    page.wait_for_timeout(1200)
    _click_action(page, "RollDice")
    page.wait_for_timeout(1000)
    # Vertex targets (settlement/city) come before edge targets in the DOM, so
    # the first build is a settlement/city whenever one is affordable.
    targets = page.locator("circle.highlight-vertex, rect.edge-hit-target")
    for _ in range(GIF_MAX_BUILDS):
        if targets.count() == 0:
            break
        targets.first.hover()
        page.wait_for_timeout(800)
        targets.first.click()
        page.wait_for_selector("em:has-text('applying your move')", state="detached")
        page.wait_for_timeout(1000)
    replay_s = time.monotonic() - started
    page.get_by_role("button", name="EndTurn", exact=True).click()
    page.wait_for_selector("em:has-text('applying your move')", state="detached")
    page.wait_for_timeout(int(GIF_TAIL_SECONDS * 1000))
    end_s = time.monotonic() - started
    video = page.video
    assert video is not None
    ctx.close()  # finalizes the webm
    _encode_gif(Path(video.path()), start_s, replay_s, end_s, tmp)


def _encode_gif(
    webm: Path, start_s: float, replay_s: float, end_s: float, tmp: Path
) -> None:
    """Two-pass palettegen/paletteuse; ``[start_s, replay_s)`` at 1x, the rest
    (the bots' replay) at ``GIF_REPLAY_SPEEDUP`` x. Times are seconds into the
    recording.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("--gif needs ffmpeg on PATH")
    palette = tmp / "palette.png"
    chain = (
        f"[0:v]split=2[a][b];"
        f"[a]trim=start={start_s:.2f}:end={replay_s:.2f},setpts=PTS-STARTPTS[a1];"
        f"[b]trim=start={replay_s:.2f}:end={end_s:.2f},"
        f"setpts=(PTS-STARTPTS)/{GIF_REPLAY_SPEEDUP}[b1];"
        f"[a1][b1]concat=n=2:v=1:a=0,fps={GIF_FPS},"
        f"scale={GIF_WIDTH}:-1:flags=lanczos"
    )
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm)]
    subprocess.run(
        [*base, "-filter_complex", f"{chain},palettegen=stats_mode=diff", str(palette)],
        check=True,
    )
    out = OUT_DIR / GIF_NAME
    use = "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    subprocess.run(
        [
            *base,
            "-i",
            str(palette),
            "-filter_complex",
            f"{chain}[x];[x][1:v]{use}",
            str(out),
        ],
        check=True,
    )
    size = out.stat().st_size
    logger.info("wrote %s (%.1f MB)", out.relative_to(ROOT), size / 1024 / 1024)
    if size > MAX_GIF_BYTES:
        raise RuntimeError(f"{GIF_NAME} is {size} bytes, over the {MAX_GIF_BYTES} cap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--gif", action="store_true", help=f"also record {GIF_NAME} (needs ffmpeg)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        persistence.SESSION_DIR = Path(tmp) / "sessions"
        persistence.GAMES_DIR = Path(tmp) / "games"
        server = _start_server()
        try:
            mid_id, mid = _build_session(_at_turn(SHOWCASE_TURN))
            over_id, over = _build_session(_never)
            _log_state("mid-game", mid.state)
            _log_state("game-over", over.state)
            gif_id = None
            if args.gif:
                gif_id, gif = _build_session(_roll_enables_build)
                _log_state("gif", gif.state)
            _capture(_base_url(server), mid_id, over_id, gif_id, Path(tmp))
        finally:
            server.should_exit = True


def _log_state(label: str, state: GameState) -> None:
    logger.info(
        "%s: phase=%s acting=%d dice_history_ok",
        label,
        state.phase.name,
        acting_player(state) if state.phase is not Phase.GAME_OVER else -1,
    )


if __name__ == "__main__":
    main()
