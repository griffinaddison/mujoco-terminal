#!/usr/bin/env python3
"""MuJoCo renderer that outputs to terminal via Kitty image protocol or ASCII fallback.
Supports interactive camera orbit via click-and-drag."""

import sys
import os
import io
import time
import base64
import shutil
import select
import termios
import tty
import mujoco
import numpy as np
from PIL import Image


# ── Terminal helpers ──────────────────────────────────────────────────────────

def supports_kitty():
    return bool(os.environ.get("KITTY_WINDOW_ID") or os.environ.get("TERM") == "xterm-kitty")


class RawTerminal:
    """Context manager for raw terminal mode with mouse reporting."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None
        self.is_tty = os.isatty(self.fd)

    def __enter__(self):
        if self.is_tty:
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
            # Enable SGR extended mouse reporting (button events + drag)
            sys.stdout.write("\033[?1003h")  # all motion tracking
            sys.stdout.write("\033[?1006h")  # SGR extended format
            sys.stdout.write("\033[?25l")    # hide cursor
            sys.stdout.flush()
        return self

    def __exit__(self, *args):
        if self.is_tty:
            # Disable mouse reporting, show cursor, restore terminal
            sys.stdout.write("\033[?1003l")
            sys.stdout.write("\033[?1006l")
            sys.stdout.write("\033[?25h")
            sys.stdout.write("\033[0m")
            sys.stdout.flush()
            if self.old_settings:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_available(self):
        """Read all available bytes without blocking."""
        if not self.is_tty:
            return b""
        data = b""
        while select.select([self.fd], [], [], 0)[0]:
            data += os.read(self.fd, 1024)
        return data


def parse_mouse_events(data):
    """Parse SGR mouse events from raw terminal data.

    SGR format: ESC [ < button ; col ; row M (press/motion)
                ESC [ < button ; col ; row m (release)
    Button bits: 0=left, 1=middle, 2=right, 32+=motion, 64+=scroll
    """
    events = []
    text = data.decode("utf-8", errors="ignore")
    i = 0
    while i < len(text):
        # Look for ESC [ <
        if text[i:i+3] == "\033[<":
            i += 3
            end = i
            while end < len(text) and text[end] not in ("M", "m"):
                end += 1
            if end < len(text):
                parts = text[i:end].split(";")
                if len(parts) == 3:
                    try:
                        button = int(parts[0])
                        col = int(parts[1])
                        row = int(parts[2])
                        pressed = text[end] == "M"
                        events.append((button, col, row, pressed))
                    except ValueError:
                        pass
                i = end + 1
            else:
                i += 1
        elif text[i] == "q" or text[i] == "\x03":  # q or Ctrl+C
            events.append(("quit", 0, 0, False))
            i += 1
        elif text[i] == "r":
            events.append(("reset", 0, 0, False))
            i += 1
        elif text[i] == " ":
            events.append(("pause", 0, 0, False))
            i += 1
        else:
            i += 1
    return events


# ── Camera controllers ───────────────────────────────────────────────────────

class CameraController:
    """Handles orbit, pan, zoom, body selection, and perturbation forces.

    Controls:
        Left-drag:        Orbit camera
        Right-drag:       Pan camera
        Scroll:           Zoom
        Double-click:     Select body
        Ctrl+left-drag:   Apply torque to selected body
        Ctrl+right-drag:  Apply force to selected body
    """

    def __init__(self, orbit_sensitivity=0.5, pan_sensitivity=0.02, zoom_sensitivity=0.15):
        self.orbit_sensitivity = orbit_sensitivity
        self.pan_sensitivity = pan_sensitivity
        self.zoom_sensitivity = zoom_sensitivity
        self.left_dragging = False
        self.right_dragging = False
        self.last_col = 0
        self.last_row = 0
        # Double-click detection
        self._last_click_time = 0
        self._last_click_col = 0
        self._last_click_row = 0
        self._double_click_threshold = 0.4  # seconds
        # Perturbation state
        self.ctrl_dragging = False
        self.perturb_active = False  # True while Ctrl+drag is in progress

    def handle_event(self, button, col, row, pressed, camera,
                     model=None, data=None, pert=None, scn=None, opt=None,
                     viewport_width=640, viewport_height=480,
                     display_cols=None, display_rows=None):
        if isinstance(button, str):
            return

        btn_id = button & 0x03
        is_motion = (button & 32) != 0
        is_scroll = (button & 64) != 0
        is_ctrl = (button & 16) != 0

        # Scroll wheel zoom
        if is_scroll and pressed:
            if btn_id == 0:  # scroll up
                camera.distance *= (1 - self.zoom_sensitivity)
            elif btn_id == 1:  # scroll down
                camera.distance *= (1 + self.zoom_sensitivity)
            camera.distance = max(0.1, camera.distance)
            return

        is_left = btn_id == 0
        is_right = btn_id == 2

        if pressed and not is_motion:
            # Mouse down
            if is_ctrl and pert is not None and pert.select > 0:
                # Ctrl+click with a selected body — start perturbation
                self.ctrl_dragging = True
                self.perturb_active = True
                if is_left:
                    pert.active = int(mujoco.mjtPertBit.mjPERT_ROTATE)
                elif is_right:
                    pert.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
                if scn is not None:
                    mujoco.mjv_initPerturb(model, data, scn, pert)
            else:
                # Normal click — check for double-click (body selection)
                if is_left and not is_ctrl:
                    now = time.perf_counter()
                    dt = now - self._last_click_time
                    dist = abs(col - self._last_click_col) + abs(row - self._last_click_row)
                    if dt < self._double_click_threshold and dist < 3:
                        # Double-click — select body
                        self._select_body(col, row, model, data, pert, scn, opt,
                                          camera, viewport_width, viewport_height,
                                          display_cols, display_rows)
                    self._last_click_time = now
                    self._last_click_col = col
                    self._last_click_row = row
                if is_left:
                    self.left_dragging = True
                elif is_right:
                    self.right_dragging = True
            self.last_col = col
            self.last_row = row
        elif is_motion and pressed:
            dx = col - self.last_col
            dy = row - self.last_row
            if self.ctrl_dragging and pert is not None and scn is not None:
                # Perturbation drag
                reldx = dx / viewport_height * 20  # scale for terminal cells
                reldy = -dy / viewport_height * 20
                if pert.active & int(mujoco.mjtPertBit.mjPERT_ROTATE):
                    mujoco.mjv_movePerturb(model, data, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                                           reldx, reldy, scn, pert)
                elif pert.active & int(mujoco.mjtPertBit.mjPERT_TRANSLATE):
                    mujoco.mjv_movePerturb(model, data, mujoco.mjtMouse.mjMOUSE_MOVE_V,
                                           reldx, reldy, scn, pert)
            elif self.left_dragging:
                # Orbit
                camera.azimuth -= dx * self.orbit_sensitivity
                camera.elevation -= dy * self.orbit_sensitivity
                camera.elevation = max(-90, min(90, camera.elevation))
            elif self.right_dragging:
                # Pan — move lookat in camera-local right/up directions
                az = np.radians(camera.azimuth)
                el = np.radians(camera.elevation)
                right = np.array([-np.sin(az), np.cos(az), 0])
                up = np.array([
                    -np.cos(az) * np.sin(el),
                    -np.sin(az) * np.sin(el),
                    np.cos(el),
                ])
                scale = camera.distance * self.pan_sensitivity
                camera.lookat[:] += (dx * right + dy * up) * scale
            self.last_col = col
            self.last_row = row
        elif not pressed:
            if self.ctrl_dragging:
                self.ctrl_dragging = False
                self.perturb_active = False
                if pert is not None:
                    pert.active = 0
            if is_left:
                self.left_dragging = False
            elif is_right:
                self.right_dragging = False

    def _select_body(self, col, row, model, data, pert, scn, opt, camera,
                     viewport_width, viewport_height, display_cols, display_rows):
        """Select a body under the terminal cursor via ray-casting."""
        if model is None or pert is None or scn is None or opt is None:
            return

        # Map terminal cell to normalized viewport coordinates [0, 1]
        if display_cols and display_rows:
            relx = col / display_cols
            rely = 1.0 - (row / display_rows)  # terminal y is top-down, viewport is bottom-up
        else:
            relx = col / 80
            rely = 1.0 - (row / 24)

        relx = max(0.0, min(1.0, relx))
        rely = max(0.0, min(1.0, rely))

        aspect = viewport_width / viewport_height
        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        flexid = np.zeros(1, dtype=np.int32)
        skinid = np.zeros(1, dtype=np.int32)

        body_id = mujoco.mjv_select(model, data, opt, aspect, relx, rely,
                                     scn, selpnt, geomid, flexid, skinid)
        if body_id >= 0:
            pert.select = body_id
            pert.localpos[:] = selpnt
        else:
            pert.select = 0


ORBIT_SENSITIVITY = 3.0


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_frame(model, data, renderer, camera):
    renderer.update_scene(data, camera)
    pixels = renderer.render()
    return pixels


def _kitty_chunked_write(payload, header_params, frame_id):
    """Write a kitty image protocol payload in chunks."""
    chunk_size = 4096
    parts = []
    if frame_id > 0:
        parts.append("\033[H")
    n = len(payload)
    for start in range(0, n, chunk_size):
        chunk = payload[start:start + chunk_size]
        m = 1 if start + chunk_size < n else 0
        if start == 0:
            parts.append(f"\033_G{header_params},q=2,m={m};{chunk}\033\\")
        else:
            parts.append(f"\033_Gm={m};{chunk}\033\\")
    sys.stdout.write("".join(parts))
    sys.stdout.flush()


def display_kitty_png(pixels, frame_id=0, display_cols=0):
    """Kitty via PNG. Good compression, moderate encode cost."""
    img = Image.fromarray(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    cols_param = f",c={display_cols}" if display_cols > 0 else ""
    _kitty_chunked_write(payload, f"a=T,f=100,i=1{cols_param}", frame_id)



def display_kitty_raw(pixels, frame_id=0, display_cols=0):
    """Kitty via raw RGB. No encode overhead, larger payload."""
    h, w = pixels.shape[:2]
    payload = base64.b64encode(pixels.tobytes()).decode("ascii")
    cols_param = f",c={display_cols}" if display_cols > 0 else ""
    _kitty_chunked_write(payload, f"a=T,f=24,s={w},v={h},i=1{cols_param}", frame_id)


def display_kitty_png_fast(pixels, frame_id=0, display_cols=0):
    """Kitty via PNG with zero compression. Fast encode, larger file."""
    img = Image.fromarray(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    cols_param = f",c={display_cols}" if display_cols > 0 else ""
    _kitty_chunked_write(payload, f"a=T,f=100,i=1{cols_param}", frame_id)


def display_kitty_raw_zlib(pixels, frame_id=0, display_cols=0):
    """Kitty via zlib-compressed raw RGB. Good balance of speed and size."""
    import zlib
    h, w = pixels.shape[:2]
    compressed = zlib.compress(pixels.tobytes(), level=1)
    payload = base64.b64encode(compressed).decode("ascii")
    cols_param = f",c={display_cols}" if display_cols > 0 else ""
    _kitty_chunked_write(payload, f"a=T,f=24,s={w},v={h},o=z,i=1{cols_param}", frame_id)


def display_kitty_jpeg(pixels, frame_id=0, display_cols=0):
    """Kitty via JPEG. Not in official spec but some terminals support it."""
    img = Image.fromarray(pixels)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    cols_param = f",c={display_cols}" if display_cols > 0 else ""
    _kitty_chunked_write(payload, f"a=T,f=100,i=1{cols_param}", frame_id)


KITTY_ENCODINGS = {
    "png": display_kitty_png,
    "png-fast": display_kitty_png_fast,
    "raw": display_kitty_raw,
    "raw-zlib": display_kitty_raw_zlib,
    "jpeg": display_kitty_jpeg,
}


def display_ascii(pixels, cols, frame_id=0):
    ascii_chars = " .:-=+*#%@"
    img = Image.fromarray(pixels)
    img_gray = img.convert("L")
    w, h = img_gray.size
    aspect = h / w
    rows = int(cols * aspect * 0.45)
    img_resized = img_gray.resize((cols, rows))
    arr = np.array(img_resized)
    indices = (arr / 255 * (len(ascii_chars) - 1)).astype(int)
    lines = []
    for row in indices:
        lines.append("".join(ascii_chars[i] for i in row))
    art = "\r\n".join(lines)
    if frame_id > 0:
        n_lines = len(lines)
        sys.stdout.write(f"\033[{n_lines}A\r")
    sys.stdout.write(art + "\r\n")
    sys.stdout.flush()


def display_halfblock(pixels, cols, frame_id=0):
    img = Image.fromarray(pixels)
    w, h = img.size
    aspect = h / w
    char_rows = int(cols * aspect * 0.45)
    pixel_rows = char_rows * 2
    img_resized = img.resize((cols, pixel_rows))
    arr = np.array(img_resized, dtype=np.uint8)
    # Grab top/bottom row pairs
    top = arr[0::2]   # shape (char_rows, cols, 3)
    bot = arr[1::2]
    if top.shape[0] > bot.shape[0]:
        bot = np.concatenate([bot, top[-1:]], axis=0)
    # Flatten to lists for fast iteration (avoids numpy scalar overhead)
    top_flat = top.reshape(-1, 3).tolist()
    bot_flat = bot.reshape(-1, 3).tolist()
    n_rows = top.shape[0]
    lines = []
    idx = 0
    for _ in range(n_rows):
        parts = []
        for _ in range(cols):
            tr, tg, tb = top_flat[idx]
            br, bg, bb = bot_flat[idx]
            parts.append(f"\033[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m\u2580")
            idx += 1
        lines.append("".join(parts) + "\033[0m")
    art = "\r\n".join(lines)
    if frame_id > 0:
        n_lines = len(lines)
        sys.stdout.write(f"\033[{n_lines}A\r")
    sys.stdout.write(art + "\r\n")
    sys.stdout.flush()


# ── Library API ───────────────────────────────────────────────────────────────

def launch(model, data=None, *, mode="auto", width=640, height=480, fps=60,
           encoding="raw-zlib", cols=None, camera=None):
    """Launch an interactive terminal viewer for a MuJoCo model.

    Usage:
        import mujoco
        import mujoco_terminal_render as mtr

        model = mujoco.MjModel.from_xml_path("robot.xml")
        mtr.launch(model)

        # Or with existing data / custom camera:
        data = mujoco.MjData(model)
        mtr.launch(model, data, fps=30, mode="kitty")

    Args:
        model: MjModel instance.
        data: MjData instance. Created from model if not provided.
        mode: Render mode — "auto", "kitty", "block", or "ascii".
        width: Offscreen render width in pixels.
        height: Offscreen render height in pixels.
        fps: Target FPS (0 for uncapped).
        encoding: Kitty encoding — "raw-zlib", "png", "png-fast", "raw", "jpeg".
        cols: Terminal columns for block/ascii (None = auto-fit).
        camera: MjvCamera instance. Uses a default free camera if not provided.
    """
    _run_viewer(model, data=data, mode=mode, width=width, height=height,
                fps=fps, encoding=encoding, cols=cols, camera=camera,
                benchmark=False, duration=0)


def _run_viewer(model, *, data=None, mode="auto", width=640, height=480,
                fps=60, encoding="raw-zlib", cols=None, camera=None,
                benchmark=False, duration=0):
    """Internal viewer loop shared by launch() and main()."""

    # Determine render mode
    if mode == "auto":
        render_mode = "kitty" if supports_kitty() else "block"
    else:
        render_mode = mode

    dynamic_cols = cols is None
    if dynamic_cols and render_mode != "kitty":
        cols = shutil.get_terminal_size().columns - 1
    prev_term_size = shutil.get_terminal_size()

    mode_names = {
        "kitty": "Kitty image protocol",
        "block": f"half-block color ({cols} cols)",
        "ascii": f"ASCII ({cols} cols)",
    }
    enc_label = f" [{encoding}]" if render_mode == "kitty" else ""
    print(f"MuJoCo Terminal Renderer — {mode_names[render_mode]}{enc_label}")
    print(f"L-drag=orbit | R-drag=pan | Scroll=zoom | DblClick=select body")
    print(f"Ctrl+L-drag=torque | Ctrl+R-drag=force | Space=pause | R=reset | Q=quit")
    time.sleep(1)

    # Ensure offscreen framebuffer is large enough for requested resolution
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

    # Save initial state for reset
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()

    renderer = mujoco.Renderer(model, height=height, width=width)

    # Set up camera
    if camera is None:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 3.0
        camera.azimuth = 90
        camera.elevation = -20
        camera.lookat[:] = [0, 0, 0.8]

    # Kitty encoding function
    kitty_display = KITTY_ENCODINGS[encoding]

    # Perturbation and scene objects
    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=1000)

    # Camera controller
    controller = CameraController(orbit_sensitivity=ORBIT_SENSITIVITY)

    frame_interval = 1.0 / fps if fps > 0 else 0
    physics_dt = model.opt.timestep
    frame_id = 0
    paused = False
    sim_time = 0.0
    t_start = time.perf_counter()
    t_last_frame = t_start
    fps_smooth = fps  # exponential moving average
    frame_times = [] if benchmark else None
    resize_pending = None
    cur_term_size = prev_term_size

    # Clear screen
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    with RawTerminal() as term:
        try:
            while True:
                # Read input
                raw = term.read_available()
                if raw:
                    events = parse_mouse_events(raw)
                    for ev in events:
                        if ev[0] == "quit":
                            raise KeyboardInterrupt
                        elif ev[0] == "reset":
                            data.qpos[:] = qpos0
                            data.qvel[:] = qvel0
                            sim_time = 0.0
                            mujoco.mj_forward(model, data)
                        elif ev[0] == "pause":
                            paused = not paused
                        else:
                            controller.handle_event(
                                *ev, camera,
                                model=model, data=data, pert=pert,
                                scn=scn, opt=opt,
                                viewport_width=width, viewport_height=height,
                                display_cols=cur_term_size.columns,
                                display_rows=cur_term_size.lines,
                            )

                # Update scene for perturbation ray-casting
                mujoco.mjv_updateScene(
                    model, data, opt, pert, camera,
                    mujoco.mjtCatBit.mjCAT_ALL, scn,
                )

                # Step physics to keep up with real time
                if not paused:
                    wall_elapsed = time.perf_counter() - t_start
                    while sim_time < wall_elapsed:
                        # Apply perturbation forces before each step
                        if controller.perturb_active:
                            mujoco.mjv_applyPerturbForce(model, data, pert)
                        mujoco.mj_step(model, data)
                        sim_time += physics_dt

                # Render — detect terminal resize (debounced)
                cur_term_size = shutil.get_terminal_size()
                if cur_term_size != prev_term_size:
                    prev_term_size = cur_term_size
                    resize_pending = time.perf_counter()
                if resize_pending and time.perf_counter() - resize_pending > 0.15:
                    resize_pending = None
                    if dynamic_cols:
                        cols = cur_term_size.columns - 1
                    sys.stdout.write("\033[2J\033[H")
                    sys.stdout.flush()
                    frame_id = 0

                # Cap cols so rendered height fits in terminal
                term_rows = cur_term_size.lines - 1  # leave room for status
                img_aspect = height / width

                pixels = render_frame(model, data, renderer, camera)

                if render_mode == "kitty":
                    max_cols = int(term_rows / img_aspect * 2)
                    display_cols = min(cur_term_size.columns, max_cols)
                    kitty_display(pixels, frame_id, display_cols=display_cols)
                elif render_mode == "block":
                    max_cols = int(term_rows / (img_aspect * 0.45))
                    capped_cols = min(cols, max_cols)
                    display_halfblock(pixels, capped_cols, frame_id)
                else:
                    max_cols = int(term_rows / (img_aspect * 0.45))
                    capped_cols = min(cols, max_cols)
                    display_ascii(pixels, capped_cols, frame_id)

                # Metrics
                t_now = time.perf_counter()
                dt = t_now - t_last_frame
                if frame_times is not None:
                    frame_times.append(dt)
                instant_fps = 1.0 / dt if dt > 0 else 0
                fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1
                wall_elapsed = t_now - t_start
                realtime_factor = sim_time / wall_elapsed if wall_elapsed > 0 else 0
                status = f" FPS: {fps_smooth:5.1f} | Sim: {sim_time:6.2f}s | RT: {realtime_factor:.2f}x "
                if pert.select > 0:
                    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, pert.select)
                    status += f"| Body: {body_name or pert.select} "
                if paused:
                    status += "| PAUSED "
                # Write status line below render area
                sys.stdout.write(f"\033[0m\033[K{status}\r")
                sys.stdout.flush()
                t_last_frame = t_now

                frame_id += 1

                # Timing
                if duration > 0 and wall_elapsed >= duration:
                    break

                if frame_interval > 0:
                    expected = t_start + frame_id * frame_interval
                    sleep_time = expected - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass

    elapsed = time.perf_counter() - t_start
    avg_fps = frame_id / elapsed if elapsed > 0 else 0
    sys.stdout.write("\n\n")
    print(f"Done — {frame_id} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS avg)")

    if frame_times is not None and len(frame_times) > 1:
        import json
        ft = np.array(frame_times[1:])  # skip first frame (warmup)
        fps_arr = 1.0 / ft[ft > 0]
        stats = {
            "mode": render_mode,
            "encoding": encoding,
            "resolution": f"{width}x{height}",
            "frames": len(fps_arr),
            "duration_s": round(elapsed, 2),
            "fps_mean": round(float(np.mean(fps_arr)), 2),
            "fps_std": round(float(np.std(fps_arr)), 2),
            "fps_p1": round(float(np.percentile(fps_arr, 1)), 2),
            "fps_p99": round(float(np.percentile(fps_arr, 99)), 2),
            "frame_ms_mean": round(float(np.mean(ft)) * 1000, 2),
            "frame_ms_std": round(float(np.std(ft)) * 1000, 2),
        }
        sys.stderr.write(json.dumps(stats) + "\n")

    renderer.close()


