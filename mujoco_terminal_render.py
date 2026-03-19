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
import argparse

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


# ── Orbit controllers ────────────────────────────────────────────────────────

class DirectOrbit:
    """Direct 1:1 mapping from mouse delta to camera angle. No smoothing."""

    def __init__(self, sensitivity=0.5):
        self.sensitivity = sensitivity
        self.dragging = False
        self.last_col = 0
        self.last_row = 0

    def handle_event(self, button, col, row, pressed, camera):
        if isinstance(button, str):
            return

        is_left = (button & 0x03) == 0
        is_motion = (button & 32) != 0

        if is_left and pressed and not is_motion:
            # Mouse down
            self.dragging = True
            self.last_col = col
            self.last_row = row
        elif is_left and is_motion and pressed and self.dragging:
            # Drag
            dx = col - self.last_col
            dy = row - self.last_row
            camera.azimuth -= dx * self.sensitivity
            camera.elevation -= dy * self.sensitivity
            camera.elevation = max(-90, min(90, camera.elevation))
            self.last_col = col
            self.last_row = row
        elif not pressed and (button & 0x03) == 0:
            # Mouse up
            self.dragging = False


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


# ── Scenes ────────────────────────────────────────────────────────────────────

SCENES = {}

SCENES["drop"] = """
<mujoco model="demo">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.8 0.8 0.8"/>
  </visual>
  <worldbody>
    <light pos="0 -1 2" dir="0 1 -1" diffuse="1 1 1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body name="ball" pos="0 0 1.5">
      <joint type="free"/>
      <geom type="sphere" size="0.1" rgba="0.9 0.2 0.2 1" mass="1"/>
    </body>
    <body name="box" pos="0.4 0 0.15">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.2 0.6 0.9 1" mass="0.5"/>
    </body>
    <body name="capsule" pos="-0.3 0.2 0.8">
      <joint type="free"/>
      <geom type="capsule" size="0.06" fromto="0 0 -0.15 0 0 0.15" rgba="0.2 0.9 0.3 1" mass="0.8"/>
    </body>
  </worldbody>
</mujoco>
"""

SCENES["pendulum"] = """
<mujoco model="pendulum">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>
  <worldbody>
    <light pos="0 -2 3" dir="0 1 -1" diffuse="1 1 1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.25 0.25 0.3 1"/>
    <site name="pivot" pos="0 0 1.5" size="0.03" rgba="1 1 1 1"/>
    <body name="upper" pos="0 0 1.5">
      <joint name="hinge1" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" rgba="0.9 0.3 0.1 1" mass="1"/>
      <body name="lower" pos="0 0 -0.5">
        <joint name="hinge2" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" rgba="0.2 0.6 0.9 1" mass="1"/>
        <body name="bob" pos="0 0 -0.5">
          <geom type="sphere" size="0.08" rgba="0.9 0.8 0.1 1" mass="2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MuJoCo terminal renderer")
    parser.add_argument("--mode", choices=["auto", "kitty", "block", "ascii"], default="auto",
                        help="Render mode (default: auto-detect)")
    parser.add_argument("--width", type=int, default=640, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=480, help="Render height in pixels")
    parser.add_argument("--cols", type=int, default=None, help="Terminal columns (default: auto)")
    parser.add_argument("--fps", type=float, default=60, help="Target FPS (0=uncapped)")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0=infinite)")
    parser.add_argument("--encoding", choices=list(KITTY_ENCODINGS.keys()), default="raw-zlib",
                        help="Kitty image encoding: png (default), png-fast (no compression), raw (uncompressed RGB), raw-zlib (zlib-compressed RGB)")
    parser.add_argument("--xml", type=str, default=None, help="Path to MuJoCo XML model")
    parser.add_argument("--scene", type=str, default="pendulum", choices=list(SCENES.keys()),
                        help="Built-in scene (default: pendulum)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Output JSON benchmark stats to stderr on exit")
    args = parser.parse_args()

    # Determine render mode
    if args.mode == "auto":
        render_mode = "kitty" if supports_kitty() else "block"
    else:
        render_mode = args.mode

    dynamic_cols = args.cols is None
    if dynamic_cols and render_mode != "kitty":
        args.cols = shutil.get_terminal_size().columns - 1
    prev_term_size = shutil.get_terminal_size()

    mode_names = {
        "kitty": "Kitty image protocol",
        "block": f"half-block color ({args.cols} cols)",
        "ascii": f"ASCII ({args.cols} cols)",
    }
    enc_label = f" [{args.encoding}]" if render_mode == "kitty" else ""
    print(f"MuJoCo Terminal Renderer — {mode_names[render_mode]}{enc_label}")
    print(f"Drag to orbit | Space=pause | R=reset | Q=quit")
    time.sleep(1)

    # Load model
    if args.xml:
        model = mujoco.MjModel.from_xml_path(args.xml)
    else:
        model = mujoco.MjModel.from_xml_string(SCENES[args.scene])

    # Ensure offscreen framebuffer is large enough for requested resolution
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

    data = mujoco.MjData(model)

    if not args.xml and args.scene == "pendulum":
        data.qpos[0] = np.pi / 2
        data.qpos[1] = np.pi / 4
        mujoco.mj_forward(model, data)

    # Save initial state for reset
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Set up camera
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.azimuth = 90
    camera.elevation = -20
    camera.lookat[:] = [0, 0, 0.8]

    # Kitty encoding function
    kitty_display = KITTY_ENCODINGS[args.encoding]

    # Orbit controller
    orbit = DirectOrbit(sensitivity=ORBIT_SENSITIVITY)

    frame_interval = 1.0 / args.fps if args.fps > 0 else 0
    # Physics runs at real-time regardless of render FPS
    physics_dt = model.opt.timestep
    frame_id = 0
    paused = False
    sim_time = 0.0
    t_start = time.perf_counter()
    t_last_frame = t_start
    fps_smooth = args.fps  # exponential moving average
    frame_times = [] if args.benchmark else None
    resize_pending = None

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
                            orbit.handle_event(*ev, camera)

                # Step physics to keep up with real time
                if not paused:
                    wall_elapsed = time.perf_counter() - t_start
                    while sim_time < wall_elapsed:
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
                        args.cols = cur_term_size.columns - 1
                    sys.stdout.write("\033[2J\033[H")
                    sys.stdout.flush()
                    frame_id = 0

                # Cap cols so rendered height fits in terminal
                term_rows = cur_term_size.lines - 1  # leave room for status
                img_aspect = args.height / args.width

                pixels = render_frame(model, data, renderer, camera)

                if render_mode == "kitty":
                    max_cols = int(term_rows / img_aspect * 2)
                    display_cols = min(cur_term_size.columns, max_cols)
                    kitty_display(pixels, frame_id, display_cols=display_cols)
                elif render_mode == "block":
                    # halfblock: rows = cols * aspect * 0.45, each row = 2 pixel rows
                    max_cols = int(term_rows / (img_aspect * 0.45))
                    cols = min(args.cols, max_cols)
                    display_halfblock(pixels, cols, frame_id)
                else:
                    # ascii: rows = cols * aspect * 0.45
                    max_cols = int(term_rows / (img_aspect * 0.45))
                    cols = min(args.cols, max_cols)
                    display_ascii(pixels, cols, frame_id)

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
                if paused:
                    status += "| PAUSED "
                # Write status line below render area
                sys.stdout.write(f"\033[0m\033[K{status}\r")
                sys.stdout.flush()
                t_last_frame = t_now

                frame_id += 1

                # Timing
                if args.duration > 0 and wall_elapsed >= args.duration:
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
            "encoding": args.encoding if render_mode == "kitty" else None,
            "resolution": f"{args.width}x{args.height}",
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


if __name__ == "__main__":
    main()
