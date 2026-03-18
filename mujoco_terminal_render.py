#!/usr/bin/env python3
"""MuJoCo renderer that outputs to terminal via Kitty image protocol or ASCII fallback."""

import sys
import os
import io
import time
import base64
import shutil
import argparse

import mujoco
import numpy as np
from PIL import Image


def supports_kitty():
    """Check if terminal supports Kitty image protocol."""
    return bool(os.environ.get("KITTY_WINDOW_ID") or os.environ.get("TERM") == "xterm-kitty")


def render_frame(model, data, renderer, width, height):
    """Render a single frame from MuJoCo and return as PIL Image."""
    renderer.update_scene(data)
    pixels = renderer.render()
    return Image.fromarray(pixels)


def display_kitty(img, frame_id=0):
    """Display image using Kitty image protocol."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")

    # Kitty image protocol: transmit + display, overwriting previous frame
    # a=T: transmit and display, f=100: PNG, i=1: image id, q=2: suppress responses
    chunk_size = 4096
    chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)]

    # Move cursor to top-left for overwrite
    if frame_id > 0:
        sys.stdout.write("\033[H")

    for idx, chunk in enumerate(chunks):
        m = 1 if idx < len(chunks) - 1 else 0
        if idx == 0:
            sys.stdout.write(f"\033_Ga=T,f=100,i=1,q=2,m={m};{chunk}\033\\")
        else:
            sys.stdout.write(f"\033_Gm={m};{chunk}\033\\")

    sys.stdout.flush()


def pixels_to_ascii(img, cols):
    """Convert PIL Image to ASCII art."""
    ascii_chars = " .:-=+*#%@"
    img_gray = img.convert("L")
    w, h = img_gray.size
    aspect = h / w
    rows = int(cols * aspect * 0.45)  # terminal chars are ~2x tall as wide
    img_resized = img_gray.resize((cols, rows))
    pixels = np.array(img_resized)
    # Map pixel values to ASCII characters
    indices = (pixels / 255 * (len(ascii_chars) - 1)).astype(int)
    lines = []
    for row in indices:
        lines.append("".join(ascii_chars[i] for i in row))
    return "\n".join(lines)


def display_ascii(img, cols, frame_id=0):
    """Display image as ASCII art."""
    art = pixels_to_ascii(img, cols)
    if frame_id > 0:
        # Move cursor up to overwrite previous frame
        n_lines = art.count("\n") + 1
        sys.stdout.write(f"\033[{n_lines}A")
    sys.stdout.write(art + "\n")
    sys.stdout.flush()


def pixels_to_halfblock(img, cols):
    """Convert PIL Image to half-block colored characters.

    Each terminal cell represents 2 vertical pixels using the upper half block
    character (▀). The foreground color is the top pixel, background is the bottom.
    """
    w, h = img.size
    aspect = h / w
    # Each row of characters covers 2 pixel rows
    char_rows = int(cols * aspect * 0.45)
    pixel_rows = char_rows * 2
    img_resized = img.resize((cols, pixel_rows))
    pixels = np.array(img_resized)

    lines = []
    for y in range(0, pixel_rows, 2):
        line = []
        top_row = pixels[y]
        bot_row = pixels[y + 1] if y + 1 < pixel_rows else top_row
        for x in range(cols):
            tr, tg, tb = int(top_row[x][0]), int(top_row[x][1]), int(top_row[x][2])
            br, bg, bb = int(bot_row[x][0]), int(bot_row[x][1]), int(bot_row[x][2])
            # ▀ with fg=top pixel, bg=bottom pixel
            line.append(f"\033[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m\u2580")
        lines.append("".join(line) + "\033[0m")
    return "\n".join(lines)


def display_halfblock(img, cols, frame_id=0):
    """Display image using half-block colored characters."""
    art = pixels_to_halfblock(img, cols)
    if frame_id > 0:
        n_lines = art.count("\n") + 1
        sys.stdout.write(f"\033[{n_lines}A")
    sys.stdout.write(art + "\n")
    sys.stdout.flush()


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

    <!-- Pivot point -->
    <site name="pivot" pos="0 0 1.5" size="0.03" rgba="1 1 1 1"/>

    <body name="upper" pos="0 0 1.5">
      <joint name="hinge1" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" rgba="0.9 0.3 0.1 1" mass="1"/>

      <body name="lower" pos="0 0 -0.5">
        <joint name="hinge2" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" rgba="0.2 0.6 0.9 1" mass="1"/>

        <!-- Bob -->
        <body name="bob" pos="0 0 -0.5">
          <geom type="sphere" size="0.08" rgba="0.9 0.8 0.1 1" mass="2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser(description="MuJoCo terminal renderer")
    parser.add_argument("--mode", choices=["auto", "kitty", "block", "ascii"], default="auto",
                        help="Render mode: kitty (image protocol), block (half-block color), ascii (grayscale). Default: auto-detect")
    parser.add_argument("--width", type=int, default=640, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=480, help="Render height in pixels")
    parser.add_argument("--cols", type=int, default=None, help="ASCII art columns (default: terminal width)")
    parser.add_argument("--fps", type=float, default=30, help="Target FPS")
    parser.add_argument("--duration", type=float, default=10, help="Duration in seconds (0=infinite)")
    parser.add_argument("--xml", type=str, default=None, help="Path to MuJoCo XML model")
    parser.add_argument("--scene", type=str, default="pendulum", choices=list(SCENES.keys()),
                        help="Built-in scene (default: pendulum)")
    args = parser.parse_args()

    # Determine render mode
    if args.mode == "auto":
        if supports_kitty():
            render_mode = "kitty"
        else:
            render_mode = "block"
    else:
        render_mode = args.mode

    if render_mode != "kitty" and args.cols is None:
        args.cols = min(shutil.get_terminal_size().columns, 120)

    mode_names = {"kitty": "Kitty image protocol", "block": f"half-block color ({args.cols} cols)", "ascii": f"ASCII ({args.cols} cols)"}
    mode_name = mode_names[render_mode]
    print(f"MuJoCo Terminal Renderer — {mode_name}")
    print(f"Rendering at {args.width}x{args.height}, {args.fps} FPS")
    print("Press Ctrl+C to stop\n")
    time.sleep(1)

    # Clear screen
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    # Load model
    if args.xml:
        model = mujoco.MjModel.from_xml_path(args.xml)
    else:
        model = mujoco.MjModel.from_xml_string(SCENES[args.scene])

    data = mujoco.MjData(model)

    # Start pendulum displaced so it swings
    if not args.xml and args.scene == "pendulum":
        data.qpos[0] = np.pi / 2  # upper arm 90 degrees
        data.qpos[1] = np.pi / 4  # lower arm 45 degrees
        mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Simulation loop
    frame_interval = 1.0 / args.fps
    steps_per_frame = max(1, int(frame_interval / model.opt.timestep))
    frame_id = 0
    t_start = time.time()

    try:
        while True:
            # Step physics
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            # Render
            img = render_frame(model, data, renderer, args.width, args.height)

            if render_mode == "kitty":
                display_kitty(img, frame_id)
            elif render_mode == "block":
                display_halfblock(img, args.cols, frame_id)
            else:
                display_ascii(img, args.cols, frame_id)

            frame_id += 1

            # Timing
            elapsed = time.time() - t_start
            if args.duration > 0 and elapsed >= args.duration:
                break

            expected = frame_id * frame_interval
            sleep_time = expected - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - t_start
    avg_fps = frame_id / elapsed if elapsed > 0 else 0
    # Move below rendered content
    sys.stdout.write("\n\n")
    print(f"Done — {frame_id} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS avg)")
    renderer.close()


if __name__ == "__main__":
    main()
