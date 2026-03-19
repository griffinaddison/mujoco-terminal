#!/usr/bin/env python3
"""Benchmark display functions in isolation (encode/format only, output to /dev/null)."""

import sys
import os
import io
import time
import base64
import argparse

import mujoco
import numpy as np
from PIL import Image

# Redirect stdout to /dev/null for benchmarking (we only care about encode time)
class DevNull:
    def write(self, s): pass
    def flush(self): pass

# Import the display functions from the main module
sys.path.insert(0, os.path.dirname(__file__))

PENDULUM_XML = """
<mujoco model="pendulum">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual><global offwidth="640" offheight="480"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/></visual>
  <worldbody>
    <light pos="0 -2 3" dir="0 1 -1" diffuse="1 1 1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.25 0.25 0.3 1"/>
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


def bench(name, func, args, n=200, warmup=10):
    """Benchmark a function, return avg ms per call."""
    for _ in range(warmup):
        func(*args)
    t0 = time.perf_counter()
    for _ in range(n):
        func(*args)
    elapsed = time.perf_counter() - t0
    ms = elapsed / n * 1000
    return ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="Label for this run (e.g. 'before' or 'after')")
    parser.add_argument("--n", type=int, default=200, help="Iterations per benchmark")
    parser.add_argument("--cols", type=int, default=120, help="Columns for ascii/block modes")
    cli_args = parser.parse_args()

    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
    data = mujoco.MjData(model)
    data.qpos[0] = np.pi / 2
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.azimuth = 90
    camera.elevation = -20
    camera.lookat[:] = [0, 0, 0.8]

    renderer.update_scene(data, camera)
    pixels = renderer.render()

    # Sink stdout
    real_stdout = sys.stdout
    sys.stdout = DevNull()

    # Import display functions
    import mujoco_terminal_render as mtr

    results = {}

    # Kitty encodings
    for enc_name in ["png", "png-fast", "raw", "raw-zlib", "jpeg"]:
        fn = mtr.KITTY_ENCODINGS[enc_name]
        ms = bench(f"kitty-{enc_name}", fn, (pixels, 1), n=cli_args.n)
        results[f"kitty-{enc_name}"] = ms

    # Halfblock
    ms = bench("halfblock", mtr.display_halfblock, (pixels, cli_args.cols, 1), n=cli_args.n)
    results["halfblock"] = ms

    # ASCII
    ms = bench("ascii", mtr.display_ascii, (pixels, cli_args.cols, 1), n=cli_args.n)
    results["ascii"] = ms

    # Restore stdout and print results
    sys.stdout = real_stdout
    tag = f" [{cli_args.tag}]" if cli_args.tag else ""
    print(f"\nBenchmark results{tag} ({cli_args.n} iterations, 640x480)")
    print(f"{'Function':<20} {'ms/frame':>10}")
    print("-" * 32)
    for name, ms in results.items():
        print(f"{name:<20} {ms:>9.2f}ms")
    print()

    renderer.close()


if __name__ == "__main__":
    main()
