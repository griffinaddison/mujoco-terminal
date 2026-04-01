#!/usr/bin/env python3
"""Demo scenes for mujoco-terminal.

Shows how to use mujoco_terminal.launch() with built-in scenes.

Usage:
    python demo.py                          # default pendulum scene
    python demo.py --scene drop             # falling objects scene
    python demo.py --scene pendulum --mode block
    python demo.py --xml path/to/model.xml  # custom model
"""

import argparse

import mujoco
import numpy as np

import mujoco_terminal as mtr


# ── Built-in scenes ──────────────────────────────────────────────────────────

SCENES = {}

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


def main():
    parser = argparse.ArgumentParser(description="MuJoCo terminal renderer demos")
    parser.add_argument("--scene", choices=list(SCENES.keys()), default="pendulum",
                        help="Built-in scene (default: pendulum)")
    parser.add_argument("--xml", type=str, default=None, help="Path to MuJoCo XML model")
    parser.add_argument("--mode", choices=["auto", "kitty", "block", "ascii"], default="auto",
                        help="Render mode (default: auto-detect)")
    parser.add_argument("--width", type=int, default=640, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=480, help="Render height in pixels")
    parser.add_argument("--fps", type=float, default=60, help="Target FPS (0=uncapped)")
    parser.add_argument("--encoding", choices=["raw-zlib", "png", "png-fast", "raw", "jpeg"],
                        default="raw-zlib", help="Kitty encoding (default: raw-zlib)")
    args = parser.parse_args()

    # Load model
    if args.xml:
        model = mujoco.MjModel.from_xml_path(args.xml)
    else:
        model = mujoco.MjModel.from_xml_string(SCENES[args.scene])

    data = mujoco.MjData(model)

    # Set up initial conditions for pendulum
    if not args.xml and args.scene == "pendulum":
        data.qpos[0] = np.pi / 2
        data.qpos[1] = np.pi / 4
        mujoco.mj_forward(model, data)

    # Launch the terminal viewer
    mtr.launch(model, data,
               mode=args.mode,
               width=args.width,
               height=args.height,
               fps=args.fps,
               encoding=args.encoding)


if __name__ == "__main__":
    main()
