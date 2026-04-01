"""Compatibility shim -- use ``import mujoco_terminal`` instead."""
from mujoco_terminal import *  # noqa: F401,F403
from mujoco_terminal import (
    render_frame, display_halfblock, display_kitty_png, display_ascii,
    launch, supports_kitty, KITTY_ENCODINGS, RawTerminal, CameraController,
)
