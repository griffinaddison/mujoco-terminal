"""MuJoCo renderer for terminal output via Kitty image protocol or ASCII fallback.

Public API
----------
render_frame(model, data, renderer, camera, scene_option=None, perturb=None)
    Render a single frame, returns numpy pixel array (H, W, 3).

display_halfblock(pixels, cols, frame_id=0)
    Display pixels as half-block Unicode art.

display_kitty_png(pixels, frame_id=0, display_cols=0)
    Display pixels via Kitty terminal image protocol (PNG encoding).

display_ascii(pixels, cols, frame_id=0)
    Display pixels as ASCII art.

TerminalRenderer
    High-level class wrapping terminal setup/teardown and rendering.

launch(model, data=None, *, ...)
    Launch an interactive terminal viewer (blocking).

launch_passive(model, data, *, ...)
    Launch a passive terminal viewer (non-blocking), returns a Handle.

Handle
    Object returned by launch_passive() with sync/close/is_running API.
"""

from mujoco_terminal._render import (
    render_frame,
    display_halfblock,
    display_kitty_png,
    display_kitty_png_fast,
    display_kitty_raw,
    display_kitty_raw_zlib,
    display_kitty_jpeg,
    display_ascii,
    launch,
    launch_passive,
    Handle,
    supports_kitty,
    KITTY_ENCODINGS,
    RawTerminal,
    CameraController,
    parse_mouse_events,
)

import sys
import shutil
import mujoco

__all__ = [
    "render_frame",
    "display_halfblock",
    "display_kitty_png",
    "display_kitty_png_fast",
    "display_kitty_raw",
    "display_kitty_raw_zlib",
    "display_kitty_jpeg",
    "display_ascii",
    "launch",
    "launch_passive",
    "Handle",
    "supports_kitty",
    "KITTY_ENCODINGS",
    "RawTerminal",
    "CameraController",
    "TerminalRenderer",
]


class TerminalRenderer:
    """High-level renderer that wraps terminal setup/teardown.

    Usage::

        import mujoco
        import mujoco_terminal

        model = mujoco.MjModel.from_xml_path("robot.xml")
        data = mujoco.MjData(model)

        with mujoco_terminal.TerminalRenderer(model) as tr:
            for _ in range(1000):
                mujoco.mj_step(model, data)
                tr.render(data)

    Args:
        model: MjModel instance.
        mode: Display mode -- "auto", "kitty", "block", or "ascii".
        width: Offscreen render width in pixels.
        height: Offscreen render height in pixels.
        encoding: Kitty encoding -- "raw-zlib", "png", "png-fast", "raw", "jpeg".
        cols: Terminal columns for block/ascii (None = auto-fit).
        camera: MjvCamera instance. Uses a default free camera if not provided.
    """

    def __init__(self, model, *, mode="auto", width=640, height=480,
                 encoding="raw-zlib", cols=None, camera=None):
        self._model = model
        self._width = width
        self._height = height
        self._encoding = encoding
        self._dynamic_cols = cols is None
        self._cols = cols

        # Determine render mode
        if mode == "auto":
            self._mode = "kitty" if supports_kitty() else "block"
        else:
            self._mode = mode

        # Ensure offscreen framebuffer is large enough
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)

        # Create renderer and scene objects
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._opt = mujoco.MjvOption()

        # Set up camera
        if camera is not None:
            self._camera = camera
        else:
            self._camera = mujoco.MjvCamera()
            self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._camera.distance = 3.0
            self._camera.azimuth = 90
            self._camera.elevation = -20
            self._camera.lookat[:] = [0, 0, 0.8]

        # Display state
        self._frame_id = 0
        self._term = None
        self._kitty_display = KITTY_ENCODINGS.get(encoding, display_kitty_raw_zlib)

    @property
    def camera(self):
        """The MjvCamera used for rendering."""
        return self._camera

    @property
    def scene_option(self):
        """The MjvOption used for scene rendering."""
        return self._opt

    def __enter__(self):
        self._term = RawTerminal()
        self._term.__enter__()
        # Clear screen
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        return self

    def __exit__(self, *args):
        self.close()

    def render(self, data):
        """Render a frame and display it to the terminal.

        Non-blocking -- returns immediately after writing to stdout.

        Args:
            data: MjData instance with current simulation state.
        """
        pixels = render_frame(self._model, data, self._renderer,
                              self._camera, scene_option=self._opt)
        self._display(pixels)
        self._frame_id += 1

    def _display(self, pixels):
        """Display pixels using the configured mode."""
        term_size = shutil.get_terminal_size()

        if self._dynamic_cols and self._mode != "kitty":
            self._cols = term_size.columns - 1

        img_aspect = self._height / self._width
        term_rows = term_size.lines - 1

        if self._mode == "kitty":
            max_cols = int(term_rows / img_aspect * 2)
            display_cols = min(term_size.columns, max_cols)
            self._kitty_display(pixels, self._frame_id, display_cols=display_cols)
        elif self._mode == "block":
            max_cols = int(term_rows / (img_aspect * 0.45))
            capped_cols = min(self._cols, max_cols)
            display_halfblock(pixels, capped_cols, self._frame_id)
        else:
            max_cols = int(term_rows / (img_aspect * 0.45))
            capped_cols = min(self._cols, max_cols)
            display_ascii(pixels, capped_cols, self._frame_id)

    def close(self):
        """Restore terminal state and release resources."""
        if self._term is not None:
            self._term.__exit__(None, None, None)
            self._term = None
        self._renderer.close()
