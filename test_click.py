#!/usr/bin/env python3
"""Test script to verify mouse click position mapping in terminal."""

import sys
import os
import select
import termios
import tty
import time


def main():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        # Enable mouse reporting
        sys.stdout.write("\033[?1003h")  # all motion tracking
        sys.stdout.write("\033[?1006h")  # SGR extended format
        sys.stdout.write("\033[?25l")    # hide cursor
        sys.stdout.write("\033[2J\033[H")  # clear screen
        sys.stdout.flush()

        # Draw a grid
        term_cols = 80
        term_rows = 24
        try:
            import shutil
            s = shutil.get_terminal_size()
            term_cols = s.columns
            term_rows = s.lines
        except Exception:
            pass

        # Draw border and crosshairs
        def draw():
            sys.stdout.write("\033[H")
            for r in range(1, term_rows):
                for c in range(1, term_cols + 1):
                    if r == 1 or r == term_rows - 2:
                        ch = "-"
                    elif c == 1 or c == term_cols:
                        ch = "|"
                    elif r == term_rows // 2 and c == term_cols // 2:
                        ch = "+"
                    elif r == term_rows // 2:
                        ch = "-"
                    elif c == term_cols // 2:
                        ch = "|"
                    else:
                        ch = " "
                    sys.stdout.write(ch)
                sys.stdout.write("\r\n")
            sys.stdout.flush()

        draw()

        # Status at bottom
        status = f"Terminal: {term_cols}x{term_rows} | Click anywhere | Q=quit"
        sys.stdout.write(f"\033[{term_rows};1H\033[K{status}\r")
        sys.stdout.flush()

        last_click_time = 0
        last_click_col = 0
        last_click_row = 0

        while True:
            if select.select([fd], [], [], 0.05)[0]:
                data = b""
                while select.select([fd], [], [], 0)[0]:
                    data += os.read(fd, 1024)

                text = data.decode("utf-8", errors="ignore")
                i = 0
                while i < len(text):
                    if text[i:i+3] == "\033[<":
                        i += 3
                        end = i
                        while end < len(text) and text[end] not in ("M", "m"):
                            end += 1
                        if end < len(text):
                            parts = text[i:end].split(";")
                            if len(parts) == 3:
                                button = int(parts[0])
                                col = int(parts[1])
                                row = int(parts[2])
                                pressed = text[end] == "M"

                                btn_id = button & 0x03
                                is_motion = (button & 32) != 0
                                is_ctrl = (button & 16) != 0
                                is_scroll = (button & 64) != 0

                                if pressed and not is_motion and not is_scroll:
                                    # Normalize to [0, 1]
                                    relx = col / term_cols
                                    rely = 1.0 - (row / term_rows)

                                    btn_name = ["LEFT", "MID", "RIGHT", "?"][btn_id]
                                    ctrl = "Ctrl+" if is_ctrl else ""

                                    # Check double-click
                                    now = time.perf_counter()
                                    dt = now - last_click_time
                                    dist = abs(col - last_click_col) + abs(row - last_click_row)
                                    dbl = " DOUBLE-CLICK!" if dt < 0.4 and dist < 3 else ""
                                    last_click_time = now
                                    last_click_col = col
                                    last_click_row = row

                                    # Draw marker at click position
                                    sys.stdout.write(f"\033[{row};{col}H\033[91mX\033[0m")

                                    # Status
                                    info = (f"  {ctrl}{btn_name} cell=({col},{row}) "
                                            f"rel=({relx:.3f},{rely:.3f}){dbl}  ")
                                    sys.stdout.write(f"\033[{term_rows};1H\033[K{info}\r")
                                    sys.stdout.flush()
                            i = end + 1
                        else:
                            i += 1
                    elif text[i] == "q" or text[i] == "\x03":
                        return
                    else:
                        i += 1

    finally:
        sys.stdout.write("\033[?1003l")
        sys.stdout.write("\033[?1006l")
        sys.stdout.write("\033[?25h")
        sys.stdout.write("\033[0m\033[2J\033[H")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("Done")


if __name__ == "__main__":
    main()
