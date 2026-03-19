#!/usr/bin/env python3
"""Test script to diagnose terminal block rendering alignment issues."""

import shutil

s = shutil.get_terminal_size()
print(f"Terminal: cols={s.columns} rows={s.lines}")
print()

# Test 1: Half-block gradient (should be seamless, no gaps between rows)
print("=== Test 1: Half-block gradient (look for horizontal gaps) ===")
for y in range(20):
    line = ""
    for x in range(40):
        r = int(x / 40 * 255)
        g = int(y / 20 * 255)
        line += f"\033[38;2;{r};{g};0;48;2;0;{255-g};{255-r}m\u2580"
    print(line + "\033[0m")

print()

# Test 2: Full-width half-block (look for wrapping/double lines)
print(f"=== Test 2: Full-width fill ({s.columns} cols, then {s.columns-1} cols) ===")
print(f"--- {s.columns} cols (may band if terminal wraps): ---")
for y in range(5):
    line = ""
    for x in range(s.columns):
        line += f"\033[48;2;{50};{100};{200}m\u2580"
    print(line + "\033[0m")

print(f"--- {s.columns - 1} cols (should be clean): ---")
for y in range(5):
    line = ""
    for x in range(s.columns - 1):
        line += f"\033[48;2;{50};{100};{200}m\u2580"
    print(line + "\033[0m")

print()

# Test 3: Checkerboard (tests alignment between adjacent rows)
print("=== Test 3: Checkerboard (rows should interlock perfectly) ===")
for y in range(10):
    line = ""
    for x in range(40):
        if (x + y) % 2 == 0:
            line += f"\033[38;2;255;255;255;48;2;0;0;0m\u2580"
        else:
            line += f"\033[38;2;0;0;0;48;2;255;255;255m\u2580"
    print(line + "\033[0m")

print()
print("If you see horizontal gaps/lines in tests 1 or 3,")
print("try adding to ~/.config/ghostty/config:")
print("  adjust-cell-height = -1")
print()
print("If test 2 top section bands but bottom doesn't,")
print("the cols-1 fix is needed (already applied).")
