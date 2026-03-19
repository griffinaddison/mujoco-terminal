# Benchmark Results

## Test Environment

| Component | Details |
|---|---|
| **CPU** | AMD Ryzen 9 9950X3D 16-Core (32 threads) |
| **RAM** | 64 GB DDR5 |
| **GPU** | NVIDIA GeForce RTX 5090 |
| **GPU Driver** | 570.195.03 |
| **OS** | Ubuntu 22.04.5 LTS (kernel 6.8.0-60-generic) |
| **Terminal** | Ghostty 1.3.0-main+17da138 (GTK 4.20.3, OpenGL renderer) |
| **Python** | 3.10.12 |
| **MuJoCo** | 3.6.0 |

## Results — 2026-03-18

Rendering the built-in pendulum scene at 640x480, uncapped FPS, 10 seconds per mode.

| Mode | FPS mean | FPS std | FPS p1 | ms/frame | CPU % | GPU % | VRAM MB |
|---|---|---|---|---|---|---|---|
| kitty:jpeg* | 1194.7 | 131.8 | 912.9 | 0.85 | 94.7 | 50.0 | 660 |
| ascii | 900.7 | 38.7 | 775.8 | 1.11 | 107.5 | 37.4 | 663 |
| block | 249.4 | 29.9 | 159.3 | 4.09 | 86.8 | 26.8 | 663 |
| kitty:raw-zlib | 197.2 | 58.7 | 86.3 | 5.73 | 71.9 | 67.0 | 660 |
| kitty:png | 167.7 | 17.6 | 97.6 | 6.05 | 90.8 | 64.7 | 662 |

*\*kitty:jpeg — JPEG is not part of the Kitty image protocol spec. The terminal silently drops the image data, so the inflated FPS reflects encode-only cost with no terminal rendering. Included for reference but not a usable mode.*

### Key observations

- **ascii** mode is the fastest usable mode (~900 FPS), CPU-bound with minimal GPU usage.
- **block** (half-block Unicode) mode runs at ~250 FPS. The bottleneck is generating thousands of ANSI escape sequences per frame.
- **kitty:png** and **kitty:raw-zlib** are similar (~170-200 FPS). PNG encode and zlib compression are the bottleneck, but they push more work to the GPU (64-67%) since the terminal decodes and renders the image on the GPU.
- **CPU >100%** indicates multi-core usage (MuJoCo rendering + Python encoding on separate cores).
- All modes are well above 60 FPS at 640x480, so the default `--fps 60` cap is achievable on this hardware.

## Running benchmarks

```bash
# Full benchmark (all modes, 10s each, uncapped FPS)
python3 benchmarks/benchmark.py --duration 10 --modes kitty:png kitty:raw-zlib block ascii

# Specific modes
python3 benchmarks/benchmark.py --duration 10 --modes block ascii

# At a target FPS to measure headroom
python3 benchmarks/benchmark.py --duration 10 --fps 60 --modes block ascii

# Results are appended to /tmp/mujoco_bench.log by default
python3 benchmarks/benchmark.py --log benchmarks/results.log
```

The encode-only microbenchmark (no terminal output) can be run with:

```bash
python3 benchmarks/bench.py --tag mytag --n 300
```
