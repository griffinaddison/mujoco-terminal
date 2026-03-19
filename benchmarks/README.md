# Benchmark Results

## Test Environments

### Desktop — Ubuntu / RTX 5090

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

### Laptop — MacBook Pro M1 Pro

| Component | Details |
|---|---|
| **CPU/GPU** | Apple M1 Pro (10-core CPU, 16-core GPU) |
| **RAM** | 16 GB unified |
| **OS** | macOS (16" MacBook Pro) |
| **Terminal** | Ghostty |
| **Python** | 3.10+ |
| **MuJoCo** | 3.6.0 |

## Results — 2026-03-18

Rendering the built-in pendulum scene at 640x480, uncapped FPS, 10 seconds per mode.

### Desktop (RTX 5090 / Ryzen 9950X3D)

| Mode | FPS mean | FPS std | FPS p1 | ms/frame | CPU % | GPU % | VRAM MB |
|---|---|---|---|---|---|---|---|
| kitty:jpeg* | 1194.7 | 131.8 | 912.9 | 0.85 | 94.7 | 50.0 | 660 |
| ascii | 900.7 | 38.7 | 775.8 | 1.11 | 107.5 | 37.4 | 663 |
| block | 249.4 | 29.9 | 159.3 | 4.09 | 86.8 | 26.8 | 663 |
| kitty:raw-zlib | 197.2 | 58.7 | 86.3 | 5.73 | 71.9 | 67.0 | 660 |
| kitty:png | 167.7 | 17.6 | 97.6 | 6.05 | 90.8 | 64.7 | 662 |

*\*kitty:jpeg — JPEG is not part of the Kitty image protocol spec. The terminal silently drops the image data, so the inflated FPS reflects encode-only cost with no terminal rendering. Included for reference but not a usable mode.*

### Laptop (MacBook Pro 16" M1 Pro)

| Mode | FPS mean | FPS std | FPS p1 | ms/frame |
|---|---|---|---|---|
| kitty:jpeg* | 283.0 | 74.8 | 127.5 | 3.87 |
| ascii | 268.9 | 62.6 | 162.9 | 3.90 |
| kitty:raw-zlib | 131.1 | 36.1 | 61.6 | 8.16 |
| block | 118.6 | 7.4 | 85.7 | 8.47 |
| kitty:png | 116.1 | 17.4 | 58.4 | 8.91 |

*No CPU/GPU utilization data on macOS (no `/proc` or `nvidia-smi`).*

### Cross-platform comparison

| Mode | Desktop FPS | Mac FPS | Ratio |
|---|---|---|---|
| ascii | 900.7 | 268.9 | 3.3x |
| block | 249.4 | 118.6 | 2.1x |
| kitty:raw-zlib | 197.2 | 131.1 | 1.5x |
| kitty:png | 167.7 | 116.1 | 1.4x |

### Key observations

- **ascii** mode is the fastest usable mode on both platforms, entirely CPU-bound.
- **block** (half-block Unicode) is bottlenecked by generating thousands of ANSI escape sequences per frame.
- **kitty:png** and **kitty:raw-zlib** are similar (~170-200 FPS desktop, ~115-130 FPS laptop). Image encode is the bottleneck, but the terminal offloads rendering to the GPU.
- **CPU >100%** on desktop indicates multi-core usage (MuJoCo rendering + Python encoding on separate cores).
- The desktop-to-laptop gap is smallest for kitty modes (1.4-1.5x) since the terminal GPU handles the heavy lifting, and largest for ascii (3.3x) which is pure CPU string ops.
- All modes comfortably exceed 60 FPS on both machines at 640x480.

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
