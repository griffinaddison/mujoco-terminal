#!/usr/bin/env python3
"""Benchmark all rendering modes with CPU/GPU monitoring."""

import subprocess
import sys
import os
import json
import time
import threading
import argparse
from datetime import datetime


def sample_gpu(samples, stop_event, interval=0.25):
    """Sample GPU utilization in a background thread."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            parts = out.split(", ")
            if len(parts) == 2:
                samples.append({
                    "gpu_util": int(parts[0]),
                    "gpu_mem_mb": int(parts[1]),
                })
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            break
        stop_event.wait(interval)


def sample_cpu(pid, samples, stop_event, interval=0.25):
    """Sample CPU utilization for a process via /proc."""
    try:
        clock_ticks = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError):
        clock_ticks = 100
    prev_total = 0.0
    prev_time = 0.0
    first = True
    while not stop_event.is_set():
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().split()
            utime = int(fields[13])
            stime = int(fields[14])
            total = (utime + stime) / clock_ticks
            now = time.perf_counter()
            if not first:
                dt = now - prev_time
                if dt > 0:
                    cpu_pct = (total - prev_total) / dt * 100
                    samples.append(cpu_pct)
            first = False
            prev_total = total
            prev_time = now
        except (FileNotFoundError, IndexError, ValueError):
            break
        stop_event.wait(interval)


def run_mode(mode, encoding, duration, fps, extra_args=None):
    """Run the renderer in a given mode and collect stats."""
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "..", "mujoco_terminal_render.py"),
        "--mode", mode,
        "--duration", str(duration),
        "--fps", str(fps),
        "--benchmark",
    ]
    if encoding and mode == "kitty":
        cmd += ["--encoding", encoding]
    if extra_args:
        cmd += extra_args

    gpu_samples = []
    cpu_samples = []
    stop_event = threading.Event()

    proc = subprocess.Popen(
        cmd, stdin=sys.stdin, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    gpu_thread = threading.Thread(target=sample_gpu, args=(gpu_samples, stop_event))
    cpu_thread = threading.Thread(target=sample_cpu, args=(proc.pid, cpu_samples, stop_event))
    gpu_thread.start()
    cpu_thread.start()

    proc.wait()
    stop_event.set()
    stderr = proc.stderr.read() if proc.stderr else ""
    gpu_thread.join()
    cpu_thread.join()

    # Parse JSON stats from stderr
    stats = {}
    for line in stderr.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                stats = json.loads(line)
            except json.JSONDecodeError:
                pass

    # Aggregate CPU/GPU samples
    if cpu_samples:
        import numpy as np
        stats["cpu_pct_mean"] = round(float(np.mean(cpu_samples)), 1)
        stats["cpu_pct_max"] = round(float(np.max(cpu_samples)), 1)
    if gpu_samples:
        import numpy as np
        gpu_utils = [s["gpu_util"] for s in gpu_samples]
        gpu_mems = [s["gpu_mem_mb"] for s in gpu_samples]
        stats["gpu_pct_mean"] = round(float(np.mean(gpu_utils)), 1)
        stats["gpu_pct_max"] = round(float(np.max(gpu_utils)), 1)
        stats["gpu_mem_mb_mean"] = round(float(np.mean(gpu_mems)), 0)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark all rendering modes")
    parser.add_argument("--duration", type=float, default=10, help="Seconds per mode (default: 10)")
    parser.add_argument("--fps", type=float, default=0, help="Target FPS, 0=uncapped (default: 0)")
    parser.add_argument("--modes", nargs="+",
                        default=["kitty:png", "kitty:raw-zlib", "kitty:jpeg", "block", "ascii"],
                        help="Modes to test (e.g. kitty:png block ascii)")
    parser.add_argument("--log", type=str, default="/tmp/mujoco_bench.log",
                        help="Append results to this file (default: /tmp/mujoco_bench.log)")
    args = parser.parse_args()

    print(f"Benchmarking {len(args.modes)} modes, {args.duration}s each, fps={args.fps or 'uncapped'}")
    print()

    results = []
    for mode_spec in args.modes:
        if ":" in mode_spec:
            mode, encoding = mode_spec.split(":", 1)
        else:
            mode, encoding = mode_spec, None

        label = f"{mode}:{encoding}" if encoding else mode
        print(f"Running {label}...", end=" ", flush=True)
        stats = run_mode(mode, encoding, args.duration, args.fps)
        stats["label"] = label
        results.append(stats)
        print(f"{stats.get('fps_mean', '?')} FPS")

    # Print table
    print()
    has_gpu = any("gpu_pct_mean" in r for r in results)
    has_cpu = any("cpu_pct_mean" in r for r in results)

    header = f"{'Mode':<18} {'FPS mean':>9} {'FPS std':>8} {'FPS p1':>7} {'ms/frame':>9}"
    if has_cpu:
        header += f" {'CPU %':>6}"
    if has_gpu:
        header += f" {'GPU %':>6} {'VRAM MB':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r.get('label', '?'):<18} {r.get('fps_mean', 0):>9.1f} {r.get('fps_std', 0):>8.1f} {r.get('fps_p1', 0):>7.1f} {r.get('frame_ms_mean', 0):>9.2f}"
        if has_cpu:
            line += f" {r.get('cpu_pct_mean', 0):>6.1f}"
        if has_gpu:
            line += f" {r.get('gpu_pct_mean', 0):>6.1f} {r.get('gpu_mem_mb_mean', 0):>8.0f}"
        print(line)
    print()

    # Append to log file
    with open(args.log, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Benchmark: {datetime.now().isoformat()} | "
                f"duration={args.duration}s fps={args.fps or 'uncapped'}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in results:
            line = f"{r.get('label', '?'):<18} {r.get('fps_mean', 0):>9.1f} {r.get('fps_std', 0):>8.1f} {r.get('fps_p1', 0):>7.1f} {r.get('frame_ms_mean', 0):>9.2f}"
            if has_cpu:
                line += f" {r.get('cpu_pct_mean', 0):>6.1f}"
            if has_gpu:
                line += f" {r.get('gpu_pct_mean', 0):>6.1f} {r.get('gpu_mem_mb_mean', 0):>8.0f}"
            f.write(line + "\n")
        # Also write raw JSON for each result
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results appended to {args.log}")


if __name__ == "__main__":
    main()
