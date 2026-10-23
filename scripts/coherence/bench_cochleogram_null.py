#!/usr/bin/env python3
"""Re-validate the numbers ``get_null_cochleogram_coherence.py`` bases its design on.

**Run this on an idle box.** The figures currently quoted in that script's docstring were taken on
2026-08-13 while another job was running, so they are if anything understated -- the *ratios*
between them are far more trustworthy than the absolute GFLOP/s. Nothing here writes to the
dataset; the matmul benchmark allocates a few GB and the rest only reads metadata.

    python scripts/coherence/bench_cochleogram_null.py             # matmul + disk, ~2 min
    python scripts/coherence/bench_cochleogram_null.py --scatter   # + the per-channel fold cost
    python scripts/coherence/bench_cochleogram_null.py --full-f    # at the real F=2561, ~20 min

Four questions, in the order they can invalidate the design:

1. **Does splitting the channel axis still cost ~2x?** The whole reason the 61 channels stay in a
   single matmul is that ``Cx`` *is* the arithmetic intensity -- exactly ``Cx`` FLOP per byte of
   ``Yf`` at complex64. If the measured falloff from Cx=61 to Cx=15 is small on an idle box,
   per-channel parallelism becomes viable and the design should be revisited.
2. **Does numpy's matmul still release the GIL at 24 threads?** The single-process threaded design
   rests on it. Anything much below ~6x scaling and a process pool comes back into contention --
   at which point per-worker duplication of Yf_t (1.37 GB) and Xf_t (1.31 GB) becomes the binding
   limit.
3. **Is the per-channel scatter cheap?** (``--scatter``) Accumulating straight into 61 separate
   channel arrays is what keeps peak disk at ~40 GB rather than 80 GB, and it is paid for with a
   strided read per channel per draw. Expected to be small against a ~128 ms matmul; unmeasured.
4. **Is there room on disk?** The accumulators *are* the output arrays, so the requirement is the
   39.99 GB working set plus one 655.6 MB channel -- comfortable on /data, unlike the earlier
   combined-then-split design that needed ~160 GB.

Then validate end to end before committing ~10 h:

    python scripts/coherence/get_null_cochleogram_coherence.py --max-trials 8 --n-null 10

and cross-check its ``gammatone_null.npz`` against script 3's broadband output using the identity
in plan section 3.1 -- they should agree to complex64 roundoff, since draw *i* is the same shift.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SRC, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Realistic shapes: F=2561 one-sided bins, ~1048 Welch frames for a 10-minute trial, 64 EEG
# channels, 61 stimulus features (60 subbands + broadband).
F_FULL, N_FRAMES, N_EEG, N_FEATURES = 2561, 1048, 64, 61
N_TRIALS, N_DRAWS = 563, 500


def bench_matmul(F: int, Cx: int, threads: int, dtype: type, reps: int = 3) -> tuple[float, float]:
    """Best-of-``reps`` wall time and GFLOP/s for one draw's batched matmul, split across threads."""
    X = np.empty((F, Cx, N_FRAMES), dtype)
    Y = np.empty((F, N_FRAMES, N_EEG), dtype)
    X.real, Y.real = 1.0, 1.0
    bounds = np.linspace(0, F, threads + 1).astype(int)

    def work(i: int) -> None:
        np.matmul(X[bounds[i] : bounds[i + 1]], Y[bounds[i] : bounds[i + 1]])

    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        if threads == 1:
            work(0)
        else:
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(work, range(threads)))
        best = min(best, time.perf_counter() - start)
    return best, 8 * F * Cx * N_FRAMES * N_EEG / 1e9 / best


def report_matmul(F: int) -> dict[tuple[str, int, int], float]:
    print(f"\n=== batched matmul, F={F}, n_frames={N_FRAMES}, Cy={N_EEG} ===")
    print(f"{'dtype':>11} {'Cx':>4} {'threads':>8} {'GFLOP/s':>9} {'scaling':>8} {'FLOP/byte':>10}")
    results: dict[tuple[str, int, int], float] = {}
    for dtype, name in ((np.complex64, "complex64"), (np.complex128, "complex128")):
        itemsize = np.dtype(dtype).itemsize
        for Cx in (N_FEATURES, 30, 15, 1):
            base = None
            for threads in (1, 12, 24):
                elapsed, gflops = bench_matmul(F, Cx, threads, dtype)
                base = base or elapsed
                print(
                    f"{name:>11} {Cx:>4} {threads:>8} {gflops:>9.1f} {base / elapsed:>7.2f}x "
                    f"{8 * Cx / itemsize:>10.1f}"
                )
                results[(name, Cx, threads)] = gflops
    return results


def report_projection(results: dict, dtype_name: str = "complex64") -> None:
    """Turn the best measured throughput into a wall-clock projection for the real run."""
    per_draw_gflop = 8 * F_FULL * N_FEATURES * N_FRAMES * N_EEG / 1e9
    total_pflop = N_TRIALS * N_DRAWS * per_draw_gflop / 1e6
    print(f"\n=== projection ({dtype_name}, {N_TRIALS} trials x {N_DRAWS} draws) ===")
    print(f"per draw {per_draw_gflop:.1f} GFLOP, total {total_pflop:.1f} PFLOP")
    for Cx_label, Cx in (("all 61 in one matmul", N_FEATURES), ("split into groups of 15", 15)):
        gflops = results.get((dtype_name, Cx, 24))
        if gflops:
            print(f"  {Cx_label:<24} {gflops:>6.1f} GF/s -> {total_pflop * 1e6 / gflops / 3600:>5.1f} h of matmul")


def bench_scatter(F: int, reps: int = 5) -> None:
    """Cost of scattering one draw across 61 per-channel accumulators vs one combined array.

    The per-channel layout is what keeps peak disk at 40 GB instead of 80 GB, and it is paid for
    with a strided read of ``cross[:, k, :]`` per channel per draw. That read takes 64 contiguous
    complex64 -- 512-byte runs -- so cache lines are fully used and the penalty should be
    prefetcher efficiency rather than wasted bandwidth. This measures whether that holds.
    """
    cross = np.ones((F, N_FEATURES, N_EEG), np.complex64)
    combined = np.zeros((F, N_FEATURES, N_EEG), np.complex64)
    per_channel = [np.zeros((F, 1, N_EEG), np.complex64) for _ in range(N_FEATURES)]

    def time_it(fn) -> float:
        best = float("inf")
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
        return best

    def contiguous() -> None:
        combined += cross

    def scattered() -> None:
        for k, acc in enumerate(per_channel):
            acc[:, 0, :] += cross[:, k, :]

    one, many = time_it(contiguous), time_it(scattered)
    moved = cross.nbytes / 1e6
    print(f"\n=== scatter cost, F={F} ({moved:.0f} MB per draw) ===")
    print(f"  combined (F, 61, 64) +=      {one * 1e3:>7.1f} ms")
    print(f"  61 x per-channel  +=         {many * 1e3:>7.1f} ms   ({many / one:.2f}x)")
    print(f"  against a ~128 ms matmul: {'negligible' if many - one < 0.013 else 'worth a second look'}")


def report_disk() -> None:
    """Peak disk for the per-channel design: the accumulators *are* the output arrays.

    Finalize swaps one .npy for one .npz of the same size at a time, so the requirement is the
    working set plus a single channel -- not double it.
    """
    total = N_FEATURES * N_DRAWS * F_FULL * N_EEG * 8
    one_channel = total // N_FEATURES
    peak = total + one_channel
    print("\n=== disk (complex64, 61 per-channel accumulators) ===")
    print(f"working set {total / 1e9:.1f} GB + one channel {one_channel / 1e6:.0f} MB = peak {peak / 1e9:.1f} GB")
    for path in ("/data", "/data_storage"):
        if Path(path).exists():
            free = shutil.disk_usage(path).free
            print(f"  {path:<15} {free / 1e9:>8.1f} GB free   {'OK' if free >= peak else 'TOO SMALL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full-f", action="store_true", help=f"benchmark at the real F={F_FULL} (slow)")
    parser.add_argument("--scatter", action="store_true", help="also measure the per-channel scatter cost")
    parser.add_argument("--skip-matmul", action="store_true", help="disk check only")
    args = parser.parse_args(argv)

    F = F_FULL if args.full_f else 256
    if not args.skip_matmul:
        results = report_matmul(F)
        if not args.full_f:
            print(f"\n(F={F} is 1/10th of the real axis; the batch loop is over F, so GFLOP/s scales across.)")
        report_projection(results)

    if args.scatter:
        bench_scatter(F_FULL)  # always at full F: the whole question is cache behaviour

    report_disk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
