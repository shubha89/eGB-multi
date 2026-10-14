"""Benchmark a paper-style grouped/windowed FGB prototype against fastgb.

This script targets the method in Riegger, Robertsson, and Andersson,
Phys. Rev. D 110, 082001 (2024).  It is not a full production
implementation of their wFGB algorithm.  It benchmarks the central
acceleration idea that is absent from the installed ``fastgb`` package:

* compute the slowly varying response for each source,
* group sources sharing a nearby/base Fourier bin,
* sum their slow streams before the FFT/TDI reduction,
* apply a cheap taper to represent the windowing stage.

The full paper algorithm additionally uses symmetric extension/buffer zones
and a specific time-varying-delay TDI approximation.  Those accuracy features
are not implemented here; this is a wall-time prototype for the grouping and
windowing cost model.

Run with:

    python benchmarks/wfgb_grouped_fastgb_walltime.py

Outputs:

    benchmarks/wfgb_grouped_fastgb_walltime.csv
    benchmarks/wfgb_grouped_fastgb_walltime.png
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastgb.fastgb import CLIGHT, FastGB, computeXYZ, construct_slow_part

from egb_multi.external import default_lisaorbits
from egb_multi.sources import GalacticBinaryParams


def make_sources(n_sources: int, seed: int = 1234) -> list[GalacticBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        GalacticBinaryParams(
            f0=float(rng.uniform(5.0e-5, 3.0e-4)),
            fdot=0.0,
            amplitude=1.0e-22,
            beta=float(rng.uniform(-0.9, 0.9)),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for _ in range(n_sources)
    ]


def fastgb_params(sources: list[GalacticBinaryParams]) -> np.ndarray:
    return np.stack([source.as_fastgb_array() for source in sources], axis=0)


def hanning_taper(size: int) -> np.ndarray:
    if size < 2:
        return np.ones(size)
    return np.hanning(size)


def grouped_windowed_fgb_proxy(
    model: FastGB,
    params: np.ndarray,
    *,
    bin_group_width: int = 1,
    apply_window: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return grouped frequency-domain XYZ streams and the number of groups."""

    if bin_group_width <= 0:
        raise ValueError("bin_group_width must be positive")

    p = params.copy()
    p[:, 7] *= -1.0
    p[:, 3] = np.pi / 2.0 - p[:, 3]

    cosiota = np.cos(p[:, 6])
    fstar = CLIGHT / (model.arm_length * 2.0 * np.pi)
    cosps = np.cos(2.0 * p[:, 5])
    sinps = np.sin(2.0 * p[:, 5])
    aplus = p[:, 2] * (1.0 + cosiota * cosiota)
    across = -2.0 * p[:, 2] * cosiota
    dp = aplus * cosps - 1.0j * across * sinps
    dc = -aplus * sinps - 1.0j * across * cosps

    sinth = np.sin(p[:, 3])
    costh = np.cos(p[:, 3])
    sinph = np.sin(p[:, 4])
    cosph = np.cos(p[:, 4])
    u = np.array([costh * cosph, costh * sinph, -sinth], ndmin=2)
    v = np.array([sinph, -cosph, np.zeros_like(sinph)], ndmin=2)
    k = np.array([-sinth * cosph, -sinth * sinph, -costh], ndmin=2)
    eplus = (v[None, :, :] * v[:, None, :] - u[None, :, :] * u[:, None, :]).T
    ecross = (u[None, :, :] * v[:, None, :] + v[None, :, :] * u[:, None, :]).T

    slow_links, q = construct_slow_part(
        model.T,
        model.arm_length,
        np.array(model.position),
        model.tm,
        p[:, 0],
        p[:, 1],
        fstar,
        p[:, 7],
        k,
        dp,
        dc,
        eplus,
        ecross,
    )

    q = q.astype(np.int64)
    group_keys = (q // bin_group_width) * bin_group_width
    groups = np.unique(group_keys)
    grouped_slow = np.zeros((groups.size, 6, model.N), dtype=np.complex128)
    representative_q = np.zeros(groups.size, dtype=np.int64)
    for index, group in enumerate(groups):
        selected = group_keys == group
        grouped_slow[index] = slow_links[selected].sum(axis=0)
        representative_q[index] = int(np.rint(np.mean(q[selected])))

    if apply_window:
        grouped_slow *= hanning_taper(model.N)[None, None, :]

    base_f0 = representative_q / model.T
    xyz, fmin = computeXYZ(
        model.T,
        grouped_slow,
        base_f0,
        np.zeros(groups.size),
        fstar,
        np.ones(groups.size),
        representative_q,
        model.tm,
    )
    return xyz, fmin, int(groups.size)


def median_elapsed(func, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    counts = [32, 128, 512, 2048, 8192]
    repeats = 5
    duration = 24.0 * 3600.0
    n_samples = 128
    delta_t = duration / 2048
    bin_group_width = 1

    model = FastGB(
        orbits=default_lisaorbits("equal"),
        T=duration,
        delta_t=delta_t,
        N=n_samples,
    )

    warm = fastgb_params(make_sources(128))
    model.get_fd_tdixyz(warm)
    grouped_windowed_fgb_proxy(model, warm, bin_group_width=bin_group_width)

    rows = []
    for count in counts:
        params = fastgb_params(make_sources(count))
        _, _, group_count = grouped_windowed_fgb_proxy(
            model,
            params,
            bin_group_width=bin_group_width,
        )
        fastgb_s = median_elapsed(lambda: model.get_fd_tdixyz(params), repeats)
        grouped_s = median_elapsed(
            lambda: grouped_windowed_fgb_proxy(
                model,
                params,
                bin_group_width=bin_group_width,
            ),
            repeats,
        )
        row = {
            "sources": count,
            "groups": group_count,
            "duration_s": duration,
            "fastgb_fd_xyz_s": fastgb_s,
            "wfgb_grouped_windowed_proxy_s": grouped_s,
            "group_reduction": count / group_count,
            "speedup_fastgb_over_proxy": fastgb_s / grouped_s,
            "bin_group_width": bin_group_width,
            "n_samples": n_samples,
        }
        rows.append(row)
        print(
            f"{count:5d} sources -> {group_count:4d} groups | "
            f"fastgb {fastgb_s:.5f}s | wFGB proxy {grouped_s:.5f}s | "
            f"speedup {fastgb_s / grouped_s:.2f}x"
        )

    csv_path = out_dir / "wfgb_grouped_fastgb_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    axes[0].plot(x, [row["fastgb_fd_xyz_s"] for row in rows], marker="o", label="installed fastgb")
    axes[0].plot(
        x,
        [row["wfgb_grouped_windowed_proxy_s"] for row in rows],
        marker="s",
        label="paper-style grouped/windowed FGB proxy",
    )
    axes[1].plot(x, [row["groups"] for row in rows], marker="D", label="base-frequency groups")
    axes[1].plot(x, [row["speedup_fastgb_over_proxy"] for row in rows], marker="^", label="fastgb / proxy speedup")
    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("count or ratio")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(value)) for value in x])
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("FGB vs grouped/windowed FGB proxy")
    axes[1].set_title("grouping behavior")

    png_path = out_dir / "wfgb_grouped_fastgb_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
