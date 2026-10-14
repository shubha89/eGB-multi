#!/usr/bin/env python3
"""Summarize a Sangria-subset eccentric-vs-circular comparison run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _unique_source_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        seen[(row["catalog"], row["catalog_index"])] = row
    return list(seen.values())


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def _write_summary(rows: list[dict[str, str]], unique_rows: list[dict[str, str]], out: Path) -> None:
    ecc = sorted(set(_as_float(rows, "eccentricity")))
    fbins = sorted(set(_as_float(rows, "frequency_bin_center_hz")))

    with out.open("w") as f:
        f.write("# Sangria subset diagnostic summary\n")
        f.write(f"rows = {len(rows)}\n")
        f.write(f"unique_sources = {len(unique_rows)}\n")
        f.write(f"eccentricity_grid = {[float(x) for x in ecc]}\n")
        f.write(f"frequency_bin_centers_hz = {[float(x) for x in fbins]}\n\n")

        for key in ("f0_hz", "catalog_fdot_hz_s", "catalog_amplitude"):
            vals = _as_float(unique_rows, key)
            f.write(
                f"{key}: min={vals.min():.6e}, p16={_percentile(vals, 16):.6e}, "
                f"median={np.median(vals):.6e}, p84={_percentile(vals, 84):.6e}, "
                f"max={vals.max():.6e}\n"
            )

        f.write("\n# Response-level quantities by eccentricity\n")
        for e in ecc:
            sub = [row for row in rows if float(row["eccentricity"]) == e]
            mismatch = _as_float(sub, "mismatch_AE")
            relres = _as_float(sub, "relative_noise_weighted_residual_AE")
            snr = _as_float(sub, "signal_snr_AE")
            f.write(
                f"e={e:.6g}: mismatch median={np.median(mismatch):.6e}, "
                f"p90={_percentile(mismatch, 90):.6e}, max={mismatch.max():.6e}; "
                f"relative_residual median={np.median(relres):.6e}, "
                f"p90={_percentile(relres, 90):.6e}; "
                f"signal_snr_proxy median={np.median(snr):.6e}, "
                f"p90={_percentile(snr, 90):.6e}\n"
            )


def _plot_catalog_diagnostics(unique_rows: list[dict[str, str]], rows: list[dict[str, str]], out: Path) -> None:
    f0 = _as_float(unique_rows, "f0_hz")
    fdot = _as_float(unique_rows, "catalog_fdot_hz_s")
    amp = _as_float(unique_rows, "catalog_amplitude")
    fbin = _as_float(unique_rows, "frequency_bin_center_hz")

    ecc_grid = sorted(set(_as_float(rows, "eccentricity")))
    snr_by_e = []
    for e in ecc_grid:
        sub = [row for row in rows if float(row["eccentricity"]) == e]
        snr_by_e.append(_as_float(sub, "signal_snr_AE"))

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.hist(f0, bins=np.geomspace(f0.min() * 0.9, f0.max() * 1.1, 16), color="#0072B2", alpha=0.82)
    for center in sorted(set(fbin)):
        ax.axvline(center, color="0.35", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$f_0$ [Hz]")
    ax.set_ylabel("source count")

    ax = axes[0, 1]
    colors = np.log10(amp)
    sc = ax.scatter(f0, fdot, c=colors, s=34, cmap="viridis", edgecolor="k", linewidth=0.25)
    ax.axhline(0.0, color="0.45", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-22)
    ax.set_xlabel(r"$f_0$ [Hz]")
    ax.set_ylabel(r"catalog $\dot f$ [Hz s$^{-1}$]")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}$ amplitude")

    ax = axes[1, 0]
    ax.hist(amp, bins=np.geomspace(amp.min() * 0.9, amp.max() * 1.1, 16), color="#D55E00", alpha=0.78)
    ax.set_xscale("log")
    ax.set_xlabel("catalog amplitude")
    ax.set_ylabel("source count")

    ax = axes[1, 1]
    med = np.asarray([np.median(v) for v in snr_by_e])
    p16 = np.asarray([np.percentile(v, 16) for v in snr_by_e])
    p84 = np.asarray([np.percentile(v, 84) for v in snr_by_e])
    for e, vals in zip(ecc_grid, snr_by_e):
        jitter = np.linspace(-0.035, 0.035, len(vals))
        ax.scatter(e * (1.0 + jitter), vals, s=11, color="0.2", alpha=0.42, zorder=2)
    ax.fill_between(ecc_grid, p16, p84, color="#009E73", alpha=0.22, lw=0)
    ax.plot(ecc_grid, med, color="#009E73", marker="o", lw=1.8, label="median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("signal SNR proxy")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_result_summary(rows: list[dict[str, str]], out: Path) -> None:
    ecc_grid = sorted(set(_as_float(rows, "eccentricity")))
    fbins = sorted(set(_as_float(rows, "frequency_bin_center_hz")))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(fbins)))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True, sharex=True)

    for color, fbin in zip(colors, fbins):
        med_mismatch = []
        med_relres = []
        for e in ecc_grid:
            sub = [
                row
                for row in rows
                if float(row["frequency_bin_center_hz"]) == fbin and float(row["eccentricity"]) == e
            ]
            med_mismatch.append(np.median(_as_float(sub, "mismatch_AE")))
            med_relres.append(np.median(_as_float(sub, "relative_noise_weighted_residual_AE")))
        label = f"{fbin:.0e} Hz"
        axes[0].plot(ecc_grid, med_mismatch, marker="o", lw=1.8, color=color, label=label)
        axes[1].plot(ecc_grid, med_relres, marker="o", lw=1.8, color=color, label=label)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("eccentricity")
        ax.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("median mismatch")
    axes[1].set_ylabel("median relative noise-weighted residual")
    axes[1].legend(title=r"$f_0$ bin", fontsize=8, title_fontsize=8, frameon=False)

    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-prefix", type=Path)
    args = parser.parse_args()

    rows = _read_rows(args.csv)
    unique_rows = _unique_source_rows(rows)
    out_prefix = args.out_prefix or args.csv.with_suffix("")

    catalog_plot = out_prefix.with_name(out_prefix.name + "_catalog_diagnostics.png")
    summary_plot = out_prefix.with_name(out_prefix.name + "_summary_by_eccentricity.png")
    summary_txt = out_prefix.with_name(out_prefix.name + "_summary.txt")

    _plot_catalog_diagnostics(unique_rows, rows, catalog_plot)
    _plot_result_summary(rows, summary_plot)
    _write_summary(rows, unique_rows, summary_txt)

    print(f"wrote {catalog_plot}")
    print(f"wrote {summary_plot}")
    print(f"wrote {summary_txt}")


if __name__ == "__main__":
    main()
