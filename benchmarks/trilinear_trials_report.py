"""Small rerun and two-page report for the trilinear response trials.

The package API uses the historical spelling ``triliear``.  This script keeps
that spelling for imports while writing user-facing report text as
"trilinear".

Run with:

    python benchmarks/trilinear_trials_report.py

Outputs are written under:

    reports/trilinear_trials/
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import JULIAN_YEAR_S
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.response import triliear_source_batch
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_triliear_xyz, compute_triliear_xyz_eccentric, compute_xyz


REPORT_DATE = "2026-08-03"
COUNTS = [1, 4, 16]
REPEATS = 3
SAMPLES = 1024
FASTGB_BINS = 64
ECCENTRICITY = 0.1
ECCENTRIC_ORDERS = [0, 1, 2]
HARMONIC_SAMPLES = 256


def make_circular_sources(n_sources: int, seed: int = 1234) -> list[GalacticBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        GalacticBinaryParams(
            f0=float(rng.uniform(5.0e-4, 3.0e-3)),
            fdot=float(rng.uniform(-1.0e-18, 1.0e-18)),
            amplitude=float(10.0 ** rng.uniform(-23.5, -21.5)),
            beta=float(rng.uniform(-0.9, 0.9)),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for _ in range(n_sources)
    ]


def make_eccentric_sources(n_sources: int, eccentricity: float, seed: int = 1234) -> list[EccentricBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        EccentricBinaryParams(
            mean_motion=float(np.pi * rng.uniform(5.0e-4, 3.0e-3)),
            eccentricity=eccentricity,
            m1_solar=float(rng.uniform(0.35, 0.8)),
            m2_solar=float(rng.uniform(0.25, 0.8)),
            beta=float(rng.uniform(-0.9, 0.9)),
            lambda_=float(rng.uniform(-np.pi, np.pi)),
            psi=float(rng.uniform(0.0, np.pi)),
            inclination=float(rng.uniform(0.0, np.pi)),
            phi0=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for _ in range(n_sources)
    ]


def median_elapsed(func, repeats: int = REPEATS) -> float:
    func()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed))


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_circular_scaling(state, fastgb: FastGBBackend) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for count in COUNTS:
        sources = make_circular_sources(count)
        fastgb_s = median_elapsed(lambda s=sources: fastgb.frequency_domain_xyz(s))
        pair_s = median_elapsed(lambda s=sources: triliear_source_batch(state, s))
        xyz_s = median_elapsed(lambda s=sources: compute_triliear_xyz(state, s))
        exact_s = median_elapsed(
            lambda s=sources: compute_xyz(state, s, generation=1, measurement_order=3, delay_order=3)
        )
        rows.append(
            {
                "sources": count,
                "fastgb_s": fastgb_s,
                "trilinear_pair_s": pair_s,
                "trilinear_xyz_s": xyz_s,
                "exact_link_xyz_s": exact_s,
                "trilinear_xyz_over_fastgb": xyz_s / fastgb_s,
                "exact_link_xyz_over_trilinear_xyz": exact_s / xyz_s,
            }
        )
        print(
            f"circular {count:2d} sources | FastGB {fastgb_s:.6f}s | "
            f"trilinear pair {pair_s:.6f}s | trilinear XYZ {xyz_s:.6f}s | "
            f"exact-link XYZ {exact_s:.6f}s"
        )
    return rows


def run_eccentric_order_scaling(state) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for count in COUNTS:
        sources = make_eccentric_sources(count, ECCENTRICITY)
        row: dict[str, float | int] = {"sources": count, "eccentricity": ECCENTRICITY}
        timings = []
        for order in ECCENTRIC_ORDERS:
            elapsed = median_elapsed(
                lambda s=sources, o=order: compute_triliear_xyz_eccentric(
                    state,
                    s,
                    eccentric_order=o,
                    harmonic_samples=HARMONIC_SAMPLES,
                )
            )
            row[f"eccentric_order_{order}_s"] = elapsed
            timings.append(elapsed)
            print(f"eccentric {count:2d} sources | order {order}: {elapsed:.6f}s")
        row["order_2_over_order_0"] = timings[-1] / timings[0]
        rows.append(row)
    return rows


def save_runtime_plot(rows: list[dict[str, float | int]], path: Path) -> None:
    x = np.array([row["sources"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(x, [row["fastgb_s"] for row in rows], marker="o", label="FastGB circular XYZ")
    ax.plot(x, [row["trilinear_pair_s"] for row in rows], marker="s", label="trilinear pair channels")
    ax.plot(x, [row["trilinear_xyz_s"] for row in rows], marker="D", label="trilinear pyTDI XYZ")
    ax.plot(x, [row["exact_link_xyz_s"] for row in rows], marker="^", label="exact-link pyTDI XYZ")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title("Fresh small-scale circular rerun")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_eccentric_plot(rows: list[dict[str, float | int]], path: Path) -> None:
    x = np.array([row["sources"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for order in ECCENTRIC_ORDERS:
        ax.plot(
            x,
            [row[f"eccentric_order_{order}_s"] for row in rows],
            marker="o",
            label=f"eccentric_order={order}",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title(f"Fresh eccentric trilinear rerun, e={ECCENTRICITY}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def ms(value: float) -> str:
    return f"{1000.0 * value:.2f}"


def load_existing_jax_summary(root: Path) -> str:
    path = root / "benchmarks" / "jax_harmonic_triliear_walltime.csv"
    if not path.exists():
        return "No saved JAX harmonic trilinear CSV was found."
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "The saved JAX harmonic trilinear CSV was empty."
    row = max(rows, key=lambda item: int(item["sources"]))
    return (
        f"Saved larger JAX trial: {int(row['sources'])} sources, "
        f"Python {float(row['python_triliear_xyz_s']):.2f} s, "
        f"JAX {float(row['jax_triliear_xyz_s']):.2f} s, "
        f"speedup {float(row['speedup_python_over_jax']):.2f}x."
    )


def circular_table_text(rows: list[dict[str, float | int]]) -> list[list[str]]:
    return [
        [
            str(row["sources"]),
            ms(float(row["fastgb_s"])),
            ms(float(row["trilinear_pair_s"])),
            ms(float(row["trilinear_xyz_s"])),
            ms(float(row["exact_link_xyz_s"])),
            f"{float(row['trilinear_xyz_over_fastgb']):.1f}x",
        ]
        for row in rows
    ]


def eccentric_table_text(rows: list[dict[str, float | int]]) -> list[list[str]]:
    return [
        [
            str(row["sources"]),
            ms(float(row["eccentric_order_0_s"])),
            ms(float(row["eccentric_order_1_s"])),
            ms(float(row["eccentric_order_2_s"])),
            f"{float(row['order_2_over_order_0']):.1f}x",
        ]
        for row in rows
    ]


def add_wrapped_text(ax, text: str, size: int = 9, y: float = 0.98) -> None:
    ax.axis("off")
    ax.text(0.0, y, text, va="top", ha="left", fontsize=size, wrap=True, linespacing=1.25)


def write_pdf_report(
    path: Path,
    circular_rows: list[dict[str, float | int]],
    eccentric_rows: list[dict[str, float | int]],
    runtime_plot: Path,
    eccentric_plot: Path,
    existing_jax_summary: str,
) -> None:
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(4, 1, height_ratios=[0.9, 1.35, 3.2, 1.0])
        ax_title = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
        ax_plot = fig.add_subplot(gs[2])
        ax_notes = fig.add_subplot(gs[3])

        add_wrapped_text(
            ax_title,
            "Trilinear LISA Response Trials\n"
            f"Short report generated {REPORT_DATE}. Code identifiers keep the repo spelling `triliear`.\n"
            "Scope: tensor response, vectorized source batches, pyTDI XYZ routing, and timing comparisons "
            "against FastGB and exact-link response paths.",
            size=12,
        )

        ax_table.axis("off")
        table = ax_table.table(
            cellText=circular_table_text(circular_rows),
            colLabels=["sources", "FastGB ms", "pair ms", "XYZ ms", "exact XYZ ms", "XYZ/FastGB"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.2)
        table.scale(1.0, 1.35)
        ax_table.set_title(
            f"Fresh rerun: {SAMPLES} samples, ESA-trailing orbit, median of {REPEATS} timed repeats",
            fontsize=9,
        )

        image = plt.imread(runtime_plot)
        ax_plot.imshow(image)
        ax_plot.axis("off")

        add_wrapped_text(
            ax_notes,
            "Result: the direct trilinear pair-channel path is much cheaper than full pyTDI routing, "
            "but FastGB remains the faster circular baseline in this small rerun. The trilinear XYZ path "
            "is still close to the exact-link pyTDI path for these source counts, so most of the observed "
            "cost is in response generation plus pyTDI channel construction rather than just the tensor "
            "contraction.",
            size=9,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(4, 1, height_ratios=[0.75, 3.2, 1.25, 1.25])
        ax_title = fig.add_subplot(gs[0])
        ax_plot = fig.add_subplot(gs[1])
        ax_table = fig.add_subplot(gs[2])
        ax_notes = fig.add_subplot(gs[3])

        add_wrapped_text(
            ax_title,
            "Eccentric Harmonic-Carrier Trials\n"
            f"Fresh rerun uses e={ECCENTRICITY}, harmonic_samples={HARMONIC_SAMPLES}, "
            f"orders {ECCENTRIC_ORDERS}.",
            size=12,
        )

        image = plt.imread(eccentric_plot)
        ax_plot.imshow(image)
        ax_plot.axis("off")

        ax_table.axis("off")
        table = ax_table.table(
            cellText=eccentric_table_text(eccentric_rows),
            colLabels=["sources", "order 0 ms", "order 1 ms", "order 2 ms", "order2/order0"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.2)
        table.scale(1.0, 1.35)

        add_wrapped_text(
            ax_notes,
            "Result: eccentric harmonic order increases runtime roughly monotonically, as expected from "
            "adding harmonic components before pyTDI. Existing saved trials also include larger batches "
            "and a JAX batched harmonic implementation. "
            f"{existing_jax_summary} "
            "Caveat: the FastGB comparison is a timing and scale comparison, not a final residual "
            "validation, because the saved notes indicate differing observable conventions.",
            size=9,
        )
        pdf.savefig(fig)
        plt.close(fig)


def write_markdown_report(
    path: Path,
    circular_rows: list[dict[str, float | int]],
    eccentric_rows: list[dict[str, float | int]],
    runtime_plot: Path,
    eccentric_plot: Path,
    existing_jax_summary: str,
) -> None:
    circular_lines = [
        "| sources | FastGB ms | pair ms | XYZ ms | exact XYZ ms | XYZ/FastGB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in circular_table_text(circular_rows):
        circular_lines.append("| " + " | ".join(row) + " |")

    eccentric_lines = [
        "| sources | order 0 ms | order 1 ms | order 2 ms | order2/order0 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in eccentric_table_text(eccentric_rows):
        eccentric_lines.append("| " + " | ".join(row) + " |")

    path.write_text(
        "\n".join(
            [
                "# Trilinear LISA Response Trials",
                "",
                f"Generated: {REPORT_DATE}. The package API still uses the historical spelling `triliear`.",
                "",
                "## Scope",
                "",
                "We trialled the trilinear response tensor, vectorized multi-source evaluation, pyTDI XYZ routing, "
                "and eccentric harmonic-carrier extensions. The fresh rerun below uses 1024 samples on the "
                "ESA-trailing orbit and reports medians of three timed repeats after one warm-up call.",
                "",
                "## Fresh Circular Rerun",
                "",
                "\n".join(circular_lines),
                "",
                f"![Fresh small-scale circular rerun]({runtime_plot.name})",
                "",
                "FastGB remains the faster circular baseline at this scale. The direct trilinear pair-channel "
                "path is cheaper than full XYZ routing, while trilinear XYZ lands near the exact-link pyTDI "
                "path for these small source counts.",
                "",
                "<div style=\"page-break-after: always;\"></div>",
                "",
                "## Fresh Eccentric Rerun",
                "",
                f"Eccentricity: {ECCENTRICITY}; harmonic samples: {HARMONIC_SAMPLES}.",
                "",
                "\n".join(eccentric_lines),
                "",
                f"![Fresh eccentric trilinear rerun]({eccentric_plot.name})",
                "",
                "Runtime increases with eccentric harmonic order because each source contributes more harmonic "
                "components before pyTDI channel construction.",
                "",
                "## Existing Saved Trial Context",
                "",
                existing_jax_summary,
                "",
                "Relevant saved artifacts include `benchmarks/walltime_fastgb_triliear_exact_xyz.csv`, "
                "`benchmarks/eccentric_xyz_walltime.csv`, `benchmarks/eccentric_order_walltime.csv`, and "
                "`benchmarks/jax_harmonic_triliear_walltime.csv`.",
                "",
                "Caveat: the FastGB comparison is a timing and scale comparison, not a final residual validation, "
                "because the saved README notes differing observable conventions.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "reports" / "trilinear_trials"
    out_dir.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, JULIAN_YEAR_S, SAMPLES, endpoint=False)
    orbits = default_lisaorbits("esa-trailing")
    state = state_from_lisaorbits(orbits, times)
    fastgb = FastGBBackend(delta_t=5.0, duration=JULIAN_YEAR_S, n_samples=FASTGB_BINS, orbits=orbits)

    circular_rows = run_circular_scaling(state, fastgb)
    eccentric_rows = run_eccentric_order_scaling(state)

    circular_csv = out_dir / "small_scale_circular_rerun.csv"
    eccentric_csv = out_dir / "small_scale_eccentric_order_rerun.csv"
    runtime_plot = out_dir / "small_scale_circular_rerun.png"
    eccentric_plot = out_dir / "small_scale_eccentric_order_rerun.png"
    pdf_report = out_dir / "trilinear_trials_report.pdf"
    md_report = out_dir / "trilinear_trials_report.md"

    write_csv(circular_csv, circular_rows)
    write_csv(eccentric_csv, eccentric_rows)
    save_runtime_plot(circular_rows, runtime_plot)
    save_eccentric_plot(eccentric_rows, eccentric_plot)

    existing_jax_summary = load_existing_jax_summary(repo_root)
    write_pdf_report(pdf_report, circular_rows, eccentric_rows, runtime_plot, eccentric_plot, existing_jax_summary)
    write_markdown_report(md_report, circular_rows, eccentric_rows, runtime_plot, eccentric_plot, existing_jax_summary)

    print(f"wrote {circular_csv}")
    print(f"wrote {eccentric_csv}")
    print(f"wrote {runtime_plot}")
    print(f"wrote {eccentric_plot}")
    print(f"wrote {pdf_report}")
    print(f"wrote {md_report}")


if __name__ == "__main__":
    main()
