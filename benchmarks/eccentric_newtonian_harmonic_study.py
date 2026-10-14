"""Study the grouped harmonic basis for Newtonian fixed-eccentric sources.

Run with:

    python benchmarks/eccentric_newtonian_harmonic_study.py

Outputs are written to:

    reports/eccentric_newtonian_harmonic_study/
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eccentric_grouped_trilinear_ansatz_study import (
    combined_xyz_residual,
    harmonic_order_from_ansatz,
    make_sources,
    source_harmonic_residual,
    xyz_from_links,
)

from egb_multi.eccentric import EccentricBinaryParams, eccentric_harmonic_indices
from egb_multi.eccentric_jax import eccentric_harmonic_links_grouped_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


PHYSICS_MODE = "newtonian"


def median_elapsed_value(func, repeats: int):
    timings = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        timings.append(time.perf_counter() - start)
    return value, float(np.median(timings))


def source_cutoff_rows(source: EccentricBinaryParams, eccentricity: float, *, samples: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    nmax_values = [2, 3, 4, 6, 9, 12, 16, 20, 30, 40, 60]
    for nmax in nmax_values:
        order = 0 if nmax == 2 else nmax - 2
        harmonics = eccentric_harmonic_indices(order)
        if max(harmonics) != nmax:
            continue
        rows.append(
            {
                "eccentricity": eccentricity,
                "n_max_harmonic": nmax,
                "eccentric_order": order,
                "positive_harmonic_count": len(harmonics),
                "source_harmonic_scaled_residual": source_harmonic_residual(
                    source,
                    harmonics,
                    samples=samples,
                    physics_mode=PHYSICS_MODE,
                ),
            }
        )
    return rows


def plot(rows: list[dict[str, object]], cutoff_rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    counts = sorted({int(row["sources"]) for row in rows})
    eccentricities = sorted({float(row["eccentricity"]) for row in rows})

    for count in counts:
        subset = sorted([row for row in rows if int(row["sources"]) == count], key=lambda row: float(row["eccentricity"]))
        x = [float(row["eccentricity"]) for row in subset]
        axes[0, 0].plot(x, [float(row["exact_transfer_scaled_real_residual"]) for row in subset], marker="o", label=f"N={count}")
        axes[0, 1].plot(x, [float(row["exact_transfer_speedup_vs_brute"]) for row in subset], marker="s", label=f"N={count}")

    largest_count = max(counts)
    largest = sorted([row for row in rows if int(row["sources"]) == largest_count], key=lambda row: float(row["eccentricity"]))
    x = [float(row["eccentricity"]) for row in largest]
    axes[1, 0].plot(x, [int(row["n_max_harmonic"]) for row in largest], marker="o", color="tab:blue", label="n_max ansatz")
    axis_residual = axes[1, 0].twinx()
    axis_residual.plot(
        x,
        [float(row["source_harmonic_scaled_residual"]) for row in largest],
        marker="s",
        color="tab:orange",
        label="source residual",
    )

    for eccentricity in sorted({float(row["eccentricity"]) for row in cutoff_rows}):
        subset = sorted(
            [row for row in cutoff_rows if float(row["eccentricity"]) == eccentricity],
            key=lambda row: int(row["n_max_harmonic"]),
        )
        axes[1, 1].plot(
            [int(row["n_max_harmonic"]) for row in subset],
            [float(row["source_harmonic_scaled_residual"]) for row in subset],
            marker="^",
            label=f"e={eccentricity:g}",
        )

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("full Re(XYZ) residual")
    axes[0, 0].set_xlabel("eccentricity")
    axes[0, 0].set_ylabel("scaled residual")

    axes[0, 1].axhline(1.0, color="0.4", linewidth=1.0, linestyle="--")
    axes[0, 1].set_title("exact-transfer speedup")
    axes[0, 1].set_xlabel("eccentricity")
    axes[0, 1].set_ylabel("brute wall / grouped wall")

    axes[1, 0].set_title(f"ansatz at N={largest_count}")
    axes[1, 0].set_xlabel("eccentricity")
    axes[1, 0].set_ylabel("largest positive harmonic")
    axes[1, 0].set_yscale("log")
    axis_residual.set_ylabel("source residual")
    axis_residual.set_yscale("log")

    axes[1, 1].set_title("source-only cutoff diagnostic")
    axes[1, 1].set_xlabel("largest positive harmonic")
    axes[1, 1].set_ylabel("source residual")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xscale("log")

    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axis_residual.legend(fontsize=8, loc="lower left")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, object]], cutoff_rows: list[dict[str, object]], report_dir: Path, plot_path: Path) -> None:
    largest_count = max(int(row["sources"]) for row in rows)
    largest = sorted([row for row in rows if int(row["sources"]) == largest_count], key=lambda row: float(row["eccentricity"]))
    lines = [
        "# Newtonian Fixed-Eccentricity Harmonic Study",
        "",
        "This study repeats the grouped harmonic comparison with `physics_mode=\"newtonian\"`, so the fixed-eccentric source has no periastron precession and `n_r = Omega_phi`.",
        "",
        "The reference is the JAX brute-force eccentric link response. The model is the grouped harmonic exact-transfer response using the same harmonic ansatz as the 1PN study.",
        "",
        "| e | n_max | source residual | grouped residual | brute [s] | grouped [s] | speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in largest:
        lines.append(
            f"| {float(row['eccentricity']):.2f} | "
            f"{int(row['n_max_harmonic'])} | "
            f"{float(row['source_harmonic_scaled_residual']):.3e} | "
            f"{float(row['exact_transfer_scaled_real_residual']):.3e} | "
            f"{float(row['brute_force_wall_s']):.4f} | "
            f"{float(row['trilinear_exact_transfer_wall_s']):.4f} | "
            f"{float(row['exact_transfer_speedup_vs_brute']):.2f}x |"
        )
    lines.extend(
        [
            "",
            "Interpretation: without periastron precession, the old harmonic basis is the natural source basis. The remaining high-e cost comes from needing many harmonics to represent the sharp periastron burst, not from missing precession sidebands.",
            "",
            "Source-only cutoff diagnostic:",
            "",
            "| e | n_max | source residual |",
            "|---:|---:|---:|",
        ]
    )
    for row in cutoff_rows:
        lines.append(
            f"| {float(row['eccentricity']):.2f} | "
            f"{int(row['n_max_harmonic'])} | "
            f"{float(row['source_harmonic_scaled_residual']):.3e} |"
        )
    lines.extend(["", f"![Newtonian harmonic study]({plot_path.name})", ""])
    (report_dir / "eccentric_newtonian_harmonic_study.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports" / "eccentric_newtonian_harmonic_study"
    report_dir.mkdir(parents=True, exist_ok=True)

    duration = 86400.0
    samples = 1024
    harmonic_samples = 2048
    counts = [32, 128, 512]
    eccentricities = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    repeats = 3
    batch_size = 128

    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)

    warm_sources = make_sources(batch_size, max(eccentricities))
    eccentric_links_jax(warm_sources, geometry, batch_size=batch_size, physics_mode=PHYSICS_MODE)
    for eccentricity in eccentricities:
        order = harmonic_order_from_ansatz(eccentricity)
        warm = [EccentricBinaryParams(**{**source.__dict__, "eccentricity": eccentricity}) for source in warm_sources]
        eccentric_harmonic_links_grouped_jax(
            warm,
            geometry,
            eccentric_order=order,
            harmonic_samples=harmonic_samples,
            batch_size=batch_size,
            response_mode="fastgb_transfer",
            physics_mode=PHYSICS_MODE,
        )

    rows: list[dict[str, object]] = []
    for eccentricity in eccentricities:
        order = harmonic_order_from_ansatz(eccentricity)
        harmonics = eccentric_harmonic_indices(order)
        n_max = max(harmonics)
        for count in counts:
            sources = make_sources(count, eccentricity)
            source_residual = source_harmonic_residual(
                sources[0],
                harmonics,
                samples=harmonic_samples,
                physics_mode=PHYSICS_MODE,
            )
            brute_links, brute_links_s = median_elapsed_value(
                lambda s=sources: eccentric_links_jax(s, geometry, batch_size=batch_size, physics_mode=PHYSICS_MODE),
                repeats,
            )
            brute_xyz, brute_tdi_s = median_elapsed_value(lambda: xyz_from_links(state, brute_links), repeats)
            exact_links, exact_links_s = median_elapsed_value(
                lambda s=sources, o=order: eccentric_harmonic_links_grouped_jax(
                    s,
                    geometry,
                    eccentric_order=o,
                    harmonic_samples=harmonic_samples,
                    batch_size=batch_size,
                    response_mode="fastgb_transfer",
                    physics_mode=PHYSICS_MODE,
                ),
                repeats,
            )
            exact_xyz, exact_tdi_s = median_elapsed_value(lambda: xyz_from_links(state, exact_links), repeats)
            brute_wall_s = brute_links_s + brute_tdi_s
            exact_wall_s = exact_links_s + exact_tdi_s
            row = {
                "sources": count,
                "eccentricity": eccentricity,
                "physics_mode": PHYSICS_MODE,
                "duration_days": duration / 86400.0,
                "samples": samples,
                "harmonic_samples": harmonic_samples,
                "n_max_harmonic": n_max,
                "positive_harmonic_count": len(harmonics),
                "positive_harmonics": " ".join(str(harmonic) for harmonic in harmonics),
                "eccentric_order": order,
                "source_harmonic_scaled_residual": source_residual,
                "brute_force_links_s": brute_links_s,
                "brute_force_tdi_s": brute_tdi_s,
                "brute_force_wall_s": brute_wall_s,
                "trilinear_exact_transfer_links_s": exact_links_s,
                "trilinear_exact_transfer_tdi_s": exact_tdi_s,
                "trilinear_exact_transfer_wall_s": exact_wall_s,
                "exact_transfer_raw_real_residual": combined_xyz_residual(exact_xyz, brute_xyz, scaled=False),
                "exact_transfer_scaled_real_residual": combined_xyz_residual(exact_xyz, brute_xyz, scaled=True),
                "exact_transfer_speedup_vs_brute": brute_wall_s / exact_wall_s,
            }
            rows.append(row)
            print(
                f"e={eccentricity:.2f} N={count:4d} nmax={n_max:2d} "
                f"src={source_residual:.3e} xyz={row['exact_transfer_scaled_real_residual']:.3e} "
                f"brute={brute_wall_s:.4f}s grouped={exact_wall_s:.4f}s "
                f"speedup={row['exact_transfer_speedup_vs_brute']:.2f}x",
                flush=True,
            )

    cutoff_rows: list[dict[str, object]] = []
    for eccentricity in [0.3, 0.5, 0.7]:
        source = make_sources(1, eccentricity)[0]
        cutoff_rows.extend(source_cutoff_rows(source, eccentricity, samples=4096))

    csv_path = report_dir / "eccentric_newtonian_harmonic_study.csv"
    with csv_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cutoff_csv = report_dir / "eccentric_newtonian_harmonic_cutoff.csv"
    with cutoff_csv.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(cutoff_rows[0]))
        writer.writeheader()
        writer.writerows(cutoff_rows)

    plot_path = report_dir / "eccentric_newtonian_harmonic_study.png"
    plot(rows, cutoff_rows, plot_path)
    write_report(rows, cutoff_rows, report_dir, plot_path)

    print(f"wrote {csv_path}")
    print(f"wrote {cutoff_csv}")
    print(f"wrote {plot_path}")
    print(f"wrote {report_dir / 'eccentric_newtonian_harmonic_study.md'}")


if __name__ == "__main__":
    main()
