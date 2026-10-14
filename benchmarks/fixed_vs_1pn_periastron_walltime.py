"""Wall-time comparison for fixed eccentric and full 1PN+periastron models.

This benchmark compares two fixed-eccentricity source settings:

* ``newtonian``: fixed eccentric Keplerian source;
* ``1pn``: fixed eccentric source with 1PN orbital corrections and periastron
  advance.

The timed response path is the accelerated JAX retarded-link calculation used
for the response-level studies, followed optionally by pyTDI XYZ construction.
Peters--Mathews radiation reaction is not included in this JAX link benchmark.

Run with:

    python benchmarks/fixed_vs_1pn_periastron_walltime.py
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

from jax_bruteforce_links_walltime import xyz_from_links

from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


def make_sources(n_sources: int, *, eccentricity: float = 0.3, seed: int = 20260810) -> list[EccentricBinaryParams]:
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


def elapsed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def median_elapsed(func, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        _, seconds = elapsed(func)
        values.append(seconds)
    return float(np.median(values))


def link_relerr(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = sum(np.linalg.norm(model[label] - reference[label]) ** 2 for label in reference)
    denominator = sum(np.linalg.norm(reference[label]) ** 2 for label in reference)
    return float(np.sqrt(numerator / denominator))


def xyz_relerr(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = sum(np.linalg.norm(model[channel] - reference[channel]) ** 2 for channel in ("X", "Y", "Z"))
    denominator = sum(np.linalg.norm(reference[channel]) ** 2 for channel in ("X", "Y", "Z"))
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 128
    eccentricity = 0.3
    batch_size = 128
    repeats = 7
    duration_days = 7.0
    samples = 4096
    times = np.linspace(0.0, duration_days * 24.0 * 3600.0, samples, endpoint=False)
    sources = make_sources(n_sources, eccentricity=eccentricity)

    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)

    modes = [
        ("fixed eccentric", "newtonian"),
        ("full 1PN+periastron", "1pn"),
    ]

    print(
        f"sources={n_sources}, e={eccentricity}, samples={samples}, "
        f"duration_days={duration_days}, batch_size={batch_size}, repeats={repeats}"
    )

    rows = []
    cached_links: dict[str, dict[str, np.ndarray]] = {}
    cached_xyz: dict[str, dict[str, np.ndarray]] = {}

    for _, mode in modes:
        eccentric_complex_strain_batch_jax(sources, times, physics_mode=mode)
        links = eccentric_links_jax(sources, geometry, batch_size=batch_size, physics_mode=mode)
        xyz_from_links(state, links)

    for label, mode in modes:
        source_s = median_elapsed(
            lambda mode=mode: eccentric_complex_strain_batch_jax(sources, times, physics_mode=mode),
            repeats,
        )
        links, _ = elapsed(lambda mode=mode: eccentric_links_jax(sources, geometry, batch_size=batch_size, physics_mode=mode))
        links_s = median_elapsed(
            lambda mode=mode: eccentric_links_jax(sources, geometry, batch_size=batch_size, physics_mode=mode),
            repeats,
        )
        pytdi_xyz_s = median_elapsed(
            lambda links=links: xyz_from_links(state, links),
            repeats,
        )
        cached_links[mode] = links
        cached_xyz[mode] = xyz_from_links(state, links)
        rows.append(
            {
                "label": label,
                "physics_mode": mode,
                "sources": n_sources,
                "eccentricity": eccentricity,
                "samples": samples,
                "duration_days": duration_days,
                "batch_size": batch_size,
                "repeats": repeats,
                "source_batch_s": source_s,
                "links_s": links_s,
                "pytdi_xyz_only_s": pytdi_xyz_s,
                "links_plus_pytdi_xyz_s": links_s + pytdi_xyz_s,
            }
        )

    full_links = cached_links["1pn"]
    full_xyz = cached_xyz["1pn"]
    for row in rows:
        mode = row["physics_mode"]
        row["link_relative_residual_vs_full_1pn"] = link_relerr(cached_links[mode], full_links)
        row["xyz_relative_residual_vs_full_1pn"] = xyz_relerr(cached_xyz[mode], full_xyz)

    csv_path = out_dir / "fixed_vs_1pn_periastron_walltime.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.bar(x - width, [row["source_batch_s"] for row in rows], width=width, label="source only")
    ax.bar(x, [row["links_s"] for row in rows], width=width, label="retarded links")
    ax.bar(x + width, [row["links_plus_pytdi_xyz_s"] for row in rows], width=width, label="links + pyTDI XYZ")
    ax.set_xticks(x, labels)
    ax.set_ylabel("median wall time [s]")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    png_path = out_dir / "fixed_vs_1pn_periastron_walltime.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    for row in rows:
        print(
            f"{row['label']:22s} | source {row['source_batch_s']:.4f}s | "
            f"links {row['links_s']:.4f}s | links+XYZ {row['links_plus_pytdi_xyz_s']:.4f}s | "
            f"pyTDI only {row['pytdi_xyz_only_s']:.4f}s | "
            f"XYZ residual vs full 1PN {row['xyz_relative_residual_vs_full_1pn']:.3e}"
        )
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
