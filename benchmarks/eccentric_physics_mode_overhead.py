"""Wall-time overhead of eccentric source physics switches.

Modes:
    newtonian          Keplerian eccentric orbit, no explicit 1PN corrections.
    1pn_no_periastron  1PN r/rdot/e_phi/phidot corrections, no periastron advance.
    1pn                Current full source model.

Run with:

    python benchmarks/eccentric_physics_mode_overhead.py
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

from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


def make_sources(n_sources: int, *, eccentricity: float = 0.3, seed: int = 1234) -> list[EccentricBinaryParams]:
    rng = np.random.default_rng(seed)
    return [
        EccentricBinaryParams(
            mean_motion=float(np.pi * rng.uniform(3.0e-4, 2.0e-3)),
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


def median_elapsed(func, repeats: int = 5) -> float:
    values = []
    for _ in range(repeats):
        _, seconds = elapsed(func)
        values.append(seconds)
    return float(np.median(values))


def link_relerr(model: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    numerator = 0.0
    denominator = 0.0
    for label in reference:
        numerator += np.linalg.norm(model[label] - reference[label]) ** 2
        denominator += np.linalg.norm(reference[label]) ** 2
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    modes = ["newtonian", "1pn_no_periastron", "1pn"]
    source_count = 512
    link_count = 128
    source_samples = 8192
    link_samples = 4096
    batch_size = 128
    repeats = 5

    source_times = np.linspace(0.0, 30.0 * 24.0 * 3600.0, source_samples, endpoint=False)
    link_times = np.linspace(0.0, 7.0 * 24.0 * 3600.0, link_samples, endpoint=False)
    source_sources = make_sources(source_count)
    link_sources = source_sources[:link_count]
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), link_times)
    geometry = precompute_jax_link_geometry(state)

    print(
        f"source_count={source_count}, source_samples={source_samples}, "
        f"link_count={link_count}, link_samples={link_samples}, batch_size={batch_size}"
    )

    rows = []
    full_links = None
    for mode in modes:
        _, source_compile_s = elapsed(
            lambda mode=mode: eccentric_complex_strain_batch_jax(source_sources[:batch_size], source_times, physics_mode=mode)
        )
        _, link_compile_s = elapsed(
            lambda mode=mode: eccentric_links_jax(link_sources[:batch_size], geometry, batch_size=batch_size, physics_mode=mode)
        )
        source_s = median_elapsed(
            lambda mode=mode: eccentric_complex_strain_batch_jax(source_sources, source_times, physics_mode=mode),
            repeats,
        )
        links, link_s = elapsed(lambda mode=mode: eccentric_links_jax(link_sources, geometry, batch_size=batch_size, physics_mode=mode))
        link_s_median = median_elapsed(
            lambda mode=mode: eccentric_links_jax(link_sources, geometry, batch_size=batch_size, physics_mode=mode),
            repeats,
        )

        if mode == "1pn":
            full_links = links
            residual = 0.0
        else:
            if full_links is None:
                full_links = eccentric_links_jax(link_sources, geometry, batch_size=batch_size, physics_mode="1pn")
            residual = link_relerr(links, full_links)

        row = {
            "physics_mode": mode,
            "source_count": source_count,
            "source_samples": source_samples,
            "link_count": link_count,
            "link_samples": link_samples,
            "batch_size": batch_size,
            "source_compile_warmup_s": source_compile_s,
            "link_compile_warmup_s": link_compile_s,
            "source_batch_s": source_s,
            "links_s": link_s_median,
            "link_residual_vs_1pn": residual,
        }
        rows.append(row)
        print(
            f"{mode:18s} | source {source_s:.4f}s | links {link_s_median:.4f}s | "
            f"compile source/link {source_compile_s:.3f}/{link_compile_s:.3f}s | "
            f"link residual vs 1pn {residual:.3e}"
        )

    csv_path = out_dir / "eccentric_physics_mode_overhead.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["physics_mode"] for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].bar(x - 0.18, [row["source_batch_s"] for row in rows], width=0.36, label="source only")
    axes[0].bar(x + 0.18, [row["links_s"] for row in rows], width=0.36, label="retarded links")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylabel("steady-state wall time [s]")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[0].set_title("runtime")

    axes[1].bar(x, [max(row["link_residual_vs_1pn"], 1.0e-16) for row in rows])
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylabel("relative link residual vs 1pn")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_title("physics impact")

    png_path = out_dir / "eccentric_physics_mode_overhead.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
