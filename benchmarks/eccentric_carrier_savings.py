"""Compare eccentric harmonic-carrier triliear timing with brute-force links.

The brute-force path evaluates the full eccentric time-domain polarizations at
the sender and receiver retarded times for every source and link, then runs the
same pyTDI XYZ construction. The carrier path uses the small-e harmonic
decomposition in `compute_triliear_xyz_eccentric`.

Run with:

    python benchmarks/eccentric_carrier_savings.py

Outputs:

    benchmarks/eccentric_carrier_savings.csv
    benchmarks/eccentric_carrier_savings.png
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.constants import JULIAN_YEAR_S
from egb_multi.eccentric import EccentricBinaryParams, eccentric_polarizations
from egb_multi.external import default_lisaorbits, state_from_lisaorbits
from egb_multi.tdi import LINKS, LINK_LABELS, compute_triliear_xyz_eccentric, pytdi_data_from_links


def make_sources(n_sources: int, eccentricity: float, seed: int = 1234) -> list[EccentricBinaryParams]:
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


def median_elapsed(func, repeats: int) -> float:
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed))


def rotated_basis(source: EccentricBinaryParams) -> tuple[np.ndarray, np.ndarray]:
    eplus, ecross = source.polarization_basis()
    cosps = np.cos(2.0 * source.psi)
    sinps = np.sin(2.0 * source.psi)
    return cosps * eplus - sinps * ecross, sinps * eplus + cosps * ecross


def brute_force_eccentric_links(state, sources: list[EccentricBinaryParams]) -> dict[str, np.ndarray]:
    responses = {label: np.zeros(state.t.size, dtype=np.complex128) for label in LINK_LABELS}
    p = state.positions

    for source in sources:
        direction = source.propagation_direction()
        eplus, ecross = rotated_basis(source)
        for label, (sender, receiver) in zip(LINK_LABELS, LINKS, strict=True):
            n = state.unit_arm(sender, receiver)
            length = state.arm_length(sender, receiver)
            denominator = 1.0 - np.einsum("ti,i->t", n, direction)
            delay_sender = state.t - length - np.einsum("ti,i->t", p[:, sender, :], direction)
            delay_receiver = state.t - np.einsum("ti,i->t", p[:, receiver, :], direction)

            hplus_sender, hcross_sender = eccentric_polarizations(source, delay_sender)
            hplus_receiver, hcross_receiver = eccentric_polarizations(source, delay_receiver)
            qplus = np.einsum("ti,ij,tj->t", n, eplus, n)
            qcross = np.einsum("ti,ij,tj->t", n, ecross, n)
            h_sender = qplus * hplus_sender + qcross * hcross_sender
            h_receiver = qplus * hplus_receiver + qcross * hcross_receiver
            responses[label] += 0.5 * (h_sender - h_receiver) / denominator

    return responses


def brute_force_eccentric_xyz(state, sources: list[EccentricBinaryParams]) -> dict[str, np.ndarray]:
    from pytdi.michelson import compute_factorized_michelson

    data = pytdi_data_from_links(state, brute_force_eccentric_links(state, sources))
    return {
        "X": compute_factorized_michelson(data, rot=0, order=3, delay_order=3, generation=1, unit="frequency"),
        "Y": compute_factorized_michelson(data, rot=1, order=3, delay_order=3, generation=1, unit="frequency"),
        "Z": compute_factorized_michelson(data, rot=2, order=3, delay_order=3, generation=1, unit="frequency"),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    counts = [1, 2, 4, 8, 16, 32]
    eccentricity = 0.1
    repeats = 5
    times = np.linspace(0.0, JULIAN_YEAR_S, 2048, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)

    rows = []
    for count in counts:
        sources = make_sources(count, eccentricity)
        carrier_order1_s = median_elapsed(
            lambda: compute_triliear_xyz_eccentric(state, sources, eccentric_order=1, harmonic_samples=512),
            repeats,
        )
        carrier_order2_s = median_elapsed(
            lambda: compute_triliear_xyz_eccentric(state, sources, eccentric_order=2, harmonic_samples=512),
            repeats,
        )
        brute_s = median_elapsed(lambda: brute_force_eccentric_xyz(state, sources), repeats)
        rows.append(
            {
                "sources": count,
                "eccentricity": eccentricity,
                "carrier_order1_s": carrier_order1_s,
                "carrier_order2_s": carrier_order2_s,
                "brute_force_s": brute_s,
                "speedup_order1_vs_brute": brute_s / carrier_order1_s,
                "speedup_order2_vs_brute": brute_s / carrier_order2_s,
            }
        )
        print(
            f"{count:3d} sources | carrier order1 {carrier_order1_s:.6f} s | "
            f"carrier order2 {carrier_order2_s:.6f} s | brute force {brute_s:.6f} s | "
            f"speedup order1 {brute_s / carrier_order1_s:.2f}x"
        )

    csv_path = out_dir / "eccentric_carrier_savings.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["sources"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["carrier_order1_s"] for row in rows], marker="o", label="carrier order1")
    axes[0].plot(x, [row["carrier_order2_s"] for row in rows], marker="s", label="carrier order2")
    axes[0].plot(x, [row["brute_force_s"] for row in rows], marker="^", label="brute force eccentric")
    axes[1].plot(x, [row["speedup_order1_vs_brute"] for row in rows], marker="o", label="order1")
    axes[1].plot(x, [row["speedup_order2_vs_brute"] for row in rows], marker="s", label="order2")
    axes[0].set_ylabel("median wall time [s]")
    axes[1].set_ylabel("brute force / carrier speedup")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of sources")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_yscale("log")
    axes[0].set_title("wall time")
    axes[1].set_title("carrier speedup")

    png_path = out_dir / "eccentric_carrier_savings.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
