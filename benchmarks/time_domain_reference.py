"""Compare models against a brute-force time-domain response.

Two comparisons are reported:

1. beta-triliear pair channels vs brute-force pair channels,
2. beta-link pyTDI X/Y/Z, exact-link pyTDI X/Y/Z, and FastGB sparse inverse-FFT
   X/Y/Z vs brute-force pyTDI X/Y/Z.

Run with:

    python benchmarks/time_domain_reference.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egb_multi.external import FastGBBackend, default_lisaorbits, state_from_lisaorbits
from egb_multi.reference import brute_force_pair_channels, brute_force_xyz
from egb_multi.response import triliear_tensors, triliear_waveforms
from egb_multi.sources import GalacticBinaryParams
from egb_multi.tdi import compute_beta_xyz, compute_triliear_xyz, compute_xyz


def best_fit_scale(model: np.ndarray, reference: np.ndarray) -> complex:
    denom = np.vdot(model, model)
    return 0.0j if denom == 0 else complex(np.vdot(model, reference) / denom)


def relerr(model: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def fastgb_sparse_time_xyz(result, n_time: int, dt: float) -> dict[str, np.ndarray]:
    """Reconstruct a complex time series from FastGB's narrow-band FD output."""

    out = {}
    for name, fd in {"X": result.x[0], "Y": result.y[0], "Z": result.z[0]}.items():
        spectrum = np.zeros(n_time, dtype=np.complex128)
        kmin = int(result.kmin[0])
        spectrum[kmin : kmin + fd.size] = fd / dt
        out[name] = np.fft.ifft(spectrum)
    return out


def analytic(x: np.ndarray) -> np.ndarray:
    return hilbert(np.real(x))


def beta_triliear_pair_full_carrier(state, source: GalacticBinaryParams) -> np.ndarray:
    carrier = np.exp(1.0j * (2.0 * np.pi * source.f0 * state.t + np.pi * source.fdot * state.t**2 + source.phi0))
    model = triliear_waveforms(
        carrier[None, :],
        triliear_tensors(state, [source.f0], fdot=[source.fdot]),
        source.tracefree_coefficients()[None, :],
        source.propagation_direction()[None, :],
    )
    return model[0]


def compare_frequency(f0: float, out_dir: Path) -> dict[str, float]:
    dt = 100.0
    duration = 30.0 * 24.0 * 3600.0
    n_time = int(duration / dt)
    duration = n_time * dt
    t = np.arange(n_time, dtype=np.float64) * dt

    source = GalacticBinaryParams(
        f0=f0,
        fdot=0.0,
        amplitude=1.0e-22,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=1.0,
        phi0=0.2,
    )
    orbit_kind = "esa-trailing"
    orbits = default_lisaorbits(orbit_kind)
    state = state_from_lisaorbits(orbits, t)

    brute_pairs = brute_force_pair_channels(state, source, orbits)
    triliear_pairs = beta_triliear_pair_full_carrier(state, source)

    fastgb = FastGBBackend(delta_t=dt, duration=duration, n_samples=128, orbits=orbits)
    fast_time = fastgb_sparse_time_xyz(fastgb.frequency_domain_xyz(source), n_time, dt)
    beta_xyz_by_order = {
        "triliear_default": compute_triliear_xyz(state, source),
        "beta1": compute_beta_xyz(state, source, order=1, generation=1, measurement_order=5, delay_order=5),
        "beta2": compute_beta_xyz(state, source, order=2, generation=1, measurement_order=5, delay_order=5),
        "beta3": compute_beta_xyz(state, source, order=3, generation=1, measurement_order=5, delay_order=5),
        "beta_exact_transfer": compute_beta_xyz(
            state,
            source,
            order="exact-transfer",
            generation=1,
            measurement_order=5,
            delay_order=5,
        ),
        "beta1_retarded": compute_beta_xyz(
            state,
            source,
            order=1,
            retarded_geometry=True,
            generation=1,
            measurement_order=5,
            delay_order=5,
        ),
        "beta_exact_transfer_retarded": compute_beta_xyz(
            state,
            source,
            order="exact-transfer",
            retarded_geometry=True,
            generation=1,
            measurement_order=5,
            delay_order=5,
        ),
        "beta1_bary": compute_beta_xyz(
            state,
            source,
            order=1,
            barycentric_phase=True,
            generation=1,
            measurement_order=5,
            delay_order=5,
        ),
        "beta_exact_transfer_bary": compute_beta_xyz(
            state,
            source,
            order="exact-transfer",
            barycentric_phase=True,
            generation=1,
            measurement_order=5,
            delay_order=5,
        ),
        "beta_exact_transfer_retarded_bary": compute_beta_xyz(
            state,
            source,
            order="exact-transfer",
            retarded_geometry=True,
            barycentric_phase=True,
            generation=1,
            measurement_order=5,
            delay_order=5,
        ),
    }
    exact_link_xyz = compute_xyz(state, source, generation=1, measurement_order=5, delay_order=5)
    brute_xyz = brute_force_xyz(state, source, orbits, generation=1, measurement_order=5, delay_order=5)

    row: dict[str, float | str] = {"f0": f0, "orbit": orbit_kind}
    fig, axes = plt.subplots(3, 4, figsize=(18.0, 7.0), sharex=True, constrained_layout=True)
    window = slice(0, min(n_time, 2000))
    tt_days = t[window] / 86400.0

    for i, channel in enumerate(("P12", "P13", "P23")):
        model = triliear_pairs[:, i]
        reference = analytic(brute_pairs[:, i])
        alpha = best_fit_scale(model, reference)
        scaled = alpha * model
        row[f"{channel}_triliear_scaled_relerr"] = relerr(scaled, reference)
        row[f"{channel}_triliear_raw_relerr"] = relerr(model, reference)

        axes[i, 0].plot(tt_days, np.real(reference[window]), label="brute force analytic", linewidth=1.0)
        axes[i, 0].plot(tt_days, np.real(scaled[window]), label="scaled beta-triliear", linestyle="--", linewidth=1.0)
        axes[i, 0].set_ylabel(channel)
        axes[i, 0].grid(True, alpha=0.3)

    for i, channel in enumerate(("X", "Y", "Z")):
        reference = analytic(brute_xyz[channel])

        best_beta_name = ""
        best_beta_scaled = None
        best_beta_err = np.inf
        for beta_name, beta_xyz in beta_xyz_by_order.items():
            beta_model = analytic(beta_xyz[channel])
            beta_alpha = best_fit_scale(beta_model, reference)
            beta_scaled = beta_alpha * beta_model
            beta_err = relerr(beta_scaled, reference)
            row[f"{channel}_{beta_name}_scaled_relerr"] = beta_err
            row[f"{channel}_{beta_name}_raw_relerr"] = relerr(beta_model, reference)
            if beta_err < best_beta_err:
                best_beta_name = beta_name
                best_beta_scaled = beta_scaled
                best_beta_err = beta_err

        exact_model = analytic(exact_link_xyz[channel])
        exact_alpha = best_fit_scale(exact_model, reference)
        exact_scaled = exact_alpha * exact_model
        row[f"{channel}_exact_link_scaled_relerr"] = relerr(exact_scaled, reference)
        row[f"{channel}_exact_link_raw_relerr"] = relerr(exact_model, reference)

        fast_model = fast_time[channel]
        fast_alpha = best_fit_scale(fast_model, reference)
        fast_scaled = fast_alpha * fast_model
        row[f"{channel}_fastgb_scaled_relerr"] = relerr(fast_scaled, reference)
        row[f"{channel}_fastgb_raw_relerr"] = relerr(fast_model, reference)

        axes[i, 1].plot(tt_days, np.real(reference[window]), label="brute force analytic", linewidth=1.0)
        assert best_beta_scaled is not None
        axes[i, 1].plot(tt_days, np.real(best_beta_scaled[window]), label=f"scaled {best_beta_name}", linestyle="--", linewidth=1.0)
        axes[i, 1].set_ylabel(channel)
        axes[i, 1].grid(True, alpha=0.3)

        axes[i, 2].plot(tt_days, np.real(reference[window]), label="brute force analytic", linewidth=1.0)
        axes[i, 2].plot(tt_days, np.real(exact_scaled[window]), label="scaled exact-link pyTDI", linestyle="--", linewidth=1.0)
        axes[i, 2].set_ylabel(channel)
        axes[i, 2].grid(True, alpha=0.3)

        axes[i, 3].plot(tt_days, np.real(reference[window]), label="brute force analytic", linewidth=1.0)
        axes[i, 3].plot(tt_days, np.real(fast_scaled[window]), label="scaled FastGB sparse iFFT", linestyle="--", linewidth=1.0)
        axes[i, 3].set_ylabel(channel)
        axes[i, 3].grid(True, alpha=0.3)

    axes[0, 0].set_title("pair channels: brute force vs beta-triliear")
    axes[0, 1].set_title("X/Y/Z: brute force vs best beta-link pyTDI")
    axes[0, 2].set_title("X/Y/Z: brute force vs exact-link pyTDI")
    axes[0, 3].set_title("X/Y/Z: brute force vs FastGB sparse iFFT")
    axes[-1, 0].set_xlabel("time [days]")
    axes[-1, 1].set_xlabel("time [days]")
    axes[-1, 2].set_xlabel("time [days]")
    axes[-1, 3].set_xlabel("time [days]")
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[0, 2].legend()
    axes[0, 3].legend()
    fig.suptitle(f"Brute-force time-domain reference, {orbit_kind}, f0={f0:.1e} Hz")
    plot = out_dir / f"time_domain_reference_{orbit_kind}_f0_{f0:.0e}.png"
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    return row


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = [compare_frequency(f0, out_dir) for f0 in (1.0e-4, 3.0e-4, 1.0e-3)]
    csv_path = out_dir / "time_domain_reference.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"orbit={row['orbit']} f0={row['f0']:.1e}")
        print(
            "  beta-triliear pair scaled relerr:",
            [f"{row[f'{ch}_triliear_scaled_relerr']:.3e}" for ch in ("P12", "P13", "P23")],
        )
        print(
            "  exact-link XYZ scaled relerr:",
            [f"{row[f'{ch}_exact_link_scaled_relerr']:.3e}" for ch in ("X", "Y", "Z")],
        )
        for beta_name in (
            "triliear_default",
            "beta1",
            "beta2",
            "beta3",
            "beta_exact_transfer",
            "beta1_retarded",
            "beta_exact_transfer_retarded",
            "beta1_bary",
            "beta_exact_transfer_bary",
            "beta_exact_transfer_retarded_bary",
        ):
            print(
                f"  {beta_name} XYZ scaled relerr:",
                [f"{row[f'{ch}_{beta_name}_scaled_relerr']:.3e}" for ch in ("X", "Y", "Z")],
            )
        print(
            "  FastGB XYZ scaled relerr:",
            [f"{row[f'{ch}_fastgb_scaled_relerr']:.3e}" for ch in ("X", "Y", "Z")],
        )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
