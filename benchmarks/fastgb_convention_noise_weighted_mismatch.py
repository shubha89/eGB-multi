"""Noise-weighted mismatch between eccentric and circular GB waveforms.

This uses the Welch PSD estimated from `lisainstrument` noise. Waveforms are
generated with the FastGB-convention response algebra, converted from X/Y/Z to
A/E/T in the frequency domain, and compared with a multi-channel inner product.

Run with:

    python benchmarks/fastgb_convention_noise_weighted_mismatch.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastgb_convention_eccentric_accuracy import as_fastgb_source

from egb_multi.constants import KILOPARSEC_M
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import FastGBBackend, FastGBResult, default_lisaorbits
from egb_multi.fastgb_convention import fastgb_convention_circular_xyz, fastgb_convention_eccentric_xyz
from egb_multi.noise import estimate_channel_psds, simulate_lisainstrument_tdi_noise


def ensure_welch_psd(psd_csv: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the benchmark Welch PSD, generating it if needed."""

    if not psd_csv.exists():
        noise = simulate_lisainstrument_tdi_noise(size=8192, dt=10.0, seed=1234, order=5, delay_order=5)
        freq, psd = estimate_channel_psds(noise.aet, noise.dt, nperseg=2048)
        with psd_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frequency_hz", "PSD_A", "PSD_E", "PSD_T"])
            writer.writeheader()
            for i, frequency in enumerate(freq):
                writer.writerow(
                    {
                        "frequency_hz": frequency,
                        "PSD_A": psd["A"][i],
                        "PSD_E": psd["E"][i],
                        "PSD_T": psd["T"][i],
                    }
                )

    rows = []
    with psd_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    freq = np.array([float(row["frequency_hz"]) for row in rows], dtype=np.float64)
    psd = {
        "A": np.array([float(row["PSD_A"]) for row in rows], dtype=np.float64),
        "E": np.array([float(row["PSD_E"]) for row in rows], dtype=np.float64),
        "T": np.array([float(row["PSD_T"]) for row in rows], dtype=np.float64),
    }
    return freq, psd


def result_to_padded_xyz(result: FastGBResult, kmin: int, kmax: int) -> dict[str, np.ndarray]:
    out = {}
    start = int(result.kmin[0]) - kmin
    for name in ("x", "y", "z"):
        values = getattr(result, name)[0]
        padded = np.zeros(kmax - kmin, dtype=np.complex128)
        padded[start : start + values.size] = values
        out[name.upper()] = padded
    return out


def xyz_to_aet_fd(xyz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = xyz["X"]
    y = xyz["Y"]
    z = xyz["Z"]
    return {
        "A": (z - x) / np.sqrt(2.0),
        "E": (x - 2.0 * y + z) / np.sqrt(6.0),
        "T": (x + y + z) / np.sqrt(3.0),
    }


def multi_channel_auto_inner(
    left: dict[str, np.ndarray],
    psd_interp: dict[str, np.ndarray],
    df: float,
    *,
    channels: tuple[str, ...] = ("A", "E"),
) -> float:
    total = 0.0
    for channel in channels:
        psd = psd_interp[channel]
        mask = np.isfinite(psd) & (psd > 0.0)
        total += 4.0 * np.real(np.sum(left[channel][mask] * np.conjugate(left[channel][mask]) / psd[mask])) * df
    return float(total)


def multi_channel_cross_inner_complex(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    psd_interp: dict[str, np.ndarray],
    df: float,
    *,
    channels: tuple[str, ...] = ("A", "E"),
) -> complex:
    total = 0.0j
    for channel in channels:
        psd = psd_interp[channel]
        mask = np.isfinite(psd) & (psd > 0.0)
        total += 4.0 * np.sum(left[channel][mask] * np.conjugate(right[channel][mask]) / psd[mask]) * df
    return complex(total)


def multi_channel_fitting_factor(
    reference: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    psd_interp: dict[str, np.ndarray],
    df: float,
    *,
    channels: tuple[str, ...] = ("A", "E"),
) -> float:
    rr = multi_channel_auto_inner(reference, psd_interp, df, channels=channels)
    tt = multi_channel_auto_inner(template, psd_interp, df, channels=channels)
    if rr <= 0.0 or tt <= 0.0:
        return 0.0
    rt = multi_channel_cross_inner_complex(reference, template, psd_interp, df, channels=channels)
    return float(abs(rt) / np.sqrt(rr * tt))


def compare_one(
    backend: FastGBBackend,
    psd_freq: np.ndarray,
    psd: dict[str, np.ndarray],
    *,
    f0: float,
    eccentricity: float,
    eccentric_order: int,
    channels: tuple[str, ...],
) -> dict[str, float]:
    source = EccentricBinaryParams(
        mean_motion=np.pi * f0,
        eccentricity=eccentricity,
        m1_solar=0.5,
        m2_solar=0.5,
        distance_m=KILOPARSEC_M,
        beta=0.3,
        lambda_=-0.7,
        psi=0.4,
        inclination=1.0,
        phi0=0.2,
    )
    circular_source = as_fastgb_source(EccentricBinaryParams(**{**source.__dict__, "eccentricity": 0.0}))
    circular = fastgb_convention_circular_xyz(backend, circular_source)
    eccentric = fastgb_convention_eccentric_xyz(
        backend,
        source,
        order=eccentric_order,
        samples=2048,
        circular_reference=circular_source,
    )

    kmin = min(int(circular.kmin[0]), int(eccentric.kmin[0]))
    kmax = max(int(circular.kmin[0]) + circular.x.shape[-1], int(eccentric.kmin[0]) + eccentric.x.shape[-1])
    df = 1.0 / backend.model.T
    freq = np.arange(kmin, kmax, dtype=np.float64) * df
    psd_interp = {
        channel: np.interp(freq, psd_freq, psd[channel], left=np.inf, right=np.inf)
        for channel in ("A", "E", "T")
    }

    circular_aet = xyz_to_aet_fd(result_to_padded_xyz(circular, kmin, kmax))
    eccentric_aet = xyz_to_aet_fd(result_to_padded_xyz(eccentric, kmin, kmax))
    ff = multi_channel_fitting_factor(eccentric_aet, circular_aet, psd_interp, df, channels=channels)
    return {
        "f0": f0,
        "eccentricity": eccentricity,
        "eccentric_order": eccentric_order,
        "channels": "".join(channels),
        "kmin": kmin,
        "kmax": kmax,
        "frequency_min_hz": float(freq[0]),
        "frequency_max_hz": float(freq[-1]),
        "fitting_factor": ff,
        "mismatch": 1.0 - ff,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    psd_freq, psd = ensure_welch_psd(out_dir / "lisainstrument_welch_psd.csv")
    backend = FastGBBackend(
        delta_t=100.0,
        duration=90.0 * 24.0 * 3600.0,
        n_samples=128,
        orbits=default_lisaorbits("equal"),
    )
    frequencies = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3]
    eccentricities = [0.0, 1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, 1.0e-1, 2.0e-1, 3.0e-1]
    channels = ("A", "E")
    eccentric_order = 2

    rows = []
    for f0 in frequencies:
        for eccentricity in eccentricities:
            row = compare_one(
                backend,
                psd_freq,
                psd,
                f0=f0,
                eccentricity=eccentricity,
                eccentric_order=eccentric_order,
                channels=channels,
            )
            rows.append(row)
            print(
                f"f0={f0:.1e} Hz e={eccentricity:.4f} | "
                f"FF={row['fitting_factor']:.8f} | mismatch={row['mismatch']:.3e}"
            )

    csv_path = out_dir / "fastgb_convention_noise_weighted_mismatch.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for f0 in frequencies:
        subset = [row for row in rows if row["f0"] == f0]
        ax.plot(
            [row["eccentricity"] for row in subset],
            [max(row["mismatch"], 1.0e-16) for row in subset],
            marker="o",
            label=f"f0={f0:.0e} Hz",
        )
    ax.set_xscale("symlog", linthresh=1.0e-4)
    ax.set_yscale("log")
    ax.set_xlabel("eccentricity")
    ax.set_ylabel("noise-weighted mismatch, 1 - FF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Eccentric vs circular GB mismatch using lisainstrument Welch PSD")
    png_path = out_dir / "fastgb_convention_noise_weighted_mismatch.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
