"""Plot one-month exact vs cubic-interpolated eccentric XYZ channels.

Run with:

    python benchmarks/month_e03_xyz_waveform_plot.py

Outputs:

    benchmarks/month_e03_xyz_full.png
    benchmarks/month_e03_xyz_zoom.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jax_bruteforce_links_walltime import xyz_from_links
from month_e03_interpolation_benchmark import make_sources, xyz_relerr

from egb_multi.eccentric_jax import eccentric_links_interpolated_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


def plot_xyz(
    times_days: np.ndarray,
    exact_xyz: dict[str, np.ndarray],
    cubic_xyz: dict[str, np.ndarray],
    path: Path,
    *,
    title: str,
    sample_slice: slice,
) -> None:
    channels = ("X", "Y", "Z")
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.0), sharex=True, constrained_layout=True)
    for ax, channel in zip(axes, channels, strict=True):
        t = times_days[sample_slice]
        exact = np.real(exact_xyz[channel][sample_slice])
        cubic = np.real(cubic_xyz[channel][sample_slice])
        residual = cubic - exact
        ax.plot(t, exact, linewidth=1.0, label="exact-retarded JAX")
        ax.plot(t, cubic, linewidth=0.8, linestyle="--", label="cubic interpolated JAX")
        ax.plot(t, residual, linewidth=0.7, alpha=0.8, label="cubic - exact")
        ax.set_ylabel(f"Re {channel}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [days]")
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    n_sources = 64
    eccentricity = 0.3
    batch_size = 8
    interpolation_order = 3
    duration = 30.0 * 24.0 * 3600.0
    dt = 10.0
    samples = int(duration / dt)
    times = np.arange(samples, dtype=np.float64) * dt
    times_days = times / 86400.0

    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    sources = make_sources(n_sources, eccentricity)

    print(
        f"sources={n_sources}, e={eccentricity}, samples={samples}, dt={dt:.3f} s, "
        f"batch_size={batch_size}, interpolation_order={interpolation_order}"
    )

    eccentric_links_jax(sources[:batch_size], geometry, batch_size=batch_size)
    eccentric_links_interpolated_jax(
        sources[:batch_size],
        geometry,
        batch_size=batch_size,
        interpolation_order=interpolation_order,
    )

    exact_links = eccentric_links_jax(sources, geometry, batch_size=batch_size)
    exact_xyz = xyz_from_links(state, exact_links)
    cubic_links = eccentric_links_interpolated_jax(
        sources,
        geometry,
        batch_size=batch_size,
        interpolation_order=interpolation_order,
    )
    cubic_xyz = xyz_from_links(state, cubic_links)
    residual = xyz_relerr(cubic_xyz, exact_xyz)
    print(f"XYZ cubic residual: {residual:.3e}")

    full_step = max(1, samples // 12000)
    full_path = out_dir / "month_e03_xyz_full.png"
    plot_xyz(
        times_days,
        exact_xyz,
        cubic_xyz,
        full_path,
        title=f"64 eccentric sources, e=0.3, 30 days; cubic XYZ residual {residual:.3e}",
        sample_slice=slice(None, None, full_step),
    )

    zoom_hours = 6.0
    zoom_samples = int(zoom_hours * 3600.0 / dt)
    zoom_start = samples // 2
    zoom_stop = min(samples, zoom_start + zoom_samples)
    zoom_path = out_dir / "month_e03_xyz_zoom.png"
    plot_xyz(
        times_days,
        exact_xyz,
        cubic_xyz,
        zoom_path,
        title=f"Zoom: {zoom_hours:g} hours near day {times_days[zoom_start]:.2f}",
        sample_slice=slice(zoom_start, zoom_stop),
    )

    print(f"wrote {full_path}")
    print(f"wrote {zoom_path}")


if __name__ == "__main__":
    main()
