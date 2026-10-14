"""Plot the lisainstrument Welch PSD used by the manuscript.

This script reuses the saved CSV from ``lisainstrument_welch_psd.py`` so the
figure can be regenerated without rerunning the noise simulation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import LEGEND_SIZE, apply_tick_style, use_paper_plot_style


def read_psd_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        key: np.array([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


def main() -> None:
    use_paper_plot_style()
    out_dir = Path(__file__).resolve().parent
    data = read_psd_csv(out_dir / "lisainstrument_welch_psd.csv")
    frequency = data["frequency_hz"]
    positive = frequency > 0.0

    colors = {
        "X": "#0072B2",
        "Y": "#D55E00",
        "Z": "#009E73",
        "A": "#0072B2",
        "E": "#D55E00",
        "T": "#009E73",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.25), constrained_layout=True)
    for channel in ("X", "Y", "Z"):
        axes[0].loglog(
            frequency[positive],
            data[f"PSD_{channel}"][positive],
            color=colors[channel],
            lw=1.35,
            label=channel,
        )
    for channel in ("A", "E", "T"):
        axes[1].loglog(
            frequency[positive],
            data[f"PSD_{channel}"][positive],
            color=colors[channel],
            lw=1.35,
            label=channel,
        )

    for ax in axes:
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("Welch PSD")
        ax.grid(True, which="both", color="0.75", alpha=0.35, lw=0.6)
        ax.legend(frameon=False, fontsize=LEGEND_SIZE, ncol=3, loc="upper right")

    axes[0].set_title("X/Y/Z")
    axes[1].set_title("A/E/T")
    apply_tick_style(axes)

    png_path = out_dir / "lisainstrument_welch_psd_detail.png"
    fig.savefig(png_path, dpi=240)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
