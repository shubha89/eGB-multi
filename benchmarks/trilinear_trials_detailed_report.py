"""Detailed report for trilinear response trials and JAXGBaccurate comparison.

Run with:

    python benchmarks/trilinear_trials_detailed_report.py

Inputs:

    reports/trilinear_trials/small_scale_circular_rerun.csv
    reports/trilinear_trials/small_scale_eccentric_order_rerun.csv
    benchmarks/jaxgbaccurate_calibrated_eccentric_study.csv

Outputs:

    reports/trilinear_trials_detailed/
"""

from __future__ import annotations

import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


REPORT_DATE = "2026-08-03"
F0_COLORS = {
    1.0e-4: "#0072B2",
    3.0e-4: "#009E73",
    1.0e-3: "#D55E00",
    3.0e-3: "#CC79A7",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def format_ms(seconds: float) -> str:
    return f"{1000.0 * seconds:.2f}"


def save_circular_plot(rows: list[dict[str, str]], path: Path) -> None:
    x = np.array([as_float(row, "sources") for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.plot(x, [as_float(row, "fastgb_s") for row in rows], marker="o", label="FastGB circular XYZ")
    ax.plot(x, [as_float(row, "trilinear_pair_s") for row in rows], marker="s", label="trilinear pair channels")
    ax.plot(x, [as_float(row, "trilinear_xyz_s") for row in rows], marker="D", label="trilinear pyTDI XYZ")
    ax.plot(x, [as_float(row, "exact_link_xyz_s") for row in rows], marker="^", label="exact-link pyTDI XYZ")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title("Small-scale circular timing rerun")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_eccentric_order_plot(rows: list[dict[str, str]], path: Path) -> None:
    x = np.array([as_float(row, "sources") for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for order in (0, 1, 2):
        ax.plot(x, [as_float(row, f"eccentric_order_{order}_s") for row in rows], marker="o", label=f"order {order}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of sources")
    ax.set_ylabel("median wall time [s]")
    ax.set_title("Small-scale eccentric harmonic-order timing")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_jaxgbaccurate_residual_plot(rows: list[dict[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    for f0, color in F0_COLORS.items():
        subset = sorted(
            [
                row
                for row in rows
                if np.isclose(as_float(row, "f0_hz"), f0) and as_float(row, "eccentricity") > 0.0
            ],
            key=lambda row: as_float(row, "eccentricity"),
        )
        x = np.array([as_float(row, "eccentricity") for row in subset])
        local = np.array([as_float(row, "local_e0_noise_weighted_residual_AE") for row in subset])
        conv = np.array([as_float(row, "noise_weighted_residual_AE_convention_removed") for row in subset])
        axes[0].plot(x, np.maximum(local, 1.0e-8), marker="o", color=color, label=f"{f0:.0e} Hz")
        axes[1].plot(x, np.maximum(conv, 1.0e-8), marker="o", color=color, label=f"{f0:.0e} Hz")

    axes[0].set_title("local eccentric vs local e=0")
    axes[1].set_title("JAXGBaccurate residual, e=0 floor removed")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("eccentricity")
        ax.set_ylabel("relative noise-weighted AE residual")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_jaxgbaccurate_timing_plot(rows: list[dict[str, str]], path: Path) -> None:
    e0_rows = sorted(
        [row for row in rows if np.isclose(as_float(row, "eccentricity"), 0.0)],
        key=lambda row: as_float(row, "f0_hz"),
    )
    e01_rows = sorted(
        [row for row in rows if np.isclose(as_float(row, "eccentricity"), 0.1)],
        key=lambda row: as_float(row, "f0_hz"),
    )
    x = np.array([as_float(row, "f0_hz") for row in e0_rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.plot(x, [as_float(row, "jaxgbaccurate_xyz_s") for row in e0_rows], marker="o", label="JAXGBaccurate circular")
    ax.plot(x, [as_float(row, "bruteforce_xyz_s") for row in e0_rows], marker="s", label="JAX brute e=0 links + pyTDI")
    ax.plot(
        [as_float(row, "f0_hz") for row in e01_rows],
        [as_float(row, "bruteforce_xyz_s") for row in e01_rows],
        marker="^",
        label="JAX brute e=0.1 links + pyTDI",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("carrier frequency [Hz]")
    ax.set_ylabel("wall time [s]")
    ax.set_title("JAXGBaccurate timing context")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def jaxgb_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    by_ecc: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        eccentricity = as_float(row, "eccentricity")
        if eccentricity > 0.0:
            by_ecc[eccentricity].append(row)

    summary = []
    for eccentricity in sorted(by_ecc):
        subset = by_ecc[eccentricity]
        local = np.array([as_float(row, "local_e0_noise_weighted_residual_AE") for row in subset])
        conv = np.array([as_float(row, "noise_weighted_residual_AE_convention_removed") for row in subset])
        best_index = int(np.nanargmin(conv))
        summary.append(
            {
                "eccentricity": eccentricity,
                "local_median": float(np.nanmedian(local)),
                "convention_removed_best": float(conv[best_index]),
                "best_f0": as_float(subset[best_index], "f0_hz"),
                "convention_removed_median": float(np.nanmedian(conv)),
            }
        )
    return summary


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    align = ["---:" for _ in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(
    path: Path,
    circular_rows: list[dict[str, str]],
    eccentric_rows: list[dict[str, str]],
    jaxgb_rows: list[dict[str, str]],
    jaxgb_summary: list[dict[str, float]],
    plots: dict[str, Path],
) -> None:
    circular_table = markdown_table(
        ["sources", "FastGB ms", "pair ms", "XYZ ms", "exact XYZ ms", "XYZ/FastGB"],
        [
            [
                row["sources"],
                format_ms(as_float(row, "fastgb_s")),
                format_ms(as_float(row, "trilinear_pair_s")),
                format_ms(as_float(row, "trilinear_xyz_s")),
                format_ms(as_float(row, "exact_link_xyz_s")),
                f"{as_float(row, 'trilinear_xyz_over_fastgb'):.1f}x",
            ]
            for row in circular_rows
        ],
    )
    eccentric_table = markdown_table(
        ["sources", "order 0 ms", "order 1 ms", "order 2 ms", "order2/order0"],
        [
            [
                row["sources"],
                format_ms(as_float(row, "eccentric_order_0_s")),
                format_ms(as_float(row, "eccentric_order_1_s")),
                format_ms(as_float(row, "eccentric_order_2_s")),
                f"{as_float(row, 'order_2_over_order_0'):.1f}x",
            ]
            for row in eccentric_rows
        ],
    )
    jaxgb_table = markdown_table(
        ["e", "median local residual", "best JAXGBacc residual", "best f0 Hz", "median JAXGBacc residual"],
        [
            [
                f"{row['eccentricity']:.1e}",
                f"{row['local_median']:.3e}",
                f"{row['convention_removed_best']:.3e}",
                f"{row['best_f0']:.1e}",
                f"{row['convention_removed_median']:.3e}",
            ]
            for row in jaxgb_summary
        ],
    )
    first_jaxgb = jaxgb_rows[0]
    path.write_text(
        "\n".join(
            [
                "# Detailed Trilinear LISA Response Trials",
                "",
                f"Generated: {REPORT_DATE}. The public Python API in this repo uses the legacy spelling `triliear`; this report uses `trilinear` in prose.",
                "",
                "## Implementation Map",
                "",
                "- `src/egb_multi/response.py`: trilinear tensor and source-batched contractions.",
                "- `src/egb_multi/tdi.py`: beta-expanded one-way links and pyTDI XYZ wrappers.",
                "- `src/egb_multi/eccentric.py`: eccentric polarizations and small-e harmonic carrier extraction.",
                "- `src/egb_multi/eccentric_jax.py`: batched JAX link kernels, including harmonic trilinear links.",
                "- `benchmarks/jaxgbaccurate_calibrated_eccentric_study.py`: calibrated JAXGBaccurate circular template vs local eccentric response.",
                "",
                "## Equations Implemented",
                "",
                "Carrier and chirp frequency:",
                "",
                "$$ f_eff(t) = f_0 + \\dot f t, \\qquad \\beta(t) = -2 \\pi i f_eff(t) L(t). $$",
                "",
                "Exact one-way Doppler response from the local implementation:",
                "",
                "$$ y_{sr}(t) = \\frac{1}{2} \\frac{n^T E n}{1 - n \\cdot k} "
                "\\left[h(t - L - p_s \\cdot k) - h(t - p_r \\cdot k)\\right]. $$",
                "",
                "The trilinear pair-channel approximation is a tensor contraction over the five trace-free polarization coefficients and affine sky direction:",
                "",
                "$$ W_c(t) = F(t) \\sum_{j=1}^{5} \\sum_{l=0}^{3} A_{cjl}(t; f_0, \\dot f) e_j \\tilde{k}_l, "
                "\\qquad \\tilde{k}=(1,k_x,k_y,k_z). $$",
                "",
                "For each pair channel, the tensor coefficients use:",
                "",
                "$$ q(n)=(n_x^2-n_z^2, 2n_xn_y, 2n_xn_z, n_y^2-n_z^2, 2n_yn_z), $$",
                "$$ a_pair = 2\\beta\\left(1+\\beta+\\frac{2}{3}\\beta^2\\right), "
                "\\qquad a_comp = -\\beta^2 \\frac{r_m}{L}(1+\\beta), $$",
                "$$ A_{cj0}=\\frac{1}{2}q_j a_pair, \\qquad A_{cj,1:3}=\\frac{1}{2}q_j a_comp. $$",
                "",
                "The beta-expanded link path uses a Taylor or exact finite-arm transfer:",
                "",
                "$$ z=-2\\pi i f_eff L(1-n\\cdot k), \\qquad T_N(z)=\\sum_{p=1}^{N}\\frac{z^p}{p!}, "
                "\\qquad T_exact(z)=e^z-1. $$",
                "",
                "The eccentric trilinear trial decomposes one radial period into Fourier harmonic carriers:",
                "",
                "$$ h_+(t), h_\\times(t) \\rightarrow \\hat h_{+,m}, \\hat h_{\\times,m}, \\qquad "
                "f_m=\\frac{m n}{2\\pi}. $$",
                "",
                "The implemented small-e tiers are order 0: `{2}`, order 1: `{1,2,3}`, and order 2: `{1,2,3,4}`.",
                "",
                "For the JAXGBaccurate comparison, the circular template is calibrated to the local e=0 response by one complex scale per A/E/T channel:",
                "",
                "$$ \\alpha_c = \\frac{\\langle m_c,r_c\\rangle}{\\langle m_c,m_c\\rangle}, \\qquad m'_c(t)=\\Re[\\alpha_c m_c(t)]. $$",
                "",
                "A/E accuracy metrics use the noise-weighted inner product:",
                "",
                "$$ \\langle a,b\\rangle_{AE}=4\\Delta f\\sum_{f\\in M}\\left(\\frac{a_A b_A^*}{S_A}+\\frac{a_E b_E^*}{S_E}\\right), $$",
                "$$ FF=\\frac{|\\langle h_e,h_c\\rangle|}{\\sqrt{\\langle h_e,h_e\\rangle\\langle h_c,h_c\\rangle}}, "
                "\\qquad mismatch=1-FF, $$",
                "$$ R_N=\\sqrt{\\langle h_e-h_c,h_e-h_c\\rangle/\\langle h_e,h_e\\rangle}. $$",
                "",
                "## Fresh Small-Scale Timing Rerun",
                "",
                circular_table,
                "",
                f"![Circular timing]({plots['circular'].name})",
                "",
                "The direct trilinear pair-channel contraction is substantially cheaper than trilinear XYZ because the latter enters the pyTDI channel construction. FastGB remains the faster circular baseline in this scale regime.",
                "",
                "## Eccentric Harmonic-Order Timing",
                "",
                eccentric_table,
                "",
                f"![Eccentric order timing]({plots['eccentric_order'].name})",
                "",
                "Runtime rises with harmonic order, consistent with the implemented carrier tiers. The order-2 path is about 1.7x to 3.3x the order-0 runtime over 1 to 16 sources in the fresh rerun.",
                "",
                "## JAXGBaccurate Calibrated Comparison",
                "",
                f"Input grid: {first_jaxgb['duration_days']} days, {first_jaxgb['samples']} samples, dt={float(first_jaxgb['dt_s']):.2f} s. The saved comparison covers f0 in 1e-4, 3e-4, 1e-3, 3e-3 Hz and eccentricities from 0 to 0.3.",
                "",
                jaxgb_table,
                "",
                f"![JAXGBaccurate residuals]({plots['jaxgb_residuals'].name})",
                "",
                f"![JAXGBaccurate timing]({plots['jaxgb_timing'].name})",
                "",
                "The useful physics trend is the local eccentric-vs-local-circular residual, plus the JAXGBaccurate residual after subtracting the e=0 convention floor. The raw calibrated JAXGBaccurate mismatch has a nonzero floor even at e=0, so it is not the clean eccentricity-only diagnostic.",
                "",
                "## Caveats",
                "",
                "- FastGB/JAXGB and the local response do not share every convention, so raw residuals are not final validation metrics.",
                "- The small-e harmonic trilinear approximation is explicitly experimental; the code recommends brute-force eccentric links for e >= 0.1.",
                "- Timing numbers are machine-local and should be rerun before quoting absolute performance.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def add_wrapped(ax, text: str, *, x: float = 0.04, y: float = 0.96, size: float = 10.0, width: int = 98) -> None:
    ax.axis("off")
    wrapped = "\n".join(textwrap.fill(part, width=width) for part in text.split("\n"))
    ax.text(x, y, wrapped, ha="left", va="top", fontsize=size, linespacing=1.24)


def add_equations(ax) -> None:
    ax.axis("off")
    lines = [
        r"$f_{eff}(t)=f_0+\dot f t,\quad \beta(t)=-2\pi i f_{eff}(t)L(t)$",
        r"$y_{sr}(t)=\frac{1}{2}\frac{n^TEn}{1-n\cdot k}\left[h(t-L-p_s\cdot k)-h(t-p_r\cdot k)\right]$",
        r"$W_c(t)=F(t)\sum_{j=1}^{5}\sum_{l=0}^{3}A_{cjl}(t)e_j\tilde{k}_l,\quad \tilde{k}=(1,k_x,k_y,k_z)$",
        r"$q(n)=(n_x^2-n_z^2,\;2n_xn_y,\;2n_xn_z,\;n_y^2-n_z^2,\;2n_yn_z)$",
        r"$a_{pair}=2\beta(1+\beta+\frac{2}{3}\beta^2),\quad a_{comp}=-\beta^2\frac{r_m}{L}(1+\beta)$",
        r"$A_{cj0}=\frac{1}{2}q_ja_{pair},\quad A_{cj,1:3}=\frac{1}{2}q_ja_{comp}$",
        r"$z=-2\pi i f_{eff}L(1-n\cdot k),\quad T_N(z)=\sum_{p=1}^{N}\frac{z^p}{p!},\quad T_{exact}=e^z-1$",
        r"$h_+(t),h_\times(t)\rightarrow \hat h_{+,m},\hat h_{\times,m},\quad f_m=\frac{mn}{2\pi}$",
    ]
    y = 0.96
    for line in lines:
        ax.text(0.04, y, line, ha="left", va="top", fontsize=11)
        y -= 0.105


def add_metrics(ax) -> None:
    ax.axis("off")
    lines = [
        r"$\alpha_c=\frac{\langle m_c,r_c\rangle}{\langle m_c,m_c\rangle},\quad m'_c(t)=\Re[\alpha_c m_c(t)]$",
        r"$\langle a,b\rangle_{AE}=4\Delta f\sum_{f\in M}\left(\frac{a_Ab_A^*}{S_A}+\frac{a_Eb_E^*}{S_E}\right)$",
        r"$FF=\frac{|\langle h_e,h_c\rangle|}{\sqrt{\langle h_e,h_e\rangle\langle h_c,h_c\rangle}},\quad mismatch=1-FF$",
        r"$R_N=\sqrt{\langle h_e-h_c,h_e-h_c\rangle/\langle h_e,h_e\rangle}$",
        r"$R_{conv-free}(e)=\sqrt{\max(R_N(e)^2-R_N(0)^2,0)}$",
    ]
    y = 0.92
    for line in lines:
        ax.text(0.04, y, line, ha="left", va="top", fontsize=11)
        y -= 0.15


def add_table(ax, headers: list[str], rows: list[list[str]], title: str) -> None:
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.3)
    ax.set_title(title, fontsize=10)


def add_image(ax, path: Path) -> None:
    ax.imshow(plt.imread(path))
    ax.axis("off")


def write_pdf(
    path: Path,
    circular_rows: list[dict[str, str]],
    eccentric_rows: list[dict[str, str]],
    jaxgb_summary: list[dict[str, float]],
    plots: dict[str, Path],
) -> None:
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.35, 2.4])
        add_wrapped(
            fig.add_subplot(gs[0]),
            "Detailed Trilinear LISA Response Trials\n"
            f"Generated {REPORT_DATE}. The code keeps the historical API spelling `triliear`, while the report uses `trilinear` in prose.",
            size=12,
        )
        add_wrapped(
            fig.add_subplot(gs[1]),
            "Implementation map\n"
            "- response.py: tensor A_cjl and vectorized source contractions.\n"
            "- tdi.py: beta-expanded one-way links and pyTDI X/Y/Z wrappers.\n"
            "- eccentric.py: quadrupole eccentric polarizations and small-e harmonic carriers.\n"
            "- eccentric_jax.py: batched JAX exact/interpolated/harmonic links.\n"
            "- jaxgbaccurate_calibrated_eccentric_study.py: calibrated JAXGBaccurate circular template versus local eccentric response.",
            size=9.3,
        )
        add_wrapped(
            fig.add_subplot(gs[2]),
            "Headline results\n"
            "FastGB remains the fastest circular baseline in the small rerun. Direct trilinear pair-channel evaluation is cheaper than pyTDI XYZ construction, while trilinear XYZ is close to exact-link XYZ at these counts. Eccentric runtime rises with harmonic order. In the JAXGBaccurate study, the clean eccentricity trend is best read from local e-vs-e0 residuals or from the convention-floor-subtracted JAXGBaccurate residuals.",
            size=10,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        add_equations(fig.add_subplot(111))
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.4])
        add_wrapped(
            fig.add_subplot(gs[0]),
            "JAXGBaccurate calibration and accuracy metrics\n"
            "The native JAXGBaccurate circular response is transformed back to the observation interval, converted to A/E/T, and calibrated channel-by-channel to the local e=0 response. The same calibrated circular template is then compared with local eccentric responses.",
            size=10,
        )
        add_metrics(fig.add_subplot(gs[1]))
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 3.0])
        add_table(
            fig.add_subplot(gs[0]),
            ["sources", "FastGB ms", "pair ms", "XYZ ms", "exact XYZ ms", "XYZ/FastGB"],
            [
                [
                    row["sources"],
                    format_ms(as_float(row, "fastgb_s")),
                    format_ms(as_float(row, "trilinear_pair_s")),
                    format_ms(as_float(row, "trilinear_xyz_s")),
                    format_ms(as_float(row, "exact_link_xyz_s")),
                    f"{as_float(row, 'trilinear_xyz_over_fastgb'):.1f}x",
                ]
                for row in circular_rows
            ],
            "Fresh small-scale circular timing rerun",
        )
        add_image(fig.add_subplot(gs[1]), plots["circular"])
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 3.0])
        add_table(
            fig.add_subplot(gs[0]),
            ["sources", "order 0 ms", "order 1 ms", "order 2 ms", "order2/order0"],
            [
                [
                    row["sources"],
                    format_ms(as_float(row, "eccentric_order_0_s")),
                    format_ms(as_float(row, "eccentric_order_1_s")),
                    format_ms(as_float(row, "eccentric_order_2_s")),
                    f"{as_float(row, 'order_2_over_order_0'):.1f}x",
                ]
                for row in eccentric_rows
            ],
            "Fresh eccentric harmonic-order timing rerun",
        )
        add_image(fig.add_subplot(gs[1]), plots["eccentric_order"])
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 2.8])
        add_table(
            fig.add_subplot(gs[0]),
            ["e", "median local", "best JAXGBacc", "best f0", "median JAXGBacc"],
            [
                [
                    f"{row['eccentricity']:.1e}",
                    f"{row['local_median']:.3e}",
                    f"{row['convention_removed_best']:.3e}",
                    f"{row['best_f0']:.1e}",
                    f"{row['convention_removed_median']:.3e}",
                ]
                for row in jaxgb_summary
            ],
            "JAXGBaccurate calibrated residual summary",
        )
        add_image(fig.add_subplot(gs[1]), plots["jaxgb_residuals"])
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0])
        add_image(fig.add_subplot(gs[0]), plots["jaxgb_timing"])
        add_wrapped(
            fig.add_subplot(gs[1]),
            "Caveats: FastGB/JAXGBaccurate and the local response have convention differences, so raw residuals include a floor. The small-e harmonic trilinear path is experimental; the code recommends brute-force eccentric links once e >= 0.1. Absolute timing numbers are machine-local.",
            size=9.5,
        )
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    short_report_dir = root / "reports" / "trilinear_trials"
    out_dir = root / "reports" / "trilinear_trials_detailed"
    out_dir.mkdir(parents=True, exist_ok=True)

    circular_csv = short_report_dir / "small_scale_circular_rerun.csv"
    eccentric_csv = short_report_dir / "small_scale_eccentric_order_rerun.csv"
    jaxgb_csv = root / "benchmarks" / "jaxgbaccurate_calibrated_eccentric_study.csv"
    missing = [str(path) for path in (circular_csv, eccentric_csv, jaxgb_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required report input(s): " + ", ".join(missing))

    circular_rows = read_rows(circular_csv)
    eccentric_rows = read_rows(eccentric_csv)
    jaxgb_rows = read_rows(jaxgb_csv)
    jaxgb_summary = jaxgb_summary_rows(jaxgb_rows)

    plots = {
        "circular": out_dir / "circular_timing.png",
        "eccentric_order": out_dir / "eccentric_order_timing.png",
        "jaxgb_residuals": out_dir / "jaxgbaccurate_residuals.png",
        "jaxgb_timing": out_dir / "jaxgbaccurate_timing.png",
    }
    save_circular_plot(circular_rows, plots["circular"])
    save_eccentric_order_plot(eccentric_rows, plots["eccentric_order"])
    save_jaxgbaccurate_residual_plot(jaxgb_rows, plots["jaxgb_residuals"])
    save_jaxgbaccurate_timing_plot(jaxgb_rows, plots["jaxgb_timing"])

    md_path = out_dir / "detailed_trilinear_trials_report.md"
    pdf_path = out_dir / "detailed_trilinear_trials_report.pdf"
    summary_csv = out_dir / "jaxgbaccurate_residual_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(jaxgb_summary[0]))
        writer.writeheader()
        writer.writerows(jaxgb_summary)
    write_markdown(md_path, circular_rows, eccentric_rows, jaxgb_rows, jaxgb_summary, plots)
    write_pdf(pdf_path, circular_rows, eccentric_rows, jaxgb_summary, plots)

    print(f"wrote {md_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {summary_csv}")
    for plot_path in plots.values():
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
