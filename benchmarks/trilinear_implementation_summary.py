"""Concise implementation summary for the trilinear response work.

Run with:

    python benchmarks/trilinear_implementation_summary.py

Outputs:

    reports/trilinear_implementation_summary/
"""

from __future__ import annotations

import csv
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPORT_DATE = "2026-08-03"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wrapped_text(ax, text: str, *, x: float = 0.04, y: float = 0.96, size: float = 9.8, width: int = 92) -> None:
    ax.axis("off")
    wrapped = "\n".join(textwrap.fill(part, width=width) for part in text.split("\n"))
    ax.text(x, y, wrapped, ha="left", va="top", fontsize=size, linespacing=1.22)


def equations_page_text() -> str:
    return (
        "Implementation in one paragraph\n"
        "The code builds a low-frequency trilinear LISA response tensor A_cjl(t) for each pair channel, "
        "contracts it with five trace-free polarization coefficients e_j and the affine source direction "
        "k_tilde=(1,kx,ky,kz), then optionally routes beta-expanded six-link responses through pyTDI to get "
        "Michelson X/Y/Z. The eccentric trial first decomposes one radial period of the source waveform into "
        "a small set of harmonic carriers and evaluates those carriers with the same trilinear/beta response machinery.\n\n"
        "Key implemented equations\n"
        r"$f_{eff}(t)=f_0+\dot f t,\quad \beta(t)=-2\pi i f_{eff}(t)L(t)$" "\n\n"
        r"$y_{sr}(t)=\frac{1}{2}\frac{n^TEn}{1-n\cdot k}[h(t-L-p_s\cdot k)-h(t-p_r\cdot k)]$" "\n\n"
        r"$W_c(t)=F(t)\sum_{j=1}^{5}\sum_{l=0}^{3}A_{cjl}(t)e_j\tilde{k}_l$" "\n\n"
        r"$q(n)=(n_x^2-n_z^2,\;2n_xn_y,\;2n_xn_z,\;n_y^2-n_z^2,\;2n_yn_z)$" "\n\n"
        r"$a_{pair}=2\beta(1+\beta+\frac{2}{3}\beta^2),\quad a_{comp}=-\beta^2\frac{r_m}{L}(1+\beta)$" "\n\n"
        r"$A_{cj0}=\frac{1}{2}q_ja_{pair},\quad A_{cj,1:3}=\frac{1}{2}q_ja_{comp}$" "\n\n"
        r"$h_+(t),h_\times(t)\rightarrow \hat h_{+,m},\hat h_{\times,m},\quad f_m=\frac{mn}{2\pi}$"
    )


def comparison_page_text(summary_rows: list[dict[str, str]]) -> str:
    by_e = {float(row["eccentricity"]): row for row in summary_rows}
    e001 = by_e.get(0.01)
    e01 = by_e.get(0.1)
    e03 = by_e.get(0.3)

    lines = [
        "The saved accurate comparison calibrates a circular JAXGBaccurate template to the local e=0 response by one complex scale per A/E/T channel, then compares it to local eccentric responses.",
        "",
        r"$\alpha_c=\frac{\langle m_c,r_c\rangle}{\langle m_c,m_c\rangle},\quad m'_c(t)=\Re[\alpha_cm_c(t)]$",
        "",
        r"$FF=\frac{|\langle h_e,h_c\rangle|}{\sqrt{\langle h_e,h_e\rangle\langle h_c,h_c\rangle}},\quad mismatch=1-FF$",
        "",
        r"$R_N=\sqrt{\langle h_e-h_c,h_e-h_c\rangle/\langle h_e,h_e\rangle}$",
        "",
    ]
    if e001 and e01 and e03:
        lines.extend(
            [
                "Representative convention-floor-removed best AE residuals:",
                f"e=0.01: {float(e001['convention_removed_best']):.3e}",
                f"e=0.10: {float(e01['convention_removed_best']):.3e}",
                f"e=0.30: {float(e03['convention_removed_best']):.3e}",
                "",
            ]
        )
    lines.append(
        "Interpretation: the eccentricity trend is cleanest after removing the e=0 convention floor. "
        "The small-e trilinear harmonic path is useful for fast experiments, but the code still treats it as experimental and recommends brute-force eccentric links once e >= 0.1."
    )
    return "\n".join(lines)


def add_image(ax, path: Path) -> None:
    ax.imshow(plt.imread(path))
    ax.axis("off")


def write_pdf(path: Path, circular_plot: Path, residual_plot: Path, summary_rows: list[dict[str, str]]) -> None:
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0])
        wrapped_text(fig.add_subplot(gs[0]), "Short Trilinear Response Implementation Summary\n" + equations_page_text(), size=9.5)
        add_image(fig.add_subplot(gs[1]), circular_plot)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25])
        wrapped_text(fig.add_subplot(gs[0]), "JAXGBaccurate comparison\n" + comparison_page_text(summary_rows), size=9.7)
        add_image(fig.add_subplot(gs[1]), residual_plot)
        pdf.savefig(fig)
        plt.close(fig)


def write_markdown(path: Path, circular_plot: Path, residual_plot: Path, summary_rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Short Trilinear Response Implementation Summary",
                "",
                f"Generated: {REPORT_DATE}. Repo API spelling is `triliear`; prose uses `trilinear`.",
                "",
                "## Implementation",
                "",
                "The code builds a low-frequency trilinear LISA response tensor `A_cjl(t)` for each pair channel, contracts it with five trace-free polarization coefficients and the affine source direction, then optionally routes beta-expanded six-link responses through pyTDI to get Michelson `X/Y/Z`. The eccentric trial decomposes one radial period into harmonic carriers and evaluates those carriers with the same response machinery.",
                "",
                "## Equations",
                "",
                r"$$f_{eff}(t)=f_0+\dot f t,\quad \beta(t)=-2\pi i f_{eff}(t)L(t)$$",
                "",
                r"$$y_{sr}(t)=\frac{1}{2}\frac{n^TEn}{1-n\cdot k}[h(t-L-p_s\cdot k)-h(t-p_r\cdot k)]$$",
                "",
                r"$$W_c(t)=F(t)\sum_{j=1}^{5}\sum_{l=0}^{3}A_{cjl}(t)e_j\tilde{k}_l,\quad \tilde{k}=(1,k_x,k_y,k_z)$$",
                "",
                r"$$a_{pair}=2\beta(1+\beta+\frac{2}{3}\beta^2),\quad a_{comp}=-\beta^2\frac{r_m}{L}(1+\beta)$$",
                "",
                r"$$h_+(t),h_\times(t)\rightarrow \hat h_{+,m},\hat h_{\times,m},\quad f_m=\frac{mn}{2\pi}$$",
                "",
                "![Circular timing](circular_timing.png)",
                "",
                "## JAXGBaccurate Comparison",
                "",
                comparison_page_text(summary_rows).replace("\n", "\n\n"),
                "",
                "![JAXGBaccurate residuals](jaxgbaccurate_residuals.png)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    detailed_dir = root / "reports" / "trilinear_trials_detailed"
    out_dir = root / "reports" / "trilinear_implementation_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_circular_plot = detailed_dir / "circular_timing.png"
    source_residual_plot = detailed_dir / "jaxgbaccurate_residuals.png"
    summary_csv = detailed_dir / "jaxgbaccurate_residual_summary.csv"
    missing = [str(path) for path in (source_circular_plot, source_residual_plot, summary_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing inputs; run benchmarks/trilinear_trials_detailed_report.py first: " + ", ".join(missing))

    circular_plot = out_dir / "circular_timing.png"
    residual_plot = out_dir / "jaxgbaccurate_residuals.png"
    shutil.copyfile(source_circular_plot, circular_plot)
    shutil.copyfile(source_residual_plot, residual_plot)
    summary_rows = read_rows(summary_csv)

    pdf_path = out_dir / "short_trilinear_implementation_summary.pdf"
    md_path = out_dir / "short_trilinear_implementation_summary.md"
    write_pdf(pdf_path, circular_plot, residual_plot, summary_rows)
    write_markdown(md_path, circular_plot, residual_plot, summary_rows)

    print(f"wrote {pdf_path}")
    print(f"wrote {md_path}")
    print(f"wrote {circular_plot}")
    print(f"wrote {residual_plot}")


if __name__ == "__main__":
    main()
