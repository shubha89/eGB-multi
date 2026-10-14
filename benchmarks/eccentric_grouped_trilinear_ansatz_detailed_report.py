"""Generate a detailed report for the eccentric grouped-trilinear ansatz study.

This script uses the already-measured population benchmark CSV and adds a small
one-source harmonic-cutoff diagnostic.  It writes plots, a LaTeX report, and a
PDF-ready source file under:

    reports/eccentric_grouped_trilinear_ansatz_study/

Run with:

    python benchmarks/eccentric_grouped_trilinear_ansatz_detailed_report.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eccentric_grouped_trilinear_ansatz_study import (
    combined_xyz_residual,
    source_harmonic_residual,
    xyz_from_links,
)

from egb_multi.constants import KILOPARSEC_M
from egb_multi.eccentric import EccentricBinaryParams, eccentric_harmonic_indices
from egb_multi.eccentric_jax import eccentric_harmonic_links_grouped_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.external import default_lisaorbits, state_from_lisaorbits


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def i(row: dict[str, str], key: str) -> int:
    return int(row[key])


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def grouped_by_eccentricity(rows: list[dict[str, str]]) -> list[tuple[float, list[dict[str, str]]]]:
    eccentricities = sorted({f(row, "eccentricity") for row in rows})
    return [(eccentricity, [row for row in rows if f(row, "eccentricity") == eccentricity]) for eccentricity in eccentricities]


def summary_by_eccentricity(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for eccentricity, subset in grouped_by_eccentricity(rows):
        out.append(
            {
                "eccentricity": eccentricity,
                "n_max_harmonic": i(subset[0], "n_max_harmonic"),
                "positive_harmonics": subset[0]["positive_harmonics"],
                "source_residual": f(subset[0], "source_harmonic_scaled_residual"),
                "median_exact_residual": median([f(row, "exact_transfer_scaled_real_residual") for row in subset]),
                "median_exact_speedup": median([f(row, "exact_transfer_speedup_vs_brute") for row in subset]),
            }
        )
    return out


def rows_at_source_count(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    return sorted([row for row in rows if i(row, "sources") == count], key=lambda row: f(row, "eccentricity"))


def make_summary_plot(rows: list[dict[str, str]], path: Path) -> None:
    largest_count = max(i(row, "sources") for row in rows)
    largest = rows_at_source_count(rows, largest_count)
    x = np.array([f(row, "eccentricity") for row in largest])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    axes[0, 0].plot(x, [f(row, "exact_transfer_scaled_real_residual") for row in largest], marker="^", label="grouped exact-transfer")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(f"accuracy at N={largest_count}")
    axes[0, 0].set_ylabel("scaled Re(XYZ) residual")

    axes[0, 1].plot(x, [f(row, "exact_transfer_speedup_vs_brute") for row in largest], marker="^", label="grouped exact-transfer")
    axes[0, 1].axhline(1.0, color="0.4", linewidth=1.0, linestyle="--")
    axes[0, 1].set_title(f"speedup at N={largest_count}")
    axes[0, 1].set_ylabel("brute wall time / model wall time")

    axes[1, 0].plot(x, [f(row, "brute_force_wall_s") for row in largest], marker="o", label="brute force")
    axes[1, 0].plot(x, [f(row, "trilinear_exact_transfer_wall_s") for row in largest], marker="^", label="grouped exact-transfer")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(f"wall time at N={largest_count}")
    axes[1, 0].set_ylabel("wall time [s]")

    ax_harmonic = axes[1, 1]
    ax_source = ax_harmonic.twinx()
    ax_harmonic.plot(x, [i(row, "n_max_harmonic") for row in largest], marker="o", color="tab:blue", label="n_max")
    ax_source.plot(x, [f(row, "source_harmonic_scaled_residual") for row in largest], marker="s", color="tab:orange", label="source residual")
    ax_harmonic.set_yscale("log")
    ax_source.set_yscale("log")
    ax_harmonic.set_title("harmonic ansatz")
    ax_harmonic.set_ylabel("largest positive harmonic")
    ax_source.set_ylabel("source harmonic residual")
    ax_harmonic.legend(fontsize=8, loc="upper left")
    ax_source.legend(fontsize=8, loc="lower right")

    for ax in axes.flat:
        ax.set_xlabel("eccentricity")
        ax.grid(True, which="both", alpha=0.3)
        if ax is not axes[1, 1]:
            ax.legend(fontsize=8)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_exact_grid_plot(rows: list[dict[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    eccentricities = sorted({f(row, "eccentricity") for row in rows})
    counts = sorted({i(row, "sources") for row in rows})
    colors = plt.cm.viridis(np.linspace(0.0, 0.88, len(eccentricities)))

    for color, eccentricity in zip(colors, eccentricities, strict=True):
        subset = sorted([row for row in rows if f(row, "eccentricity") == eccentricity], key=lambda row: i(row, "sources"))
        axes[0, 0].plot(
            [i(row, "sources") for row in subset],
            [f(row, "exact_transfer_speedup_vs_brute") for row in subset],
            marker="o",
            color=color,
            label=f"e={eccentricity:g}",
        )

    for count in counts:
        subset = sorted([row for row in rows if i(row, "sources") == count], key=lambda row: f(row, "eccentricity"))
        axes[0, 1].plot(
            [f(row, "eccentricity") for row in subset],
            [f(row, "exact_transfer_scaled_real_residual") for row in subset],
            marker="s",
            label=f"N={count}",
        )

    largest_count = max(counts)
    largest = rows_at_source_count(rows, largest_count)
    x = np.array([f(row, "eccentricity") for row in largest])
    axes[1, 0].plot(x, [f(row, "brute_force_wall_s") for row in largest], marker="o", label="brute force")
    axes[1, 0].plot(x, [f(row, "trilinear_exact_transfer_wall_s") for row in largest], marker="^", label="grouped exact-transfer")

    ax_harmonic = axes[1, 1]
    ax_source = ax_harmonic.twinx()
    ax_harmonic.plot(x, [i(row, "n_max_harmonic") for row in largest], marker="o", color="tab:blue", label="n_max")
    ax_source.plot(x, [f(row, "source_harmonic_scaled_residual") for row in largest], marker="s", color="tab:orange", label="source residual")

    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].axhline(1.0, color="0.4", linewidth=1.0, linestyle="--")
    axes[0, 0].set_title("exact-transfer speedup")
    axes[0, 0].set_xlabel("number of sources")
    axes[0, 0].set_ylabel("brute wall time / model wall time")

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("exact-transfer accuracy")
    axes[0, 1].set_xlabel("eccentricity")
    axes[0, 1].set_ylabel("scaled Re(XYZ) residual")

    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(f"wall time at N={largest_count}")
    axes[1, 0].set_xlabel("eccentricity")
    axes[1, 0].set_ylabel("wall time [s]")

    ax_harmonic.set_yscale("log")
    ax_source.set_yscale("log")
    ax_harmonic.set_title("harmonic ansatz")
    ax_harmonic.set_xlabel("eccentricity")
    ax_harmonic.set_ylabel("largest positive harmonic")
    ax_source.set_ylabel("source harmonic residual")

    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    ax_source.legend(fontsize=8, loc="lower right")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_cutoff_diagnostic(csv_path: Path, png_path: Path) -> list[dict[str, object]]:
    duration = 86400.0
    samples = 2048
    harmonic_samples = 2048
    times = np.linspace(0.0, duration, samples, endpoint=False)
    state = state_from_lisaorbits(default_lisaorbits("esa-trailing"), times)
    geometry = precompute_jax_link_geometry(state)
    base = {
        "mean_motion": np.pi * 1.0e-3,
        "m1_solar": 0.6,
        "m2_solar": 0.4,
        "distance_m": KILOPARSEC_M,
        "beta": 0.3,
        "lambda_": -0.7,
        "psi": 0.4,
        "inclination": 1.0,
        "phi0": 0.2,
        "fdot": 0.0,
    }
    eccentricities = [0.05, 0.1, 0.2, 0.3]
    nmax_values = [2, 3, 4, 6, 9, 12, 16, 20, 26, 32]
    rows: list[dict[str, object]] = []

    for eccentricity in eccentricities:
        source = EccentricBinaryParams(**base, eccentricity=eccentricity)
        brute_xyz = xyz_from_links(
            state,
            eccentric_links_jax(source, geometry, batch_size=1, physics_mode="1pn"),
        )
        for nmax in nmax_values:
            order = 0 if nmax == 2 else nmax - 2
            harmonics = eccentric_harmonic_indices(order)
            if max(harmonics) != nmax:
                continue
            links = eccentric_harmonic_links_grouped_jax(
                source,
                geometry,
                eccentric_order=order,
                harmonic_samples=harmonic_samples,
                batch_size=1,
                response_mode="fastgb_transfer",
            )
            xyz = xyz_from_links(state, links)
            rows.append(
                {
                    "eccentricity": eccentricity,
                    "n_max_harmonic": nmax,
                    "eccentric_order": order,
                    "positive_harmonic_count": len(harmonics),
                    "positive_harmonics": " ".join(str(harmonic) for harmonic in harmonics),
                    "source_harmonic_scaled_residual": source_harmonic_residual(source, harmonics, samples=harmonic_samples),
                    "exact_transfer_scaled_real_residual": combined_xyz_residual(xyz, brute_xyz, scaled=True),
                }
            )

    with csv_path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for eccentricity in eccentricities:
        subset = [row for row in rows if float(row["eccentricity"]) == eccentricity]
        x = [int(row["n_max_harmonic"]) for row in subset]
        axes[0].plot(x, [float(row["source_harmonic_scaled_residual"]) for row in subset], marker="o", label=f"e={eccentricity:g}")
        axes[1].plot(x, [float(row["exact_transfer_scaled_real_residual"]) for row in subset], marker="s", label=f"e={eccentricity:g}")
    axes[0].set_title("source-only truncation")
    axes[1].set_title("full Re(XYZ) residual")
    for ax in axes:
        ax.set_xlabel("largest positive harmonic")
        ax.set_ylabel("scaled residual")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return rows


def latex_eccentric_summary_table(summary: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$e$ & $n_{\max}$ & source err. & exact err. & exact speedup \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{float(row['eccentricity']):.2f} & "
            f"{int(row['n_max_harmonic'])} & "
            f"{float(row['source_residual']):.2e} & "
            f"{float(row['median_exact_residual']):.2e} & "
            f"{float(row['median_exact_speedup']):.2f}x \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_largest_count_table(rows: list[dict[str, str]]) -> str:
    largest_count = max(i(row, "sources") for row in rows)
    subset = rows_at_source_count(rows, largest_count)
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$e$ & $n_{\max}$ & brute [s] & exact [s] & exact err. & speedup \\",
        r"\midrule",
    ]
    for row in subset:
        lines.append(
            f"{f(row, 'eccentricity'):.2f} & "
            f"{i(row, 'n_max_harmonic')} & "
            f"{f(row, 'brute_force_wall_s'):.4f} & "
            f"{f(row, 'trilinear_exact_transfer_wall_s'):.4f} & "
            f"{f(row, 'exact_transfer_scaled_real_residual'):.2e} & "
            f"{f(row, 'exact_transfer_speedup_vs_brute'):.2f}x \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_cutoff_table(cutoff_rows: list[dict[str, object]], eccentricity: float = 0.3) -> str:
    keep = [row for row in cutoff_rows if float(row["eccentricity"]) == eccentricity and int(row["n_max_harmonic"]) in {4, 9, 12, 16, 32}]
    lines = [
        r"\begin{tabular}{rrr}",
        r"\toprule",
        r"$n_{\max}$ & source err. & full Re(XYZ) err. \\",
        r"\midrule",
    ]
    for row in keep:
        lines.append(
            f"{int(row['n_max_harmonic'])} & "
            f"{float(row['source_harmonic_scaled_residual']):.2e} & "
            f"{float(row['exact_transfer_scaled_real_residual']):.2e} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_tex(
    rows: list[dict[str, str]],
    cutoff_rows: list[dict[str, object]],
    report_dir: Path,
    summary_plot: Path,
    exact_grid_plot: Path,
    cutoff_plot: Path,
) -> Path:
    summary = summary_by_eccentricity(rows)
    largest_count = max(i(row, "sources") for row in rows)
    med_exact_residual = median([f(row, "exact_transfer_scaled_real_residual") for row in rows])
    med_speedup = median([f(row, "exact_transfer_speedup_vs_brute") for row in rows])

    tex = "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=0.8in]{geometry}",
            r"\usepackage{amsmath}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{array}",
            r"\title{Eccentric Grouped Trilinear Response with a Harmonic Ansatz}",
            r"\author{}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            r"\section*{Purpose}",
            r"This note explains the eccentric grouped trilinear experiment in plain terms. The goal is to test whether we can replace the expensive brute-force eccentric LISA response by a faster harmonic response while keeping the waveform close to the brute-force answer.",
            r"",
            r"The test is deliberately local and practical: fixed eccentricity, one day of data, 2048 samples, the ESA trailing LISA orbit, and source populations with 32, 128, 512, and 2048 binaries. The brute-force JAX eccentric links followed by pyTDI are used as the reference.",
            r"",
            r"The main question is: can the eccentric signal be split into pieces that are cheap to reuse? For a circular binary there is essentially one strong harmonic. For an eccentric binary there are several harmonics. The approach here keeps the same trilinear idea, but applies it separately to each harmonic and then adds the harmonic responses together.",
            r"\section*{Basic Objects}",
            r"The source produces two gravitational-wave polarizations, $h_+(t)$ and $h_\times(t)$. The LISA response does not measure these directly. Each one-way laser link sees a projected and delayed version of the strain. A link is labelled by $\ell$ and has a unit arm direction $\hat n_\ell(t)$, an arm length $L_\ell(t)$, and a source direction $\hat k$. These are LISA-geometry quantities. Once the orbit and sky position are fixed, they are known functions of time.",
            r"",
            r"The strain tensor is",
            r"\begin{equation}",
            r"H(t)=h_+(t)e^+ + h_\times(t)e^\times,",
            r"\end{equation}",
            r"where $e^+$ and $e^\times$ are the two polarization tensors. The link only sees the contraction $\hat n_\ell^T H\hat n_\ell$, which is a weighted sum of $h_+$ and $h_\times$.",
            r"\section*{What Is Being Compared}",
            r"\paragraph{Brute force.} For each source and each one-way LISA link, the brute-force calculation evaluates the eccentric waveform at the two retarded times needed by that link. In simplified notation, the link signal has the form",
            r"\begin{equation}",
            r"y_\ell(t) = \frac{1}{2(1-\hat n_\ell\cdot \hat k)}\left[\hat n_\ell^T H(t_{\rm emit})\hat n_\ell-\hat n_\ell^T H(t_{\rm recv})\hat n_\ell\right],",
            r"\end{equation}",
            r"where $t_{\rm emit}$ and $t_{\rm recv}$ differ by the light travel time along the arm. This is accurate but expensive because the source waveform is evaluated at many retarded times for every source and link.",
            r"",
            r"\paragraph{Grouped trilinear harmonic response.} The grouped trilinear method first writes the source waveform as a sum of harmonic carriers. For a fixed eccentric source, the carrier labels are interpreted using the same convention as the package source model:",
            r"\begin{equation}",
            r"\omega_m = 2\Omega_\phi + (m-2)n_r,",
            r"\end{equation}",
            r"where $\Omega_\phi$ is the averaged azimuthal frequency and $n_r$ is the radial frequency. Label $m=2$ is the circular quadrupole carrier. Labels above and below 2 are radial sidebands. When periastron advance is disabled, $n_r=\Omega_\phi$ and this reduces to the old integer-harmonic rule $\omega_m=m n_r$.",
            r"",
            r"The polarizations are approximated as",
            r"\begin{equation}",
            r"h_+(t) \simeq \sum_{m=1}^{n_{\max}}\left(c^+_m e^{i\omega_m t}+\overline{c^+_m}e^{-i\omega_m t}\right),\qquad",
            r"h_\times(t) \simeq \sum_{m=1}^{n_{\max}}\left(c^\times_m e^{i\omega_m t}+\overline{c^\times_m}e^{-i\omega_m t}\right).",
            r"\end{equation}",
            r"The coefficients $c^+_m$ and $c^\times_m$ are the source part. They are obtained by sampling one radial period and projecting the waveform onto the carrier frequencies $\omega_m$.",
            r"",
            r"For one positive harmonic, the link response can be written schematically as",
            r"\begin{equation}",
            r"y_{\ell m}(t)\simeq {\rm Re}\left\{e^{i\omega_m t}\left[c^+_m F^+_\ell(t)+c^\times_m F^\times_\ell(t)\right]T_{\ell m}(t)\right\}.",
            r"\end{equation}",
            r"Here $F^+_\ell$ and $F^\times_\ell$ are the sky and polarization projection factors, and $T_{\ell m}$ is the finite-arm transfer factor. This is the trilinear structure: a source coefficient, a geometry projection, and a transfer factor are multiplied together. For an eccentric source the model is a sum of these trilinear pieces, one for each retained harmonic:",
            r"\begin{equation}",
            r"y_\ell(t)\simeq \sum_{m=1}^{n_{\max}} y_{\ell m}(t).",
            r"\end{equation}",
            r"Therefore the eccentric extension is not a new single circular-like term. It is a harmonic sum of circular-like trilinear terms. The source variables that change are the circular source coefficients becoming a list of harmonic coefficients, $\{c^+_m,c^\times_m\}_{m=1}^{n_{\max}}$.",
            r"",
            r"The LISA geometry and sky projection are computed once and reused across sources, links, and harmonics. This is what ``grouped'' means in this implementation.",
            r"",
            r"\paragraph{Exact finite-arm transfer.} The implementation tested in this report keeps the finite-arm transfer factor for each harmonic,",
            r"\begin{equation}",
            r"T_m(t) = -i\,\omega_m L(t)\,{\rm sinc}\!\left[\frac{\omega_m L(t)(1-\hat n\cdot\hat k)}{2}\right]",
            r"\exp\!\left[-i\frac{\omega_m L(t)(1-\hat n\cdot\hat k)}{2}\right],",
            r"\end{equation}",
            r"with angular frequency $\omega_m=2\Omega_\phi+(m-2)n_r$. Keeping this factor matters because eccentric sources distribute power over several harmonics, and each harmonic has a different effective frequency.",
            r"\section*{Harmonic Ansatz}",
            r"The harmonic cutoff is chosen from eccentricity only:",
            r"\begin{equation}",
            r"n_{\max}(e)=2\quad(e=0),\qquad",
            r"n_{\max}(e)=\left\lceil 2.5+25e+20e^2\right\rceil\quad(e>0).",
            r"\end{equation}",
            r"The implementation uses the historical package variable ``eccentric order'', where order $p$ means harmonics $1,\ldots,p+2$. Therefore $p=n_{\max}-2$ for $e>0$.",
            r"",
            r"The source-only harmonic residual checks whether the chosen harmonic list can reconstruct the source polarizations before LISA is applied:",
            r"\begin{equation}",
            r"\epsilon_{\rm src}=\left(\frac{\|h_+-h_{+,{\rm harm}}\|_2^2+\|h_\times-h_{\times,{\rm harm}}\|_2^2}{\|h_+\|_2^2+\|h_\times\|_2^2}\right)^{1/2}.",
            r"\end{equation}",
            r"",
            r"\begin{center}",
            latex_eccentric_summary_table(summary),
            r"\end{center}",
            r"\section*{Harmonic Basis, Sideband Basis, And A Hybrid Option}",
            r"The current harmonic ansatz is best understood as the fast branch of a larger sideband expansion.  With periastron advance, a fixed eccentric binary has two orbital frequencies: the azimuthal frequency $\Omega_\phi$ and the radial frequency $n_r$.  The grouped harmonic model keeps carriers",
            r"\begin{equation}",
            r"\omega_m = 2\Omega_\phi + (m-2)n_r.",
            r"\end{equation}",
            r"Equivalently, if $q=m-2$, this is the positive quadrupole sideband family",
            r"\begin{equation}",
            r"\omega_{+,q}=2\Omega_\phi+q n_r .",
            r"\end{equation}",
            r"The harmonic model therefore uses",
            r"\begin{equation}",
            r"h_A^{\rm harm}(t)=",
            r"\sum_{q\in Q_H}\left(c^A_{+,q}e^{i\omega_{+,q}t}",
            r"+\overline{c^A_{+,q}}e^{-i\omega_{+,q}t}\right),",
            r"\qquad A\in\{+,\times\}.",
            r"\end{equation}",
            r"Here $Q_H$ is the chosen finite set of retained harmonics.  This is fast because the response sees one simple list of carriers per source, and the grouped implementation computes the LISA projection factors once per source and reuses them for every retained carrier.",
            r"",
            r"The explicit sideband basis is more complete.  It allows three carrier families:",
            r"\begin{align}",
            r"\omega_{+,q} &=  2\Omega_\phi+q n_r,\\",
            r"\omega_{-,q} &= -2\Omega_\phi+q n_r,\\",
            r"\omega_{0,q} &= q n_r .",
            r"\end{align}",
            r"The first family is the one already used by the harmonic ansatz.  The second family accounts for the fact that, once there are two independent orbital frequencies, the conjugate quadrupole branch is not the same as simply using negative frequencies of $\omega_{+,q}$.  The third family captures radial-only terms in the eccentric source amplitudes.  In this basis,",
            r"\begin{equation}",
            r"h_A^{\rm sb}(t)=",
            r"\Re\left[",
            r"\sum_{q\in Q_+} c^A_{+,q}e^{i\omega_{+,q}t}",
            r"+\sum_{q\in Q_-} c^A_{-,q}e^{i\omega_{-,q}t}",
            r"+\sum_{q\in Q_0} c^A_{0,q}e^{i\omega_{0,q}t}",
            r"\right].",
            r"\end{equation}",
            r"The sideband coefficients are obtained by a real least-squares fit over several radial cycles,",
            r"\begin{equation}",
            r"\{c^A_b\}=",
            r"\arg\min_{\{c_b\}}",
            r"\left\|h_A(t_j)-\Re\sum_b c_b e^{i\omega_b t_j}\right\|_2^2",
            r"+\lambda\sum_b |c_b|^2,",
            r"\end{equation}",
            r"where $b$ labels the retained sideband candidates.  This is more accurate at moderate and high eccentricity because it can represent source terms that do not live only on the positive quadrupole branch.  It is slower because there are more carriers and because the least-squares fit is heavier than the direct harmonic projection.",
            r"",
            r"The difference is therefore not that the two methods use different LISA responses.  Both remain linear sums of monochromatic trilinear response terms,",
            r"\begin{equation}",
            r"y_\ell(t)=\sum_b y_{\ell b}(t),",
            r"\end{equation}",
            r"with known LISA geometry and source coefficients multiplying each carrier response.  The difference is the source basis.  The harmonic approach is a sparse, fast subset of the sideband approach; the sideband approach is a fuller, better-conditioned representation of the eccentric source at the cost of more coefficients.",
            r"",
            r"A useful next model is a hybrid basis.  The idea is to keep the fast harmonic branch as the backbone and add only the sideband terms that explain the remaining source residual:",
            r"\begin{equation}",
            r"r_A(t)=h_A(t)-h_A^{\rm harm}(t),",
            r"\end{equation}",
            r"\begin{equation}",
            r"d^A_b=",
            r"\arg\min_{\{d_b\}}",
            r"\left\|r_A(t_j)-\Re\sum_{b\in C}d^A_b e^{i\omega_b t_j}\right\|_2^2",
            r"+\lambda\sum_{b\in C}|d^A_b|^2 .",
            r"\end{equation}",
            r"The hybrid waveform is then",
            r"\begin{equation}",
            r"h_A^{\rm hyb}(t)=h_A^{\rm harm}(t)+",
            r"\Re\sum_{b\in C_{\rm keep}}d^A_b e^{i\omega_b t}.",
            r"\end{equation}",
            r"The correction candidates $C$ should come mostly from the missing $\omega_{-,q}$ and $\omega_{0,q}$ families, plus any high-$q$ positive-branch terms not included in $Q_H$.  The kept set $C_{\rm keep}$ can be chosen by coefficient power,",
            r"\begin{equation}",
            r"P_b=|d^+_b|^2+|d^\times_b|^2,",
            r"\end{equation}",
            r"for example by sorting $P_b$ and retaining the smallest set whose discarded power fraction is below a tolerance.  This would preserve most of the harmonic speed for low eccentricity, while recovering the sideband accuracy where the missing radial and negative-quadrupole branches matter.",
            r"",
            r"For implementation, the important point is that the hybrid basis is still a sum of trilinear carrier responses.  It can use the same grouped JAX response kernel as the sideband model, but with many fewer correction carriers.  The coefficient fit should also be cached or separated from the response call; otherwise the least-squares solve can dominate the runtime and hide the speed benefit.",
            r"\section*{Procedure Used In The Benchmark}",
            r"\begin{enumerate}",
            r"\item Build the LISA orbit on the one-day time grid. This gives arm directions, arm lengths, and spacecraft positions at every sample.",
            r"\item Precompute the JAX link geometry from that orbit. This keeps the benchmark focused on waveform and response cost rather than repeatedly rebuilding the same LISA arrays.",
            r"\item Draw a population of binaries for each source count and eccentricity. The sky positions, orientations, masses, and frequencies are randomized with a fixed seed so the comparison is repeatable.",
            r"\item Compute the brute-force reference links. This evaluates the eccentric waveform directly at the retarded times needed by each one-way link.",
            r"\item Convert the six one-way links into the first-generation Michelson TDI channels $X,Y,Z$ using pyTDI.",
            r"\item Compute the grouped trilinear links with the exact finite-arm transfer factor.",
            r"\item Convert the grouped link set into $X,Y,Z$ with the same pyTDI call.",
            r"\item Compare wall time and residuals against the brute-force $X,Y,Z$ reference.",
            r"\end{enumerate}",
            r"\section*{Accuracy Metric}",
            r"We compare the real time-domain TDI channels $X,Y,Z$. Before measuring the residual, each channel is allowed one real scale/sign factor. This prevents a harmless global sign or amplitude convention from dominating the number:",
            r"\begin{equation}",
            r"\epsilon_{\rm XYZ} = \left(\frac{\sum_c \|a_c\,{\rm Re}\,h_{{\rm model},c}-{\rm Re}\,h_{{\rm brute},c}\|_2^2}{\sum_c \|{\rm Re}\,h_{{\rm brute},c}\|_2^2}\right)^{1/2},",
            r"\qquad",
            r"a_c=\frac{\langle{\rm Re}\,h_{{\rm model},c},{\rm Re}\,h_{{\rm brute},c}\rangle}{\langle{\rm Re}\,h_{{\rm model},c},{\rm Re}\,h_{{\rm model},c}\rangle}.",
            r"\end{equation}",
            r"\section*{Population Results}",
            rf"The most useful large-population summary is the $N={largest_count}$ row:",
            r"\begin{center}",
            latex_largest_count_table(rows),
            r"\end{center}",
            r"",
            rf"Across the full grid, the median exact-transfer residual is {med_exact_residual:.3e}, and the median exact-transfer speedup over brute force is {med_speedup:.2f}x.",
            r"",
            r"\begin{figure}[h]",
            r"\centering",
            rf"\includegraphics[width=\linewidth]{{{summary_plot.name}}}",
            r"\caption{Large-population summary for grouped exact-transfer. The speedup decreases as eccentricity grows because more harmonics are included.}",
            r"\end{figure}",
            r"",
            r"\begin{figure}[h]",
            r"\centering",
            rf"\includegraphics[width=\linewidth]{{{exact_grid_plot.name}}}",
            r"\caption{Exact-transfer benchmark view: speedup versus source count, residual versus eccentricity, wall time at $N=2048$, and the harmonic ansatz.}",
            r"\end{figure}",
            r"\section*{Why The Eccentric Residual Grows}",
            r"The residual increase is partly due to missing harmonics, but not only. A one-source diagnostic was run by increasing $n_{\max}$ beyond the ansatz and comparing both the source-only harmonic reconstruction and the full LISA response.",
            r"\begin{center}",
            latex_cutoff_table(cutoff_rows, eccentricity=0.3),
            r"\end{center}",
            r"",
            r"For $e=0.3$, increasing $n_{\max}$ from 4 to 12 reduces the source-only error from order $10^{-1}$ to order $10^{-4}$, and the full response improves strongly. With the corrected sideband-frequency convention, increasing to $n_{\max}=16$ reduces the full response residual to the few-$10^{-4}$ level. This shows that the earlier percent-level high-e floor was largely a frequency-convention problem.",
            r"",
            r"Past $n_{\max}=16$, the full response does not keep improving monotonically even though more sidebands are retained. This points to a smaller remaining issue in the coefficient projection and finite-sample conditioning, not to the original one-frequency convention. The next improvement should therefore focus on an adaptive and better-conditioned sideband fit, rather than simply adding many more harmonics.",
            r"\begin{figure}[h]",
            r"\centering",
            rf"\includegraphics[width=\linewidth]{{{cutoff_plot.name}}}",
            r"\caption{One-source harmonic cutoff diagnostic. Harmonic truncation matters at low cutoff, but the full response eventually reaches a floor.}",
            r"\end{figure}",
            r"\section*{Interpretation}",
            r"For circular sources, grouped exact-transfer matches brute force to numerical precision and is several times faster. For small eccentricities, the method remains useful: at $e=0.01$--$0.05$ the exact-transfer residual is roughly $9\times10^{-4}$ to $4\times10^{-3}$ at $N=2048$, with about a factor of two speedup. At $e=0.1$, the residual is about $1.1\times10^{-3}$ and the speedup is about $1.5$x.",
            r"",
            r"At larger eccentricities, the accuracy remains good after the frequency-convention fix: the residual is about $10^{-3}$ at $e=0.2$ and $1.4\times10^{-3}$ at $e=0.3$ for the ansatz cutoff. The limitation is now timing. At $e=0.2$ the method is close to break-even, and at $e=0.3$ it is slower than brute force because twelve positive harmonics are being processed.",
            r"\section*{What To Do Next}",
            r"\begin{enumerate}",
            r"\item Validate the corrected sideband convention over Newtonian, 1PN without periastron, and full 1PN settings on the same source grid.",
            r"\item Improve the coefficient projection. A small least-squares fit over several radial cycles should be better conditioned than a direct one-period projection, especially when many weak sidebands are included.",
            r"\item Make the harmonic cutoff adaptive. Instead of using eccentricity alone, choose $n_{\max}$ from the measured source harmonic residual and stop before weak sidebands add numerical noise.",
            r"\item Implement a hybrid basis. Use the grouped harmonic model as the backbone, fit the residual against explicit sideband candidates, and keep only the strongest correction carriers.",
            r"\item Cache or precompute fitted basis coefficients separately from the LISA response. This is especially important for the sideband and hybrid paths because repeated least-squares fitting can dominate the runtime.",
            r"\item Group sources by harmonic tier. Sources with small eccentricity should not pay for the harmonic count needed by high-e sources.",
            r"\item Move the TDI postprocessing into JAX for this benchmark path. The harmonic links are already vectorized; pyTDI is now one of the serial pieces.",
            r"\item Repeat the production comparison over longer durations and more frequencies. The one-day study is good for engineering direction, but not a final science validation.",
            r"\end{enumerate}",
            r"\section*{Bottom Line}",
            r"Using the package frequency convention in the harmonic basis is the right fix: the grouped exact-transfer residuals drop to about $10^{-3}$ across the tested eccentric grid while circular sources remain at numerical precision. The sideband basis explains what is missing at higher eccentricity, but it is currently slower. The most promising next step is a hybrid basis: keep the fast harmonic branch and add only a sparse set of sideband corrections.",
            r"\end{document}",
            "",
        ]
    )
    tex_path = report_dir / "eccentric_grouped_trilinear_ansatz_detailed_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports" / "eccentric_grouped_trilinear_ansatz_study"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = root / "benchmarks" / "eccentric_grouped_trilinear_ansatz_study.csv"
    rows = read_rows(csv_path)

    summary_plot = report_dir / "eccentric_grouped_trilinear_ansatz_summary.png"
    exact_grid_plot = report_dir / "eccentric_grouped_trilinear_exact_transfer_grid.png"
    cutoff_csv = report_dir / "eccentric_harmonic_cutoff_diagnostic.csv"
    cutoff_plot = report_dir / "eccentric_harmonic_cutoff_diagnostic.png"

    make_summary_plot(rows, summary_plot)
    make_exact_grid_plot(rows, exact_grid_plot)
    cutoff_rows = run_cutoff_diagnostic(cutoff_csv, cutoff_plot)
    tex_path = write_tex(rows, cutoff_rows, report_dir, summary_plot, exact_grid_plot, cutoff_plot)

    print(f"wrote {summary_plot}")
    print(f"wrote {exact_grid_plot}")
    print(f"wrote {cutoff_csv}")
    print(f"wrote {cutoff_plot}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
