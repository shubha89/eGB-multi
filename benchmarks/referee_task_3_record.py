"""Preserve Tasks 3.1-3.4 evidence without waveform or inference runs.

--prepare extracts archived PDFs, indexes wording and evaluates published/scalar
formulas. The default also checks the saved reports and records their hashes.
No waveform package, optimizer, likelihood sampler or simulation is executed.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/referee/task_3_20261012"
TEX = ROOT / "reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex"
FROZEN = ROOT / "outputs/referee/task_2_8_20261012_completion/frozen_production_configuration.json"
PAPERS = [
    dict(id="2310.06568v2", file="Moore_2024_2310.06568v2",
         doi="10.1093/mnras/stae1288", sections="3; 4.1, Fig. 3 and Eq. (21); Appendices A-B",
         markers=["median to the standard deviation", "scaled to the desired mission duration", "1.54", "K = 2.3"],
         role="Direct eccentric Galactic-binary measurability comparison"),
    dict(id="0809.3844v2", file="Lindblom_2008_0809.3844v2",
         doi="10.1103/PhysRevD.78.124020", sections="II.A, Eqs. (3)-(9); II.B, Eq. (10)",
         markers=["Accuracy Requirements for Measurement", "standard deviation"],
         role="Noise-norm waveform accuracy and distinction from SNR loss"),
    dict(id="1207.4759v2", file="Vallisneri_2012_1207.4759v2",
         doi="10.1103/PhysRevD.86.082001", sections="II, Eqs. (8), (11)-(14); III",
         markers=["local tangent space", "Occam"],
         role="High-SNR model-comparison geometry; contextual, not an eccentricity calibration"),
    dict(id="1211.0546v1", file="Baird_2013_1211.0546v1",
         doi="10.1103/PhysRevD.87.024035", sections="V.A, Eqs. (12), (16)-(19)",
         markers=["confidence regions", "remaining parameter space"],
         role="Parameter-count and confidence-level coefficient under Fisher assumptions"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def normalized(value):
    value = unicodedata.normalize("NFKD", value).casefold()
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def extract(pdf):
    assert pdf.read_bytes().startswith(b"%PDF"), pdf
    layout, plain = pdf.with_suffix(".txt"), pdf.with_suffix(".plain.txt")
    subprocess.run(["pdftotext", "-layout", str(pdf), str(layout)], check=True)
    subprocess.run(["pdftotext", str(pdf), str(plain)], check=True)
    pages = plain.read_text().split("\f")
    if not pages[-1].strip():
        pages.pop()
    return pages


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def moore_published_fit(f_gw_hz, rho):
    """Eq. (21), evaluated only inside the paper's directly sampled SNR span.

    This is an empirical published PE fit, not a waveform/FF computation.
    Frequency means twice the radial/periastron-to-periastron frequency.
    """
    if not 0.5e-3 <= f_gw_hz <= 10e-3 or not 8 <= rho <= 19:
        raise ValueError("Use the inspected frequency range and directly sampled rho=8..19 only")
    f_mhz = f_gw_hz / 1e-3
    return (rho**(-1.54) + rho**(-1)) * (
        1.08 + .87 * math.atan(1.08 * (f_mhz - 2.13))
        - .55 * math.atan(2.08 * (f_mhz - 1.22)))


def prepare():
    # Pin current source/code before generating research artifacts; later calls
    # fail if these inputs change rather than silently relabeling the assessment.
    extra = [TEX, ROOT / "src/egb_multi/comparison.py", ROOT / "src/egb_multi/circular_fit.py", FROZEN]
    snapshot = OUT / "initial_input_hashes.json"
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in extra}
    if snapshot.exists():
        assert json.loads(snapshot.read_text()) == hashes, "Task 3 source/configuration changed"
    else:
        save_json(snapshot.name, hashes)
    evidence = []
    for item in PAPERS:
        pdf = OUT / "literature" / (item["file"] + ".pdf")
        pages = extract(pdf)
        locators = {marker: [i + 1 for i, page in enumerate(pages)
                            if normalized(marker) in normalized(page)] for marker in item["markers"]}
        evidence.append(dict(item, url="https://arxiv.org/abs/" + item["id"],
                             pdf=str(pdf.relative_to(ROOT)), pdf_pages=len(pages),
                             marker_pdf_pages=locators, pdf_sha256=digest(pdf)))
    save_json("literature_evidence.json", evidence)
    missing = [(p["id"], marker) for p in evidence for marker, pages in p["marker_pdf_pages"].items() if not pages]
    assert not missing, missing
    source_hits = []
    pattern = re.compile(r"threshold|measur|distinguish|adequate|invalidates", re.I)
    for number, line in enumerate(TEX.read_text().splitlines(), 1):
        if pattern.search(line):
            source_hits.append(dict(line=number, text=line.strip()))
    documents = {}
    for name in ("submitted_2608.24546v1", "referee_DV14615"):
        pdf = OUT / "input_documents" / (name + ".pdf")
        pages = extract(pdf)
        hits = []
        for number, page in enumerate(pages, 1):
            for line in page.splitlines():
                if pattern.search(line):
                    hits.append(dict(pdf_page=number, text=line.strip()))
        documents[name] = dict(pdf_sha256=digest(pdf), pdf_pages=len(pages), wording_hits=hits)
    save_json("manuscript_wording_audit.json", dict(
        interpretation="Search locators, not automatic classifications of erroneous claims; see the reviewed report",
        local_tex_sha256=digest(TEX), local_source_hits=source_hits, documents=documents,
        submitted_abstract_already_states_ten_percent_SNR_loss=True,
        local_tex_differs_from_submitted_PDF=True,
        exempt_contexts=["harmonic relative-power threshold", "automatic evolution tolerance", "general astrophysical motivation", "future parameter-estimation work"]))
    rows = [dict(f_gw_hz=f, f_gw_mhz=f / 1e-3, rho=rho,
                 e_min_published_fit=moore_published_fit(f, rho),
                 origin="Moore et al. Eq. (21); scalar evaluation, not reproduced PE")
            for f in (.5e-3, 1e-3, 2e-3, 3e-3) for rho in (8., 10., 15., 19.)]
    write_csv("moore_published_fit_reference.csv", rows)
    fig3_value = moore_published_fit(3e-3, 13.)
    assert abs(fig3_value - .098) < .0055, "Eq. (21) transcription inconsistent with Fig. 3 fiducial point"
    save_json("published_fit_transcription_check.json", dict(
        source="Moore et al., Fig. 3 fiducial point and Eq. (21)",
        f_gw_hz=.003, rho=13., figure_approximate_e_min=.098,
        fit_e_min=fig3_value, absolute_difference=abs(fig3_value-.098),
        comparison_tolerance=.0055,
        interpretation="Transcription/plot consistency only; not reproduction of Bayesian posteriors"))
    projection_rows = []
    for rho in (10., 20., 100.):
        for loss in (.1, .01, .0001):
            ff = 1 - loss
            # Direct two-coordinate geometry independently exhibits projection:
            # signal=(rho*FF, rho*sqrt(1-FF^2)); unit template=(1,0).
            common_scale = rho * ff
            second_component = rho * math.sqrt((1 - ff) * (1 + ff))
            direct_residual_sq = second_component**2
            identity_residual_sq = rho**2 * (1 - ff**2)
            assert math.isclose(direct_residual_sq, identity_residual_sq, rel_tol=1e-12, abs_tol=1e-12)
            projection_rows.append(dict(rho=rho, ff=ff, SNR_loss=loss,
                common_scale=common_scale, residual_SNR=second_component,
                residual_norm_squared=direct_residual_sq,
                zero_noise_delta_log_L=direct_residual_sq / 2,
                origin="Illustrative scalar projection; not a LISA injection or detection threshold"))
    write_csv("scalar_residual_illustrations.csv", projection_rows)
    frozen = json.loads(FROZEN.read_text())
    save_json("interpretation_contract.json", dict(
        observable="Best coherent circular-template SNR fraction for a deterministic eccentric signal under one fixed metric",
        loss="1-FF", contour_ff=.9, retained_SNR_fraction=.9,
        retained_SNR_squared_fraction=.81, residual_norm_fraction_squared=.19,
        noise_metric="4 Re integral h_dagger S_AE_inverse g df; one-sided fractional-frequency covariance",
        amplitude="One common free nonnegative A/E scale; physical polarization family permits either sign of shape",
        residual_identity="R^2=rho^2*(1-FF^2)",
        likelihood_identity="Delta ln L_max=R^2/2 in zero noise when the eccentric family contains the injection",
        small_loss_scaling="R^2 approximately 2*rho^2*(1-FF)",
        universal_detection_threshold_established=False, effective_dimension_assigned=False,
        real_noise_Bayes_factor_computed=False, published_fit_extrapolated=False,
        new_waveform_runs=False, new_parameter_estimation_runs=False,
        current_reference=frozen["reference"], duration_s=frozen["duration_s"],
        template_family=frozen["template_family"],
        revised_production_grid_completed=frozen["production_grid_completed"]))
    save_json("moore_comparison_contract.json", dict(
        source="2310.06568v2 Sections 2-4.1 and Appendices A-B; Eq. (21)",
        observation_duration_years=4., signal_noise_realization="zero_noise",
        likelihood_noise="SciRD instrumental PSD plus duration-scaled Galactic confusion; independent TDI channels",
        signal_parameters="11 sampled quantities; flat priors in stated coordinates",
        injection_fiducial_masses_solar=[1.4,1.4],
        injection_sky="Galactic centre", injection_inclination_rad=math.pi/4,
        directly_sampled_snr_range=[7,19], formula_frequency_range_mhz=[.5,10.],
        local_formula_evaluation_snr_range=[8,19],
        posterior_criterion="median(e posterior)/standard_deviation(e posterior)>3",
        Bayes_crosscheck="Nested-sampling evidence ratio and Savage-Dickey density ratio; thresholds aligned empirically with posterior criterion",
        source_physics="Small-e harmonic amplitudes; prescribed linear chirp and periastron rate; not coupled evolving-e Peters integration",
        response="Klein (2021) response reference; rigid-adiabatic TDI construction inspected in Task 4",
        comparison="Different observable/model/noise/duration; reference fit is not eGB output",
        extrapolation_to_point_one_mhz=False,
        matched_configuration_numerical_agreement_claimed=False))
    print(json.dumps(dict(primary_papers=len(evidence), missing_markers=missing,
        source_wording_hits=len(source_hits), literature_locators={p["id"]:p["marker_pdf_pages"] for p in evidence},
        scalar_reference_rows=len(rows), projection_checks=len(projection_rows)), indent=2))


def record():
    prior = json.loads((ROOT / "outputs/referee/task_5_1_5_2_20261012/manifest.json").read_text())
    for relative, wanted in prior["inspected_source_hashes"].items():
        assert digest(ROOT / relative) == wanted, relative
    documents = [ROOT / "reports" / name for name in (
        "referee_task_3_subtasks.md", "referee_task_3_1_3_4_threshold_interpretation.md", "referee_revision_table.md")]
    for path in documents:
        for raw in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if raw.startswith(("http://", "https://")):
                continue
            target = re.sub(r":\d+$", "", raw.strip("<>"))
            local = Path(target) if target.startswith("/") else path.parent / target
            assert local.exists(), (str(path), target)
    for path in documents[:2]:
        shutil.copy2(path, OUT / path.name)
    state = dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),
        status="threshold_wording_statistic_and_focused_literature_assessment_completed",
        completed_subtasks=["3.1", "3.2", "3.3", "3.4"],
        remaining_subtasks=["3.5", "3.6"], entire_task_3_complete=False,
        completed_scope="Submitted/local wording audit, SNR-loss/residual identities, high-SNR assumptions and Moore comparison",
        quantitative_eGB_vs_Moore_agreement_established=False,
        new_simulation_runs=False, new_Bayesian_analysis=False,
        manuscript_edits_applied=False, source_code_modified=False,
        effective_dimension_assigned=False,
        numerical_dependency="Task 2.9 for final numerical contour conclusions",
        manuscript_workflow="Manuscript edits and final reply/checks deferred to the end")
    save_json("work_state.json", state)
    inputs = [*documents, TEX, FROZEN, ROOT / "src/egb_multi/comparison.py", ROOT / "src/egb_multi/circular_fit.py", Path(__file__).resolve()]
    outputs = [p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"]
    save_json("manifest.json", dict(created_utc=state["reviewed_utc"],
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        unchanged_source_hashes=prior["inspected_source_hashes"],
        output_sha256={str(p.relative_to(ROOT)):digest(p) for p in outputs},
        interpretation="Focused research assessment and scalar formula evaluation; no new inference or waveform validation"))
    print(json.dumps(dict(completed=state["completed_subtasks"],
                         unchanged_source_hash_errors=0, local_link_errors=0), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prepare()
    if not args.prepare:
        record()
