"""Archive and check Task 4 literature evidence; no waveform packages are run.

Run --extract-only to produce page locators. Run without arguments after the
research notes are saved to record their provenance and unchanged input hashes.
Page locators refer to one-based PDF pages, not necessarily printed page labels.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/referee/task_4_20261012"
LITERATURE = OUT / "literature"
PAPERS = [
    dict(id="2310.06568v2", file="Moore_2024_2310.06568v2",
         doi="10.1093/mnras/stae1288", role="Direct Galactic-binary waveform/data-analysis precedent",
         sections="2, Eqs. (1)-(6); 3; 4.3-4.4; Appendix C",
         markers=["model for the LISA response", "reclassification", "heterodyning"],
         finding="Multi-harmonic, chirping/precessing quasi-monochromatic model with LISA/TDI response, Bayesian inference and harmonic misidentification/reclassification."),
    dict(id="0906.0313v3", file="Yunes_2009_0906.0313v3",
         doi="10.1103/PhysRevD.80.084001", role="Established analytic Fourier-domain eccentric source construction",
         sections="III-IV; VI; abstract",
         markers=["stationary phase approximation", "eighth order"],
         finding="Newtonian post-circular harmonic waveform transformed analytically using SPA; small-eccentricity expansion. Not our native unequal-arm nested TDI implementation."),
    dict(id="1602.03081v2", file="Tanay_2016_1602.03081v2",
         doi="10.1103/PhysRevD.93.064031", role="Fourier-domain eccentricity evolution and time-domain quasi-Keplerian precedent",
         sections="II opening paragraph; II.2; III",
         markers=["not pursued in the present extension", "2PN order Fourier phase"],
         finding="Analytic FD Newtonian amplitude/2PN Fourier phase with consistent eccentricity evolution through e^6; precession-induced FD harmonic structure explicitly not pursued. Separate TD quasi-Keplerian model includes precession."),
    dict(id="2605.15265v2", file="Xuan_2026_2605.15265v2",
         doi=None, role="Contextual stellar-mass BBH harmonic confusion and PN waveform study",
         sections="IV, Eqs. (13)-(16); V, Eqs. (19)-(24); Appendix A.4",
         markers=["rather than a full waveform fit", "sky and polarization", "x model"],
         finding="Harmonic/circular confusion and apparent chirp-mass mapping based on matching frequency derivative; not a full waveform fit or demonstrated global-fit implementation. PN comparison fixes intrinsic parameters. SNR uses averaged response."),
    dict(id="1801.08542v2", file="Klein_2018_1801.08542v2",
         doi="10.1103/PhysRevD.98.104043", role="Direct counterexample to general FD precession/evolution impracticality",
         sections="II, Eqs. (3)-(5), (16)-(18); II.2; III",
         markers=["Periastron precession effects", "shifted uniform asymptotics"],
         finding="FD eccentric waveforms separate orbital/periastron phases, include PN secular evolution and spin precession via SUA. Small-e amplitude approximation and stated validation domain; not an eGB benchmark."),
    dict(id="2106.10291v1", file="Klein_2021_2106.10291v1",
         doi=None, role="Moore et al.'s cited response reference and FD LISA/TDI precedent",
         sections="IV, Eqs. (117)-(131), paragraph following Eq. (131); V; VI",
         markers=["rigid adiabatic approximation", "time delay interferometry response"],
         finding="FD eccentric/precessing construction with a rigid-adiabatic XYZ TDI response. Published implementation timings concern its own setup and cannot be transferred to eGB."),
    dict(id="gr-qc/0610139v1", file="Tessmer_2007_gr-qc_0610139v1",
         doi="10.1111/j.1365-2966.2006.11179.x", role="Existing source-physics foundation already acknowledged by our paper",
         sections="Abstract; source-polarization and quasi-Keplerian construction",
         markers=["slowly precessing orbits", "polarizations"],
         finding="Established time-domain eccentric Galactic compact-binary polarization prescription; eGB's source physics must be presented with this attribution."),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    text = unicodedata.normalize("NFKD", text).casefold()
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def extract_papers():
    records = []
    for paper in PAPERS:
        pdf = LITERATURE / (paper["file"] + ".pdf")
        assert pdf.read_bytes().startswith(b"%PDF"), pdf
        target = pdf.with_suffix(".txt")
        subprocess.run(["pdftotext", "-layout", str(pdf), str(target)], check=True)
        plain = pdf.with_suffix(".plain.txt")
        subprocess.run(["pdftotext", str(pdf), str(plain)], check=True)
        pages = plain.read_text().split("\f")
        if not pages[-1].strip():
            pages.pop()
        normalized_pages = [normalized(page) for page in pages]
        locators = {}
        for marker in paper["markers"]:
            locators[marker] = [i + 1 for i, page in enumerate(normalized_pages)
                                if normalized(marker) in page]
        records.append(dict(paper, pdf=str(pdf.relative_to(ROOT)),
                            url="https://arxiv.org/abs/" + paper["id"],
                            pdf_url="https://arxiv.org/pdf/" + paper["id"],
                            pdf_pages=len(pages), marker_pdf_pages=locators,
                            pdf_sha256=digest(pdf), text_sha256=digest(target),
                            plain_text_sha256=digest(plain)))
    save_json("literature_evidence.json", records)
    missing = [(p["id"], marker) for p in records
               for marker, pages in p["marker_pdf_pages"].items() if not pages]
    print(json.dumps(dict(papers=len(records), missing_markers=missing,
                         page_locators={p["id"]: p["marker_pdf_pages"] for p in records}), indent=2))
    assert not missing, missing


def record():
    prior_path = ROOT / "outputs/referee/task_5_1_5_2_20261012/manifest.json"
    prior = json.loads(prior_path.read_text())
    for relative, expected in prior["inspected_source_hashes"].items():
        assert digest(ROOT / relative) == expected, relative
    inspected = [ROOT / "benchmarks" / filename for filename in (
        "plot_jaxgb_comprehensive_comparison.py",
        "jaxgbaccurate_calibrated_eccentric_study.py",
        "production_eccentric_circular_mismatch_grid.py",
        "jaxgbaccurate_calibrated_eccentric_study_1pn_periastron.csv")]
    rows = list(csv.DictReader(inspected[-1].open()))
    ratios = [float(row["bruteforce_xyz_s"]) / float(row["jaxgbaccurate_xyz_s"]) for row in rows]
    save_json("existing_benchmark_audit.json", dict(
        data=str(inspected[-1].relative_to(ROOT)), row_count=len(rows),
        duration_days=sorted({float(r["duration_days"]) for r in rows}),
        dt_s=sorted({float(r["dt_s"]) for r in rows}),
        samples=sorted({int(r["samples"]) for r in rows}),
        ratio_median=statistics.median(ratios), ratio_min=min(ratios), ratio_max=max(ratios),
        physics_modes=sorted({r["physics_mode"] for r in rows}),
        circular_raw_mismatches={r["f0_hz"]: float(r["mismatch_AE"]) for r in rows if float(r["eccentricity"]) == 0},
        generated_by_script_in_this_turn=False, external_waveform_packages_executed=False,
        method="Independent complex A/E/T calibration to local e=0, fixed-source normalized complex overlap, native sparse circular versus dense eccentric workloads",
        geometry_call="precompute_jax_link_geometry(state) without native orbits keyword",
        tdi_generations=dict(egb=1, jaxgb=1.5),
        band_hz=[1e-5, 1e-2], nyquist_hz=0.5 / float(rows[0]["dt_s"]),
        interpretation="Existing restricted benchmark; not matched response accuracy, not a full fitting factor, not the frozen Task 2.8 setup",
        inspected_sha256={str(p.relative_to(ROOT)): digest(p) for p in inspected}))
    documents = [ROOT / "reports" / filename for filename in (
        "referee_task_4_subtasks.md", "referee_task_4_literature_novelty.md",
        "referee_revision_table.md")]
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
                 status="focused_literature_and_contribution_assessment_completed",
                 completed_subtasks=["4.1", "4.2", "4.3", "4.4"],
                 deferred_subtasks=["4.5", "4.6", "4.7"], entire_task_4_complete=False,
                 manuscript_edits_applied=False, source_implementation_modified=False,
                 external_waveform_packages_executed=False, new_benchmark_run=False,
                 first_or_fastest_claim_established=False,
                 completed_scope="Referee-demand audit, seven primary documents, capability matrix and defensible contribution boundaries",
                 workflow="Manuscript edits, finished reply and manuscript verification deferred to final stage",
                 numerical_result_dependency="Task 2.9 before any new full-grid FF contour claim")
    save_json("work_state.json", state)
    inputs = [prior_path, *inspected, *documents, Path(__file__).resolve(),
              ROOT / "outputs/referee/task_2_8_20261012_completion/frozen_production_configuration.json"]
    outputs = [p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"]
    save_json("manifest.json", dict(created_utc=state["reviewed_utc"],
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs},
        unchanged_source_hashes=prior["inspected_source_hashes"],
        output_sha256={str(p.relative_to(ROOT)): digest(p) for p in outputs},
        interpretation="Focused literature and source-contract assessment; no comprehensive priority proof or performance validation"))
    print(json.dumps(dict(completed=state["completed_subtasks"], deferred=state["deferred_subtasks"],
                         unchanged_input_hash_errors=0, local_link_errors=0), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    extract_papers()
    if not args.extract_only:
        record()
