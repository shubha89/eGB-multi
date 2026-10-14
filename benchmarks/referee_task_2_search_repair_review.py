"""Review a completed repaired high-PM baseline without certifying Task 2.8.

The review checks the original failure, exact annual replay, immutable resume
controls, completed basin coverage and independent native winner verification.
It records optimizer termination separately from execution completion.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/referee"


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    out = args.output.resolve()
    original = BASE / "task_2_8_20261008_annual_search_high_pm_spectral"
    repaired = BASE / "task_2_8_20261009_annual_search_high_pm_repaired"
    diagnostic = BASE / "task_2_8_20261009_boundary_repair"
    controls = BASE / "task_2_8_20261009_repair_summary"
    require(out.is_relative_to(ROOT), "Use this checkout")
    original_summary = read(original / "validation_summary.json")
    replay = read(diagnostic / "validation_summary.json")
    summary = read(repaired / "validation_summary.json")
    record = read(repaired / "high_pm_annual_search.json")
    checkpoint = read(repaired / "high_pm_search_baseline.json")
    resume = read(repaired / "high_pm_resume_controls.json")
    retained = read(repaired / "high_pm_retained_log_controls.json")
    require(original_summary["status"] == "failed_annual_search_parameter_domain",
            "Original failure must remain explicitly archived")
    require(replay["status"] == "passed_boundary_repair_diagnostic"
            and replay["original_failure_reproduced"]
            and replay["repaired_start_exactly_preserved"]
            and replay["actual_annual_objective_minus_saved"] == 0.,
            "Exact failure reproduction and actual annual repair replay required")
    require(summary["status"] == "annual_baseline_completed" and not summary["failures"],
            "A completed repaired baseline without failed winner checks is required")
    require(checkpoint["completed_basins"] == checkpoint["requested_basins"] == 12,
            "All twelve selected basins must complete")
    stages = checkpoint["basins_results"]
    require([row["basin_index"] for row in stages] == list(range(12)),
            "Exactly one completed record per selected basin is required")
    require(record["resume"]["completed_checkpoint_basins"] == original_summary["completed_basins"] == 5,
            "Original completed prefix must be preserved")
    require(len(resume) == 5 and all(row["delta_overlap"] == 0. for row in resume),
            "All five completed winners must retain exactly the original objective")
    require(len(retained) == 8 and all(row["delta_overlap"] == 0. for row in retained),
            "All eight retained log winners must retain exactly the original objective")
    for old_name, frozen_name in (
        ("high_pm_search_baseline.json", "input_resume_search_baseline.json"),
        ("high_pm_evaluations.jsonl", "input_retained_evaluations.jsonl"),
        ("high_pm_basin_selection.json", "input_resume_basin_selection.json"),
        ("manifest.json", "input_resume_manifest.json"),
    ):
        require((original / old_name).read_bytes() == (repaired / frozen_name).read_bytes(),
                "Original resume input changed: " + old_name)
    require(abs(record["native_minus_envelope"]) <= 2e-4,
            "Native winner verification exceeds its declared allowance")
    best = record["refinement"]["best"]["profile"]["overlap"]
    require(best >= original_summary["interim_best_overlap_not_certified_FF"] - 1e-13,
            "Repaired search lost a retained candidate")

    local_fits = []
    chart_controls = []
    for stage in stages:
        for index, fit in enumerate(stage["independent_start_results"]):
            local_fits.append(dict(basin_index=stage["basin_index"], start_index=index,
                success=fit["success"], message=fit["message"], iterations=fit["iterations"],
                reused_completed_prefix="checkpoint_origin" in stage))
            chart = fit["conditioning"].get("optimizer_chart_endpoint_roundoff_control")
            if chart:
                chart_controls.append(chart)
    final_fit = record["refinement"]["refinement"]
    chart_controls.append(final_fit["conditioning"]["optimizer_chart_endpoint_roundoff_control"])
    require(all(row["maximum_correction_in_bound_ulps"] <= row["roundtrip_allowance_in_bound_ulps"]
                for row in chart_controls), "Recorded endpoint correction exceeds the chart allowance")
    require(all(fit["success"] for fit in local_fits if fit["basin_index"] == 5),
            "Previously failing basin must terminate successfully")

    audits = []
    for path in (diagnostic, repaired):
        manifest = read(path / "manifest.json")
        errors = []
        for relative, expected in manifest["sha256"].items():
            frozen = path / "snapshot" / relative
            if not frozen.exists() or hashlib.sha256(frozen.read_bytes()).hexdigest() != expected:
                errors.append("snapshot:" + relative)
        for relative, expected in manifest["inputs"].items():
            source = ROOT / relative
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                errors.append("input:" + relative)
        audits.append(dict(run=str(path.relative_to(ROOT)), hash_errors=errors,
            snapshot_hashes=len(manifest["sha256"]), input_hashes=len(manifest["inputs"])))
    require(all(not row["hash_errors"] for row in audits), "Declared provenance hashes must agree")
    test_log = controls / "tests.log"
    test_match = re.search(r"(\d+) passed in ([\d.]+)s", test_log.read_text())
    require(test_match is not None and int(test_match[1]) >= 197,
            "The complete post-repair software test record is required")

    # Write only after checking a completed run; never create a passing review
    # from live checkpoints or from a successfully launched process.
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / "review_driver.py")
    snapshot = out / "snapshot" / Path(__file__).resolve().relative_to(ROOT)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, snapshot)
    input_files = [original / "validation_summary.json",
        diagnostic / "validation_summary.json", diagnostic / "boundary_replay.json",
        diagnostic / "actual_annual_repair.json", diagnostic / "manifest.json",
        repaired / "validation_summary.json", repaired / "high_pm_annual_search.json",
        repaired / "high_pm_search_baseline.json", repaired / "high_pm_resume_controls.json",
        repaired / "high_pm_retained_log_controls.json", repaired / "manifest.json", test_log]
    for path in input_files:
        target = out / "inputs" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    result = dict(status="passed_completed_high_pm_search_repair_review",
        created_utc=datetime.now(timezone.utc).isoformat(), search_repair_complete=True,
        task_2_8_complete=False, production_certified=False,
        root_cause="One-ulp lower-chirp overshoot in the old curvature-scale inverse chart",
        original_failure_preserved=True, completed_basins=12, requested_basins=12,
        native_bounded_baseline_overlap=record["native_winner_metrics"]["overlap"],
        native_minus_spectral=record["native_minus_envelope"],
        completed_prefix_objective_differences=[row["delta_overlap"] for row in resume],
        retained_log_objective_differences=[row["delta_overlap"] for row in retained],
        optimizer_termination=local_fits, final_optimizer_termination=dict(
            success=final_fit["success"], message=final_fit["message"]),
        roundoff_control=dict(local_fits_and_final=len(chart_controls),
            correction_events=sum(row["correction_events"] for row in chart_controls),
            corrected_axes=sorted({axis for row in chart_controls for axis in row["corrected_axes"]}),
            maximum_correction_in_bound_ulps=max(row["maximum_correction_in_bound_ulps"] for row in chart_controls)),
        provenance_audit=audits,
        tests=dict(passed=int(test_match[1]), wall_s=float(test_match[2])),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files},
        limitations=["Bounded annual baseline completed; finite basins and starts do not prove a global maximum",
            "Baseline secondary-only diagnostic metric is unchanged; final reference noise review remains open",
            "Numerical, band, noise-model and optimized contour convergence are separate Task 2.8 requirements"])
    (out / "repair_review.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(dict(
        created_utc=result["created_utc"],
        sha256={str(Path(__file__).resolve().relative_to(ROOT)):result["driver_sha256"]},
        inputs=result["input_sha256"], scope="Completed repair evidence review only; no new physics calculation"
    ), indent=2) + "\n")
    (out / "validation_summary.json").write_text(json.dumps({
        key:result[key] for key in ("status", "search_repair_complete", "task_2_8_complete",
            "production_certified", "completed_basins", "requested_basins", "native_bounded_baseline_overlap",
            "native_minus_spectral", "tests", "limitations")}, indent=2) + "\n")
    print(json.dumps({key:result[key] for key in ("status", "completed_basins", "native_bounded_baseline_overlap",
        "native_minus_spectral", "roundoff_control", "tests", "task_2_8_complete")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=BASE / "task_2_8_20261009_search_repair_review")
    run(parser.parse_args())
