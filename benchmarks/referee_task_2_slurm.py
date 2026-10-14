"""Portable scheduling for the unchanged, frozen Task 2.9 calculation.

Never submits a job automatically. Plans require two fresh platform-agreement
pilots before production. One worker handles one point; directory locks reject
duplicate execution. Completed results and incomplete attempts are preserved.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import referee_task_2_production as production
import referee_task_2_waveform_cache as waveform_cache

BATCH = ROOT / "benchmarks/referee_task_2_array.sbatch"
DEFAULT = ROOT / "outputs/referee/task_2_9_20261012_production"
PILOT_IDS = ("p_9ff7f1012ff0a89a", "p_06d9d5194f47b99a")
PLATFORM_FF_TOLERANCE = 1e-4  # Existing optimization allocation; same calculation.
REQUIRED_CHECKS = {"native_spectral_agreement", "inactive_frequency_chirp_boundaries",
    "final_refinement_success", "final_refinement_shift", "retains_legacy_candidates",
    "no_legacy_candidate_exceeds_winner", "all_basins_completed",
    "blind_restart_in_every_basin", "coherent_full_AE_weighting"}


def retention_policy(out):
    path = out / "parallel/waveform_retention.json"
    return production.read(path) if path.exists() else dict(mode="retain")


def configure_retention(out, mode):
    production.verify_inputs(out)
    if (out / "controller.lock").exists() or list(out.glob("**/execution.lock")):
        raise RuntimeError("Stop active workers before changing the retention policy")
    path = out / "parallel/waveform_retention.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.copy2(path, path.with_name("waveform_retention_" + stamp + ".json"))
    production.save(path, dict(mode=mode, created_utc=datetime.now(timezone.utc).isoformat(),
        temporary_waveform_file_during_fit=True, cleanup_on_normal_exit_and_caught_interrupt=(mode=="transient"),
        exceptional_hard_kill="May leave a temporary cache; inspect the stopped job before recovery",
        retained=["source parameters", "array/file hashes", "noise covariance", "search diagnostics", "fitting factors", "checkpoints"],
        numerical_source_and_objective_unchanged=True,
        checkpoint_regeneration="Require identical arrays, file digest and original metadata before checkpoint reuse"))
    return dict(retention_policy=relative(path), mode=mode, jobs_submitted=False)


def local(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Use this transferred eGB workspace")
    return path


def relative(path):
    return str(path.relative_to(ROOT))


def passed(row):
    checks = row.get("checks", {})
    return (row.get("status") == "passed_point_controls" and REQUIRED_CHECKS <= checks.keys()
            and all(v is True for v in checks.values()))


def pending_entries(out):
    queue = production.read(out / "queue.json")
    if len({e["id"] for e in queue}) != len(queue):
        raise ValueError("Duplicate scientific point in queue")
    pending = []
    for entry in queue:
        if entry["id"] != production.case_id(entry["point"]):
            raise ValueError("Queue ID does not match its scientific point")
        path = out / "points" / entry["id"] / "result.json"
        if path.exists():
            row = production.read(path)
            if row.get("point") != entry["point"] or not passed(row):
                raise RuntimeError("Existing point requires investigation: " + entry["id"])
        else:
            pending.append(entry)
    return queue, pending


@contextmanager
def point_lock(folder):
    lock = folder / "execution.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError("Point is locked; check job status before recovering " + str(lock))
    try:
        production.save(lock / "owner.json", dict(pid=os.getpid(), hostname=os.uname().nodename,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"), array_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
            created_utc=datetime.now(timezone.utc).isoformat()))
        yield
    finally:
        shutil.rmtree(lock)  # Only the lock owned by this process, never results.


def platform_gate(out):
    for ident in PILOT_IDS:
        check = out / "parallel/platform_validation/points" / ident / "platform_agreement.json"
        if not check.exists():
            raise RuntimeError("Fresh platform pilot is required: " + ident)
        record = production.read(check)
        baseline = out / "points" / ident / "result.json"
        rerun = check.parent / "result.json"
        if (record.get("status") != "passed_platform_agreement"
                or record.get("platform") != sys.platform
                or record["baseline_sha256"] != production.digest(baseline)
                or record["rerun_sha256"] != production.digest(rerun)
                or not passed(production.read(rerun))):
            raise RuntimeError("Platform pilot evidence changed or failed: " + ident)


def make_plan(out, mode, concurrency, chunk_size):
    production.verify_inputs(out)
    if (out / "controller.lock").exists():
        raise RuntimeError("Do not plan parallel work while a serial controller lock exists")
    if not 1 <= concurrency <= 128 or not 1 <= chunk_size <= 500:
        raise ValueError("Use 1..128 concurrent points and 1..500 array elements per chunk")
    queue, pending = pending_entries(out)
    if mode == "pilots":
        jobs = [next(e for e in queue if e["id"] == ident) for ident in PILOT_IDS]
        for e in jobs:
            if not passed(production.read(out / "points" / e["id"] / "result.json")):
                raise RuntimeError("Platform pilot requires an accepted local reference")
        work = out / "parallel/platform_validation"
        work.mkdir(parents=True, exist_ok=True)
        if not (work / "manifest.json").exists():
            shutil.copy2(out / "manifest.json", work / "manifest.json")
            (work / "shared").mkdir(exist_ok=True)
            (work / "shared/annual_covariance.npz").symlink_to(
                os.path.relpath(out / "shared/annual_covariance.npz", work / "shared"))
        concurrency = min(concurrency, 2)
    else:
        platform_gate(out)
        jobs, work = pending, out
    if not jobs:
        raise ValueError("No pending points; collect results and advance contour refinements")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = out / "parallel/plans" / (stamp + "_" + mode)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "logs").mkdir()
    chunks = [dict(offset=i, count=min(chunk_size, len(jobs)-i)) for i in range(0, len(jobs), chunk_size)]
    manifest = production.read(out / "manifest.json")
    production.save(directory / "plan.json", dict(mode=mode, campaign=relative(out), work_campaign=relative(work),
        created_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs, chunks=chunks, concurrency=concurrency,
        source_sha256={relative(Path(__file__)):production.digest(Path(__file__)), relative(BATCH):production.digest(BATCH)},
        cache_helper_sha256=production.digest(Path(waveform_cache.__file__)),
        retention_policy=retention_policy(out),
        retention_policy_sha256=production.digest(out / "parallel/waveform_retention.json") if (out / "parallel/waveform_retention.json").exists() else None,
        campaign_manifest_sha256=production.digest(out / "manifest.json"),
        frozen_configuration_sha256=manifest["input_sha256"][relative(production.FROZEN)],
        baseline_result_sha256={e["id"]:production.digest(out / "points" / e["id"] / "result.json") for e in jobs} if mode == "pilots" else {},
        resource_proposal=dict(cpus_per_point=1, memory_per_cpu="8G", time="24:00:00"),
        note="Resource proposal requires Euler pilot/account/storage checks; no job has been submitted"))
    # Resolve all paths on the destination, without embedding Mac paths.
    lines = ["#!/bin/bash", "set -euo pipefail", ': "${EGB_PYTHON:?Set EGB_PYTHON to the cluster Python executable}"',
        'task_root="$(pwd -P)"', '[[ -f "$task_root/benchmarks/referee_task_2_slurm.py" ]]',
        f'task_campaign="$task_root/{relative(out)}"', f'task_plan="$task_root/{relative(directory / "plan.json")}"',
        '"$EGB_PYTHON" "$task_root/benchmarks/referee_task_2_slurm.py" --campaign "$task_campaign" preflight',
        '[[ ! -e "' + relative(directory / "submitted_jobs.txt") + '" ]] || { echo "This plan was already submitted" >&2; exit 1; }',
        'task_previous=""']
    for chunk in chunks:
        dependency = '${task_previous:+--dependency=afterok:$task_previous}'
        lines += [f'task_job=$(sbatch --parsable {dependency} --array=0-{chunk["count"]-1}%{concurrency} '
            f'--output="{relative(directory)}/logs/%A_%a.out" '
            '"$task_root/benchmarks/referee_task_2_array.sbatch" "$task_root" "$EGB_PYTHON" "$task_campaign" "$task_plan" '
            f'{chunk["offset"]})', 'task_previous="${task_job%%;*}"',
            f'printf "%s\\n" "$task_previous" >> "{relative(directory)}/submitted_jobs.txt"']
    # Chunks run sequentially: concurrency is capped across the whole campaign.
    (directory / "submit.sh").write_text("\n".join(lines) + "\n")
    return directory


def preflight(out):
    production.verify_inputs(out)
    queue, pending = pending_entries(out)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "outputs/referee/.precision_deps")]
    from importlib.metadata import version
    from referee_task_2_annual_search import assert_local
    assert_local()
    from egb_multi.comparison import TDIConvention
    TDIConvention(**production.read(out / "manifest.json")["frozen_configuration"]["tdi_convention"])
    return dict(status="portable_input_and_import_checks_passed", queued=len(queue), pending=len(pending),
        python=sys.version, executable=sys.executable,
        versions={name:version(name) for name in ("numpy", "scipy", "jax", "pytdi", "lisaorbits", "mpmath")},
        source_root=str(ROOT / "src/egb_multi"), platform_pilot_gate_required_before_production=True)


def run_point(out, plan_file, index):
    plan = production.read(plan_file)
    if plan["campaign"] != relative(out) or plan["campaign_manifest_sha256"] != production.digest(out / "manifest.json"):
        raise RuntimeError("Plan/campaign provenance differs")
    for name, expected in plan["source_sha256"].items():
        if production.digest(ROOT / name) != expected:
            raise RuntimeError("Scheduling source changed: " + name)
    if plan.get("cache_helper_sha256") != production.digest(Path(waveform_cache.__file__)):
        raise RuntimeError("Waveform-cache helper changed")
    policy_path = out / "parallel/waveform_retention.json"
    policy_hash = production.digest(policy_path) if policy_path.exists() else None
    if plan.get("retention_policy_sha256") != policy_hash:
        raise RuntimeError("Waveform retention policy changed; prepare a fresh plan")
    production.verify_inputs(out)
    if not 0 <= index < len(plan["jobs"]):
        raise ValueError("Array index outside immutable plan")
    if plan["mode"] != "pilots":
        platform_gate(out)
    entry = plan["jobs"][index]
    if entry not in production.read(out / "queue.json") or entry["id"] != production.case_id(entry["point"]):
        raise RuntimeError("Plan entry is not the original queued scientific point")
    work = local(ROOT / plan["work_campaign"])
    folder = work / "points" / entry["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with point_lock(folder):
        outcome = "interrupted_or_failed"
        try:
            _run_locked_point(out, plan_file, plan, work, folder, entry)
            outcome = "point_worker_completed"
        finally:
            if retention_policy(out)["mode"] == "transient":
                waveform_cache.discard(folder, outcome)


def _run_locked_point(out, plan_file, plan, work, folder, entry):
    result_file = folder / "result.json"
    if result_file.exists():
        row = production.read(result_file)
        if not passed(row) or row["point"] != entry["point"]:
            raise RuntimeError("Saved point requires investigation")
    else:
        if shutil.disk_usage(work).free < 25*2**30:
            raise RuntimeError("Less than 25 GiB free; review retained waveform storage")
        production.save(folder / "entry.json", entry)
        # New process per point; original worker and numerical code unchanged.
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "outputs/referee/.precision_deps")]
        with waveform_cache.preserve_regenerated_metadata(production, folder):
            production.worker(work, entry)
    if plan["mode"] == "pilots":
        baseline_path = out / "points" / entry["id"] / "result.json"
        if production.digest(baseline_path) != plan["baseline_result_sha256"][entry["id"]]:
            raise RuntimeError("Local pilot reference changed")
        old, new = production.read(baseline_path), production.read(result_file)
        delta = abs(new["fitting_factor"]-old["fitting_factor"])
        agree = passed(old) and passed(new) and delta <= PLATFORM_FF_TOLERANCE
        production.save(folder / "platform_agreement.json", dict(
            status="passed_platform_agreement" if agree else "requires_investigation",
            platform=sys.platform, hostname=os.uname().nodename,
            baseline_sha256=production.digest(baseline_path), rerun_sha256=production.digest(result_file),
            absolute_FF_difference=delta, tolerance=PLATFORM_FF_TOLERANCE))
        if not agree:
            raise RuntimeError("Fresh platform pilot disagrees with local reference")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    production.save(folder / "scheduler_completion.json", dict(status="point_worker_completed",
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), peak_RSS_bytes=peak if sys.platform=="darwin" else peak*1024,
        plan_file=relative(plan_file), plan_sha256=production.digest(plan_file)))

def collect(out, advance):
    production.verify_inputs(out)
    queue, pending = pending_entries(out)
    rows = production.completed_results(out, queue)
    production.save(out / "results.json", rows)
    extra = []
    if advance:
        if pending or list((out / "points").glob("*/execution.lock")):
            raise RuntimeError("Finish and review all current points before advancing contours")
        extra = production.contour_next(out, queue, rows)
        if extra:
            queue.extend(extra)
            production.save(out / "queue.json", queue)
    state = dict(status="parallel_results_collected", completed_points=len(rows), queued_points=len(queue),
        pending_points=len(pending)+len(extra), new_contour_points=len(extra), base_points=2160,
        production_grid_complete=bool(advance and not pending and not extra), task_2_9_complete=False,
        updated_utc=datetime.now(timezone.utc).isoformat(),
        remaining="Review controls/precision and regenerate figures/table; final manuscript work deferred")
    production.save(out / "work_state.json", state)
    return state


def bundle(out, destination):
    production.verify_inputs(out)
    if (out / "controller.lock").exists() or list((out / "points").glob("*/execution.lock")):
        raise RuntimeError("Stop active jobs before sealing a transfer")
    destination = local(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {ROOT / name for name in production.read(out / "manifest.json")["input_sha256"]}
    files.update((Path(__file__), Path(waveform_cache.__file__), BATCH, ROOT / "pyproject.toml",
        ROOT / "tests/test_referee_slurm_orchestration.py", ROOT / "tests/test_referee_waveform_cache.py"))
    files.update(p for p in out.rglob("*") if p.is_file() or p.is_symlink())
    files.update(p for p in (ROOT / "outputs/referee/.precision_deps").rglob("*") if p.is_file())
    files = sorted(p for p in files if "__pycache__" not in p.parts and p.suffix != ".pyc")
    omitted = []
    if retention_policy(out)["mode"] == "transient":
        omitted = [dict(path=relative(p), sha256=production.digest(p), bytes=p.stat().st_size,
                        reason="Reproducible generated waveform; metadata and source retained")
                   for p in files if waveform_cache.generated_cache(p)]
        files = [p for p in files if not waveform_cache.generated_cache(p)]
    records = []
    for path in files:
        if path.is_symlink():
            target = local(path)
            records.append(dict(path=relative(path), symlink=os.path.relpath(target, path.parent)))
        else:
            records.append(dict(path=relative(path), sha256=production.digest(path), bytes=path.stat().st_size))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), files=records,
        omitted_generated_waveforms=omitted, waveform_retention=retention_policy(out),
        local_source=str(ROOT), relative_layout_preserved=True,
        note="Transfer seal checked before computation; future output updates legitimately differ")
    manifest_path = destination / "transfer_manifest.json"
    production.save(manifest_path, manifest)
    archive = destination / "task_2_9_euler.tar"
    with tarfile.open(archive, "w") as stream:
        for path, record in zip(files, records):
            info = stream.gettarinfo(str(path), arcname=record["path"])
            if path.is_symlink():
                info.linkname = record["symlink"]
                stream.addfile(info)
            else:
                with path.open("rb") as data:
                    stream.addfile(info, data)
        stream.add(manifest_path, arcname=relative(manifest_path), recursive=False)
    production.save(destination / "archive_check.json", dict(archive=relative(archive),
        sha256=production.digest(archive), bytes=archive.stat().st_size, sealed_files=len(records)))
    return archive


def verify_transfer(manifest_path):
    records = production.read(manifest_path)["files"]
    for record in records:
        path = ROOT / record["path"]
        if "symlink" in record:
            if not path.is_symlink() or os.readlink(path) != record["symlink"] or not path.exists():
                raise RuntimeError("Transferred link differs: " + record["path"])
        elif production.digest(path) != record["sha256"]:
            raise RuntimeError("Transferred file differs: " + record["path"])
    return dict(status="transfer_hashes_verified", files=len(records))


def terminate(signum, frame):
    raise KeyboardInterrupt("Scheduler termination signal; preserve checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--mode", choices=("pilots", "production"), required=True)
    plan.add_argument("--concurrency", type=int, default=32)
    plan.add_argument("--chunk-size", type=int, default=500)
    sub.add_parser("preflight")
    retention = sub.add_parser("retention")
    retention.add_argument("--mode", choices=("transient", "retain"), required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    aggregate = sub.add_parser("collect")
    aggregate.add_argument("--advance-contours", action="store_true")
    transfer = sub.add_parser("bundle")
    transfer.add_argument("--destination", type=Path, required=True)
    verify = sub.add_parser("verify-transfer")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    out = local(args.campaign)
    signal.signal(signal.SIGTERM, terminate)
    if args.command == "plan":
        result = dict(plan_directory=relative(make_plan(out,args.mode,args.concurrency,args.chunk_size)), submitted=False)
    elif args.command == "preflight":
        result = preflight(out)
    elif args.command == "retention":
        result = configure_retention(out,args.mode)
    elif args.command == "run":
        run_point(out,local(args.plan),args.index)
        result = dict(status="worker_finished")
    elif args.command == "collect":
        result = collect(out,args.advance_contours)
    elif args.command == "bundle":
        result = dict(archive=relative(bundle(out,args.destination)),submitted=False)
    else:
        result = verify_transfer(local(args.manifest))
    print(json.dumps(result,indent=2))
