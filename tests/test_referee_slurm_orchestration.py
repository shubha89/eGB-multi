"""Scheduling/provenance checks, without executing a scientific calculation."""

from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import referee_task_2_slurm as slurm


def accepted(entry):
    return dict(entry, status="passed_point_controls", fitting_factor=.99,
        checks={k:True for k in slurm.REQUIRED_CHECKS})


def test_pending_excludes_accepted_but_preserves_results(tmp_path):
    _, queue = slurm.production.original_grid()
    queue = queue[:3]
    slurm.production.save(tmp_path / "queue.json", queue)
    folder = tmp_path / "points" / queue[0]["id"]
    folder.mkdir(parents=True)
    path = folder / "result.json"
    slurm.production.save(path, accepted(queue[0]))
    before = path.read_bytes()
    original, pending = slurm.pending_entries(tmp_path)
    assert original == queue and pending == queue[1:]
    assert path.read_bytes() == before


def test_failed_point_is_never_silently_rescheduled(tmp_path):
    _, queue = slurm.production.original_grid()
    entry = queue[0]
    slurm.production.save(tmp_path / "queue.json", [entry])
    folder = tmp_path / "points" / entry["id"]
    folder.mkdir(parents=True)
    row = accepted(entry)
    row["checks"]["native_spectral_agreement"] = False
    slurm.production.save(folder / "result.json", row)
    with pytest.raises(RuntimeError, match="requires investigation"):
        slurm.pending_entries(tmp_path)


def test_incomplete_control_set_is_not_accepted():
    assert not slurm.passed(dict(status="passed_point_controls", checks={"dummy":True}))


def test_duplicate_queue_is_refused(tmp_path):
    _, queue = slurm.production.original_grid()
    slurm.production.save(tmp_path / "queue.json", [queue[0], queue[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        slurm.pending_entries(tmp_path)


def test_directory_lock_rejects_duplicates_and_releases_on_interrupt(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('raw evidence')
    with pytest.raises(KeyboardInterrupt):
        with slurm.point_lock(tmp_path):
            with pytest.raises(RuntimeError, match="locked"):
                with slurm.point_lock(tmp_path):
                    pass
            raise KeyboardInterrupt()
    assert not (tmp_path / "execution.lock").exists()
    assert path.read_text() == 'raw evidence'


def test_production_requires_fresh_platform_pilots(tmp_path):
    with pytest.raises(RuntimeError, match="Fresh platform pilot"):
        slurm.platform_gate(tmp_path)


def test_pilot_plan_has_two_fresh_jobs_and_valid_submission_script(monkeypatch):
    # A repository-local temporary campaign exercises generated destination
    # paths and the actual shell script without touching production evidence.
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs/referee", prefix=".slurm_test_") as temp:
        out = Path(temp)
        _, queue = slurm.production.original_grid()
        slurm.production.save(out / "queue.json", queue)
        slurm.production.save(out / "manifest.json", dict(input_sha256={slurm.relative(slurm.production.FROZEN):"test"}))
        (out / "shared").mkdir()
        (out / "shared/annual_covariance.npz").write_bytes(b'test fixture')
        for entry in queue[:2]:
            folder = out / "points" / entry["id"]
            folder.mkdir(parents=True)
            slurm.production.save(folder / "result.json", accepted(entry))
        monkeypatch.setattr(slurm.production, "verify_inputs", lambda out: None)
        directory = slurm.make_plan(out, "pilots", 32, 1)
        plan = slurm.production.read(directory / "plan.json")
        assert [e["id"] for e in plan["jobs"]] == list(slurm.PILOT_IDS)
        assert plan["concurrency"] == 2
        assert plan["chunks"] == [dict(offset=0,count=1), dict(offset=1,count=1)]
        shell = (directory / "submit.sh").read_text()
        assert "afterok" in shell and "%2" in shell and str(ROOT) not in shell
        subprocess.run(["bash", "-n", str(directory / "submit.sh")], check=True)
        assert (out / "parallel/platform_validation/shared/annual_covariance.npz").read_bytes() == b'test fixture'


def test_transfer_verification_detects_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(slurm, "ROOT", tmp_path)
    path = tmp_path / "source.py"
    path.write_text('unchanged calculation')
    manifest = tmp_path / "transfer.json"
    slurm.production.save(manifest, dict(files=[dict(path="source.py",sha256=slurm.production.digest(path))]))
    assert slurm.verify_transfer(manifest)["files"] == 1
    path.write_text('different calculation')
    with pytest.raises(RuntimeError, match="differs"):
        slurm.verify_transfer(manifest)
