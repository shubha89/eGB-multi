"""Retention and checkpoint recovery without changing numerical calculations."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tarfile
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import referee_task_2_slurm as slurm
import referee_task_2_waveform_cache as cache


def metadata():
    return dict(source={"frequency":.001}, evolution="peters_mathews", convention={"unit":"fractional_frequency"},
        duration_s=31557600., dt_s=2.5, pm_ode_rtol=1e-12, pm_ode_phase_atol=1e-10,
        signal_sha256="signal digest", signal_array_sha256={"A":"a", "E":"e"}, injection_wall_s=1.)


def test_cleanup_preserves_results_and_other_arrays(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("production_signal.npz", "production_signal.tmp.npz", "annual_covariance.npz", "production_source.json"):
        (inputs / name).write_bytes(name.encode())
    result = tmp_path / "result.json"
    result.write_bytes(b'accepted result')
    cache.discard(tmp_path, "point_worker_completed")
    assert not (inputs / "production_signal.npz").exists()
    assert not (inputs / "production_signal.tmp.npz").exists()
    assert (inputs / "annual_covariance.npz").read_bytes() == b'annual_covariance.npz'
    assert (inputs / "production_source.json").read_bytes() == b'production_source.json'
    assert result.read_bytes() == b'accepted result'
    event = json.loads((tmp_path / "waveform_cache_retention.json").read_text())["events"][0]
    assert event["deletion_completed"] and len(event["removed_generated_files"]) == 2


def test_cleanup_refuses_symbolic_waveform_input(tmp_path):
    (tmp_path / "inputs").mkdir()
    external = tmp_path / "original.npz"
    external.write_bytes(b'original')
    (tmp_path / "inputs/production_signal.npz").symlink_to(external)
    with pytest.raises(RuntimeError, match="nonregular"):
        cache.discard(tmp_path, "failed")
    assert external.read_bytes() == b'original'


def test_identical_regeneration_keeps_exact_metadata_bytes(tmp_path):
    (tmp_path / "inputs").mkdir()
    path = tmp_path / "inputs/production_source.json"
    old = metadata()
    cache.save(path, old)
    original = path.read_bytes()
    module = SimpleNamespace(save=cache.save)
    new = dict(old, injection_wall_s=100.)
    with cache.preserve_regenerated_metadata(module, tmp_path):
        module.save(path, new)
    assert path.read_bytes() == original
    assert module.save is cache.save
    records = list(tmp_path.glob("input_regeneration_history/*/regeneration.json"))
    assert json.loads(records[0].read_text())["bitwise_identical_input"]


def test_changed_regeneration_cannot_resume_completed_basin(tmp_path):
    (tmp_path / "inputs").mkdir()
    path = tmp_path / "inputs/production_source.json"
    old = metadata()
    cache.save(path, old)
    before = path.read_bytes()
    attempt = tmp_path / "attempt_000"
    attempt.mkdir()
    (attempt / "production_search_baseline.json").write_text('completed basin evidence')
    module = SimpleNamespace(save=cache.save)
    with pytest.raises(RuntimeError, match="differs from checkpoint"):
        with cache.preserve_regenerated_metadata(module, tmp_path):
            module.save(path, dict(old, signal_sha256="different"))
    assert path.read_bytes() == before
    assert module.save is cache.save


def test_changed_input_without_checkpoint_is_traced_as_fresh_search(tmp_path):
    (tmp_path / "inputs").mkdir()
    path = tmp_path / "inputs/production_source.json"
    old = metadata()
    cache.save(path, old)
    before = path.read_bytes()
    module = SimpleNamespace(save=cache.save)
    with cache.preserve_regenerated_metadata(module, tmp_path):
        module.save(path, dict(old, signal_sha256="different"))
    assert json.loads(path.read_text())["signal_sha256"] == "different"
    original = next(tmp_path.glob("input_regeneration_history/*/original_source_metadata.json"))
    assert original.read_bytes() == before


def test_changed_physical_configuration_is_refused(tmp_path):
    (tmp_path / "inputs").mkdir()
    path = tmp_path / "inputs/production_source.json"
    old = metadata()
    cache.save(path, old)
    module = SimpleNamespace(save=cache.save)
    with pytest.raises(RuntimeError, match="physical configuration"):
        with cache.preserve_regenerated_metadata(module, tmp_path):
            module.save(path, dict(old, dt_s=5.))


def test_wrapper_cleans_interrupted_waveform_and_keeps_metadata(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs/referee", prefix=".cache_test_") as name:
        out = Path(name)
        entry = slurm.production.original_grid()[1][0]
        cache.save(out / "queue.json", [entry])
        cache.save(out / "manifest.json", {})
        (out / "parallel").mkdir()
        cache.save(out / "parallel/waveform_retention.json", dict(mode="transient"))
        plan = out / "plan.json"
        cache.save(plan, dict(campaign=slurm.relative(out), work_campaign=slurm.relative(out), mode="production",
            campaign_manifest_sha256=cache.digest(out / "manifest.json"), jobs=[entry], source_sha256={},
            cache_helper_sha256=cache.digest(Path(cache.__file__)),
            retention_policy_sha256=cache.digest(out / "parallel/waveform_retention.json")))
        monkeypatch.setattr(slurm.production, "verify_inputs", lambda out:None)
        monkeypatch.setattr(slurm, "platform_gate", lambda out:None)
        def interrupted(work, selected):
            inputs = work / "points" / selected["id"] / "inputs"
            inputs.mkdir()
            (inputs / "production_signal.npz").write_bytes(b'generated')
            cache.save(inputs / "production_source.json", metadata())
            raise KeyboardInterrupt()
        monkeypatch.setattr(slurm.production, "worker", interrupted)
        with pytest.raises(KeyboardInterrupt):
            slurm.run_point(out, plan, 0)
        folder = out / "points" / entry["id"]
        assert not (folder / "inputs/production_signal.npz").exists()
        assert (folder / "inputs/production_source.json").exists()
        assert not (folder / "execution.lock").exists()


def test_transfer_omits_generated_arrays_and_seals_their_hashes(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs/referee", prefix=".bundle_test_") as name:
        base = Path(name)
        out = base / "campaign"
        inputs = out / "points/example/inputs"
        inputs.mkdir(parents=True)
        signal_file = inputs / "production_signal.npz"
        signal_file.write_bytes(b'generated waveform')
        cache.save(inputs / "production_source.json", metadata())
        cache.save(out / "manifest.json", dict(input_sha256={}))
        (out / "parallel").mkdir()
        cache.save(out / "parallel/waveform_retention.json", dict(mode="transient"))
        monkeypatch.setattr(slurm.production, "verify_inputs", lambda out:None)
        archive = slurm.bundle(out, base / "transfer")
        manifest = json.loads((base / "transfer/transfer_manifest.json").read_text())
        assert len(manifest["omitted_generated_waveforms"]) == 1
        assert manifest["omitted_generated_waveforms"][0]["sha256"] == cache.digest(signal_file)
        with tarfile.open(archive) as stream:
            names = stream.getnames()
        assert not any(Path(n).name == "production_signal.npz" for n in names)
        assert slurm.relative(inputs / "production_source.json") in names
        assert signal_file.exists()  # Export does not delete baseline data.
