"""Discard reproducible injection arrays without changing the frozen search.

The legacy worker reads a temporary NPZ. Metadata/results survive cleanup.
Regeneration must be bitwise identical before a basin checkpoint is reused;
the original metadata bytes (including old generation timing) then survive.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def generated_cache(path):
    return path.parent.name == "inputs" and path.name in (
        "production_signal.npz", "production_signal.tmp.npz")


def discard(folder, outcome):
    """Remove ONLY generated waveform files, recording their exact digests."""
    paths = [folder / "inputs" / name for name in ("production_signal.npz", "production_signal.tmp.npz")]
    existing = [p for p in paths if p.exists() or p.is_symlink()]
    if not existing:
        return
    if any(p.is_symlink() or not p.is_file() for p in existing):
        raise RuntimeError("Refuse to remove a nonregular generated-cache path")
    entries = [dict(file=p.name, sha256=digest(p), bytes=p.stat().st_size,
                    kind="complete_injection" if p.name=="production_signal.npz" else "incomplete_write") for p in existing]
    record = folder / "waveform_cache_retention.json"
    previous = json.loads(record.read_text()) if record.exists() else dict(events=[])
    event = dict(utc=datetime.now(timezone.utc).isoformat(), outcome=outcome, removed_generated_files=entries,
        parameters_and_original_input_hashes_retained=True, waveform_regeneration_required=True)
    # Commit the intention/digests before unlinking, then confirm completion.
    previous["events"].append(dict(event, deletion_completed=False))
    save(record, previous)
    for path in existing:
        path.unlink()
    previous["events"][-1]["deletion_completed"] = True
    save(record, previous)


@contextmanager
def preserve_regenerated_metadata(production, folder):
    """Guard file-driven checkpoint provenance when a discarded cache returns."""
    meta = folder / "inputs/production_source.json"
    signal_file = folder / "inputs/production_signal.npz"
    if not meta.exists() or signal_file.exists():
        yield
        return
    old_bytes = meta.read_bytes()
    old = json.loads(old_bytes)
    has_checkpoint = any(folder.glob("attempt_*/production_search_baseline.json"))
    original_save = production.save

    def guarded_save(path, value):
        if path != meta:
            return original_save(path, value)
        keys = ("source", "evolution", "convention", "duration_s", "dt_s", "pm_ode_rtol", "pm_ode_phase_atol")
        if any(value.get(key) != old.get(key) for key in keys):
            raise RuntimeError("Regenerated injection physical configuration differs")
        identical = (value["signal_sha256"] == old["signal_sha256"]
                     and value["signal_array_sha256"] == old["signal_array_sha256"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        history = folder / "input_regeneration_history" / stamp
        history.mkdir(parents=True, exist_ok=False)
        (history / "original_source_metadata.json").write_bytes(old_bytes)
        save(history / "regeneration.json", dict(bitwise_identical_input=identical,
            previous_metadata_sha256=hashlib.sha256(old_bytes).hexdigest(),
            regenerated_metadata=value, completed_basin_checkpoint_present=has_checkpoint,
            action="preserve_original_metadata_bytes" if identical else
                   "refuse_changed_checkpoint_input" if has_checkpoint else "fresh_search_without_completed_checkpoint"))
        if identical:
            return  # Original timing remains historical; new timing is in history.
        if has_checkpoint:
            raise RuntimeError("Regenerated waveform differs from checkpoint input; a fresh search requires review")
        return original_save(path, value)

    production.save = guarded_save
    try:
        yield
    finally:
        production.save = original_save
