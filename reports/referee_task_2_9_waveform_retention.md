# Task 2.9: no permanent waveform catalogue

On 13 October the user requested that the production waveforms not be saved. The prepared Euler workflow now uses **transient waveform caches**. The validated waveform, TDI, noise covariance and fitting-factor search code remain unchanged; the change affects storage and checkpoint recovery.

## What remains on disk

Preserve the source parameters, physical/numerical conventions, native array and input-file hashes, package/source versions, covariance, fitting factors, winning templates, optimizer histories, numerical controls and completed-basin checkpoints. The covariance is a shared reference input and is retained. Circular template arrays are not written as a catalogue.

The existing file-based annual search needs a temporary `production_signal.npz` while a point is active. The Slurm wrapper removes that file and any incomplete `production_signal.tmp.npz` after normal completion, failure or a caught termination signal. It records the discarded file's digest and size. It never removes the source metadata, results, covariance or other arrays. A hard kill/OOM can prevent Python cleanup and leave a temporary cache/lock; check the stopped job before recovering it. There is no claim that the legacy search is wholly diskless.

This policy applies to the new Slurm entry point. The original reference production driver is preserved with its original hash. Its direct serial entry point does not implement the new cleanup policy. No existing local baseline arrays or referee-validation artifacts were deleted; the new transfer omits those reproducible intermediates.

## Checkpoint recovery

An interrupted worker regenerates a missing waveform using the saved parameters and frozen numerical settings. To reuse a completed-basin checkpoint, the regenerated arrays and NPZ digest must match the original input exactly. The wrapper preserves the original source-metadata bytes so that the legacy manifest checks still pass; the new generation timing is recorded separately.

A changed regenerated input is refused when a basin checkpoint exists. If no completed checkpoint exists, a fresh search is permitted, with the old metadata archived and the changed input explicitly recorded. This matters when moving an incomplete Mac run to Linux. Different-platform regeneration is not assumed to be bitwise identical, and the two Euler fitting-factor pilot comparisons are still required.

## Verification and measured storage

- **16 tests passed** across scheduling, preservation, cleanup, interruption, regeneration and transfer exclusion. Cleanup only targets the two generated waveform filenames and refuses symlinked inputs.
- A **six-hour, 8,640-sample native PM/TDI2 control** at 2.5 s cadence regenerated A/E arrays and the compressed NPZ bitwise identically, preserved original metadata bytes and left no waveform files. [Control](../outputs/referee/task_2_9_20261013_waveform_cache_control/validation_summary.json). This is a short regeneration/serialization check, not annual fitting-factor or Euler validation.
- The new archive is **155,412,480 bytes (155.4 MB)**, replacing the earlier 1.88 GB transfer for future setup. Nine waveform files totalling **1,727,129,287 bytes** are omitted, with their digests recorded. Source metadata, diagnostics and all eight accepted result records remain.
- Extraction into another workspace verified **615 sealed files**, zero frozen production-input hash errors and zero waveform files. [Transfer verification](../outputs/referee/task_2_9_20261013_transfer_transient/portable_extraction_check.json).

The old permanent-waveform estimate of roughly 430 GB for the base grid no longer applies. At approximately 200 MB per active waveform, 32 workers require roughly **6–7 GB of temporary waveform storage**, in addition to reference data and growing diagnostics. This is a waveform-cache estimate, not a total campaign disk-space guarantee. Dense seed/evaluation histories can still become large and must be monitored.

Regeneration costs a few minutes for the observed annual cases; interrupted jobs may need to repeat it. The main fitting-factor search cost remains. This change does not replace parallel execution or its numerical safeguards.

## Active artifacts

- [Storage policy](../outputs/referee/task_2_9_20261012_production/parallel/waveform_retention.json)
- [Cache helper](../benchmarks/referee_task_2_waveform_cache.py)
- [Slurm wrapper](../benchmarks/referee_task_2_slurm.py)
- [Current pilot plan](../outputs/referee/task_2_9_20261012_production/parallel/plans/20261013T104755787403Z_pilots/plan.json)
- [New transfer archive](../outputs/referee/task_2_9_20261013_transfer_transient/task_2_9_euler.tar)
- [Archive checksum](../outputs/referee/task_2_9_20261013_transfer_transient/archive_check.json)
- [Transfer manifest, including omitted waveforms](../outputs/referee/task_2_9_20261013_transfer_transient/transfer_manifest.json)

The old transfer archive and earlier plans remain historical records. Their source/policy hashes differ from the new entry point; use the current transfer and generate future plans with the current wrapper. **No Euler jobs were submitted, and the user instruction to hold submissions remains active.**
