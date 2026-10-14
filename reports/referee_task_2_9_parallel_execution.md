# Task 2.9: local stop and parallel execution

**Storage update, 13 October:** the current [transient-waveform workflow](referee_task_2_9_waveform_retention.md) removes temporary waveform arrays after each worker and preserves parameters, hashes, results and diagnostics. The active transfer is **155.4 MB**; it supersedes the earlier 1.88 GB archive and earlier pilot plan. Numerical settings remain unchanged. Job submissions remain on hold.

The local controller and worker were stopped on **13 October 2026 at 10:47 Europe/Zurich**, after the user agreed that the serial throughput was impractical. The [stop record](../outputs/referee/task_2_9_20261012_production/local_stop_record.json) verifies that **all eight completed result files retain their original SHA-256 hashes**. The ninth point's native injection and partial optimization logs remain available; it had not completed a basin checkpoint when stopped, so its incomplete search will restart while reusing the saved injection. No result has been deleted or relabelled as complete.

The completed searches averaged **5,637 s (1.57 hours) per point**. Multiplying that early average by 2,160 gives **141 days of serial search time**, excluding injection generation, extra controls and contour refinement. The eight early points do not establish the runtime distribution of the full grid. Equal-throughput arithmetic would divide the search time by the concurrency, but cluster CPU speed, scheduling, difficult cases and additional controls must be measured before promising a finish date.

## Prepared implementation

The [Slurm wrapper](../benchmarks/referee_task_2_slurm.py) and [array script](../benchmarks/referee_task_2_array.sbatch) call the **unchanged frozen production worker**. They preserve the one-year observation, 2.5 s cadence, physical band, full Hermitian A/E covariance, free sky/masses/frequency/chirp and all point controls. Changing scheduling does not change the fitting-factor definition or accept a failed point.

Two fresh reruns of the original circular and eccentric pilots are required on the destination platform. Each must pass all existing point controls and agree with its saved Mac fitting factor to **1e-4**, the existing optimization allocation. These checks establish selected platform agreement; they are not a uniform physics/global-search certificate. Production planning is refused before the pilot evidence is present and valid for the current platform.

The proposed production plan uses **one job per pending point**, at most **32 concurrent jobs** across the campaign. Array chunks contain at most 500 elements and run sequentially via `afterok` dependencies, avoiding multiplication of the concurrency cap across chunks. Initial resource requests are **one CPU, 8 GB and 24 hours per point**, to be reviewed after actual Euler memory/runtime measurements. No GPU, node type or partition is selected. Euler supports array jobs and dependency/resource requests; these are documented in [ETH's Slurm guide](https://docs.hpc.ethz.ch/batchsystem/slurm/).

Duplicate point execution is refused by an atomic directory lock. Normal termination releases the lock while preserving results and checkpoints and removing generated waveform caches under the current policy; a hard-killed job can leave a lock that requires checking the job's status before manual recovery. Existing failed controls stop preparation/collection for investigation. Already accepted points are skipped. Contour refinement uses the original median-crossing calculation after all current queued points have finished.

## Verified locally

- **16 scheduling/retention tests passed**: accepted-point reuse without modification, failed-point refusal, complete control sets, duplicate queue rejection, duplicate worker locking/interruption, required platform pilots, portable submission scripts and corruption detection.
- Frozen production input/source hashes and local eGB imports passed the preflight. **2,152 base-grid points remain pending**.
- Shell syntax checks passed. No frozen eGB numerical module or original production/search driver was modified.
- The active **155.4 MB transfer archive** was extracted into a different workspace root and verified: **615 sealed files, zero production-input hash errors and zero waveform files**. Nine generated arrays are intentionally omitted with digests retained. A short native regeneration/cleanup control also passed; see the [retention note](referee_task_2_9_waveform_retention.md). This is portability/retention verification, not an Euler scientific run. The [archive checksum](../outputs/referee/task_2_9_20261013_transfer_transient/archive_check.json), [transfer manifest](../outputs/referee/task_2_9_20261013_transfer_transient/transfer_manifest.json) and [extraction check](../outputs/referee/task_2_9_20261013_transfer_transient/portable_extraction_check.json) are saved alongside [the archive](../outputs/referee/task_2_9_20261013_transfer_transient/task_2_9_euler.tar).
- Scientific destination-platform reruns, actual Slurm submission, resource measurements and cluster throughput have **not** been verified.

The local referee tree uses `src/egb_multi`. The recorded earlier Euler installation used `egb_jax_eccentric`; its installation smoke tests do not validate this new production source. Transfer this exact snapshot and verify its environment rather than substituting an older checkout. The recorded Mac package versions are preserved in the campaign's `parallel/reference_environment.json` and `parallel/reference_requirements.txt`; destination resolution and pilot agreement remain necessary.

## Destination procedure

The transfer archive preserves the relative workspace layout, frozen reference files, signal-generation metadata/hashes, results and checkpoints. Generated waveform arrays are omitted under the transient policy. Absolute covariance symlinks are replaced by relative links in the archive, without modifying the local originals. Its transfer manifest seals regular-file hashes and link targets. Check the archive checksum and transfer manifest immediately after extracting into the chosen Euler working directory, before allowing outputs to change.

Run light setup/submission from the login node and the scientific calculations through Slurm. Choose storage for temporary active-point injections plus retained references/diagnostics. The transient policy avoids the roughly 430 GB permanent waveform catalogue; 32 active waveform caches occupy roughly 6–7 GB, while diagnostic growth still requires monitoring. Keep the Python/Conda environment separate from those large data outputs. Euler's available storage and cleanup rules are described in [ETH's storage guide](https://docs.hpc.ethz.ch/hardware/storage/).

From the root of the transferred workspace, set `EGB_PYTHON` to the absolute executable in the verified destination environment. The pilot submission script resolves the workspace paths there:

```sh
"$EGB_PYTHON" benchmarks/referee_task_2_slurm.py verify-transfer \
  --manifest outputs/referee/task_2_9_20261013_transfer_transient/transfer_manifest.json
"$EGB_PYTHON" benchmarks/referee_task_2_slurm.py preflight
```

**Do not run a submission script yet.** The current pilot plan is `parallel/plans/20261013T104755787403Z_pilots` under the campaign; its script is prepared for later authorization. After the user permits jobs, the two platform-agreement checks pass and measured resources are reviewed:

```sh
"$EGB_PYTHON" benchmarks/referee_task_2_slurm.py plan --mode production --concurrency 32
```

Submit the returned plan's `submit.sh` from the same workspace root. It records the submitted job IDs. Do not submit the same plan twice. Resource requests can be supplied to `sbatch` after review; changing the array script requires generating a fresh plan because its scheduling source hash is sealed.

After each complete queue round, collect and advance actual median-crossing brackets:

```sh
"$EGB_PYTHON" benchmarks/referee_task_2_slurm.py collect --advance-contours
```

If this adds points, generate and submit another production plan. Failed numerical controls require investigation, not acceptance or removal. The calculation remains Task 2.9 until point evidence, contour precision, figures and tables have been reviewed. Manuscript changes remain deferred to the final stage.

## Euler environment information supplied by the user

The user reported login host `eu-login-08`, home `/cluster/home/stiwar` and Python executable `/cluster/home/stiwar/miniforge3/envs/egb-multi/bin/python` (Python 3.12.15). These are terminal outputs supplied by the user; the agent has not obtained direct remote-shell access. The existing `/cluster/home/stiwar/eGB-multi` directory is present, but its expected `src` subdirectory was reported absent. Its contents and scientific source have not been inspected.

The [environment probe record](../outputs/referee/task_2_9_20261013_transfer/euler_environment_probe.json) preserves the reported versions and the comparison with the Mac reference. NumPy, SciPy, JAX/JAXlib, pyTDI, lisaorbits and Astropy differ; `mpmath` is missing from that environment. This does not demonstrate that the newer versions produce incorrect results. It prevents assuming the same runtime reference before comparison. The archive includes the existing portable `mpmath` copy under `.precision_deps`; a dedicated environment with the recorded reference package versions is the preferred preparation path. The reference packages' locally inspected `Requires-Python` metadata permits Python 3.12, but destination resolution/imports and scientific agreement still require verification.

Inspect the directory and storage quotas before choosing an extraction/data destination. Euler's documented standard Home quota is 50 GB, below the historical permanent-waveform estimate; the new transient policy removes that catalogue, but diagnostics still need storage monitoring; group Work access and quotas must be checked, and personal Scratch has an automatic deletion policy. Use `lquota` for the actual Home/Scratch usage and the relevant share path for Work, as documented in [ETH's storage guide](https://docs.hpc.ethz.ch/hardware/storage/).

The later directory listing shows a nested candidate checkout at `/cluster/home/stiwar/eGB-multi/eGB-multi`; its source layout is not yet inspected. The user's `lquota` output reports Home usage 4.00 GB with 45/50 GB soft/hard limits, and nearly empty personal Scratch with 2.50/2.70 TB soft/hard limits. Scratch has ample capacity for the estimated active waveform caches and room for diagnostics; the latter are not yet uniformly measured. The proposed isolated campaign directory is `/cluster/scratch/stiwar/egb-referee-20261013`, while the sealed transfer archive is planned at `/cluster/home/stiwar/task_2_9_euler_transient.tar`. These destinations are planned; creation/transfer have not been verified remotely.

The official personal Scratch rules delete files older than **15 days**, prohibit changing timestamps to bypass deletion, and provide no backup. Preserve the transfer archive and scientific source in durable storage and establish backup for newly produced outputs before authorizing the campaign. No timestamp-based retention workaround is proposed. The local reference archive and eight completed results remain intact.

**Current status: local computation stopped; 155.4 MB transient-waveform transfer prepared; account, Python path and adequate Scratch capacity reported. Remote transfer/hash verification, the dedicated environment and durable-output backup remain pending. The user explicitly instructed: do not submit jobs yet. No Euler job has been submitted.**
