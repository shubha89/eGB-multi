# Task 2.9: Euler submission steps

These commands are instructions for the user. The agent has not submitted a job. Use the current 155.4 MB archive with transient waveform storage. Submit two platform pilots before the remaining production grid.

## 1. Transfer from the Mac

In a local Mac terminal:

```sh
scp "/Users/shubhanshutiwari/IdeaProjects/eGB-multi/outputs/referee/task_2_9_20261013_transfer_transient/task_2_9_euler.tar" stiwar@euler.ethz.ch:/cluster/home/stiwar/task_2_9_euler_transient.tar
```

## 2. Verify and extract on Euler

In the existing Euler login terminal:

```sh
export EGB_ROOT=/cluster/scratch/stiwar/egb-referee-20261013
export EGB_CAMPAIGN="$EGB_ROOT/outputs/referee/task_2_9_20261012_production"

bash -e <<'EXTRACT'
printf '%s  %s\n' 'ce4ff805aed2c66bdcf6ed1e85b055797e8beefbb7376c2447880f6c0b8ce2e1' '/cluster/home/stiwar/task_2_9_euler_transient.tar' | sha256sum -c -
mkdir "$EGB_ROOT"
tar -xf /cluster/home/stiwar/task_2_9_euler_transient.tar -C "$EGB_ROOT"
/cluster/home/stiwar/miniforge3/envs/egb-multi/bin/python "$EGB_ROOT/benchmarks/referee_task_2_slurm.py" verify-transfer --manifest "$EGB_ROOT/outputs/referee/task_2_9_20261013_transfer_transient/transfer_manifest.json"
EXTRACT
```

Expect the archive checksum to report `OK`, then `transfer_hashes_verified`. Stop if either fails. If the destination already exists, inspect it before repeating extraction; do not overwrite a working campaign.

## 3. Create the reference environment in a compute allocation

This command requests an interactive compute allocation. [ETH's interactive-job instructions](https://docs.hpc.ethz.ch/tutorials/first-job/).

```sh
srun --ntasks=1 --cpus-per-task=1 --mem-per-cpu=8G --time=01:00:00 --pty bash
```

One CPU is sufficient for environment setup; this requests 8 GB in total. The earlier two-CPU setup request produced a reported `2 > 1` allocation/step warning on Euler and was cancelled. No inherited job ID or CPU variables appeared in the subsequent shell probe; the cause of that earlier warning remains unconfirmed. Wait for the compute-node shell before continuing. This setup request is separate from the scientific job array.

The replacement request was confirmed running in user-provided Slurm output: job `16990851`, node `eu-a2p-531`, one CPU and 8 GB, with a one-hour allocation starting 2026-10-13 at 17:39:52. This confirms allocation success, not environment installation or scientific validation.

Once on the compute node, confirm the allocation:

```sh
hostname
scontrol show job "$SLURM_JOB_ID"
```

Then run:

```sh
export EGB_ROOT=/cluster/scratch/stiwar/egb-referee-20261013
export EGB_CAMPAIGN="$EGB_ROOT/outputs/referee/task_2_9_20261012_production"
module load eth_proxy
bash -e <<'SETUP'
test -f "$EGB_CAMPAIGN/parallel/reference_requirements.txt"
source /cluster/home/stiwar/miniforge3/etc/profile.d/conda.sh
test ! -e /cluster/home/stiwar/miniforge3/envs/egb-referee
conda create -n egb-referee --override-channels -c conda-forge python=3.13 pip -y
conda activate egb-referee
test "$CONDA_PREFIX" = /cluster/home/stiwar/miniforge3/envs/egb-referee
python -m pip install -r "$EGB_CAMPAIGN/parallel/reference_requirements.txt" pytest
python -m pip check
export PYTHONPATH="$EGB_ROOT/src:$EGB_ROOT/benchmarks:$EGB_ROOT/outputs/referee/.precision_deps"
python -m pytest -q "$EGB_ROOT/tests/test_referee_slurm_orchestration.py" "$EGB_ROOT/tests/test_referee_waveform_cache.py"
python "$EGB_ROOT/benchmarks/referee_task_2_slurm.py" preflight
SETUP
```

The new environment is stored under Home, consistent with [ETH's Conda storage guidance](https://docs.hpc.ethz.ch/software/package-managers/conda/). This installs the recorded scientific package versions and uses Python 3.13, matching the Mac reference's Python series. Linux dependency resolution/imports still need to succeed; do not silently substitute unavailable package versions. If the environment already exists, the guard stops setup for inspection. A successful setup reports no broken requirements, 16 passing tests and `portable_input_and_import_checks_passed`. No scientific fitting is run in this setup step.

After successful setup, leave the interactive allocation:

```sh
exit
```

## 4. Restore paths on the login node

```sh
export EGB_ROOT=/cluster/scratch/stiwar/egb-referee-20261013
export EGB_CAMPAIGN="$EGB_ROOT/outputs/referee/task_2_9_20261012_production"
export EGB_PYTHON=/cluster/home/stiwar/miniforge3/envs/egb-referee/bin/python
export PYTHONPATH="$EGB_ROOT/src:$EGB_ROOT/benchmarks:$EGB_ROOT/outputs/referee/.precision_deps"
cd "$EGB_ROOT"
```

## 5. Submit only the two platform pilots

This is the first scientific job submission:

```sh
export EGB_PILOT_PLAN="$EGB_CAMPAIGN/parallel/plans/20261013T104755787403Z_pilots"
bash "$EGB_PILOT_PLAN/submit.sh"
cat "$EGB_PILOT_PLAN/submitted_jobs.txt"
```

The pilot array runs two points. Initial requests are one CPU, 8 GB and 24 hours per point. The script records its job ID. Submit this plan once. Arrays and job dependencies are described in [ETH's Slurm guide](https://docs.hpc.ethz.ch/batchsystem/slurm/).

User-provided queue/accounting output confirmed pilot array `16991673`, tasks `0-1` with a concurrency limit of two. Its reported state at that observation was `PENDING`, with zero elapsed time; execution, environment checks and platform agreement are not established by that queue output. Do not resubmit this array while it is queued.

A later user-provided log excerpt reported `Native injection complete 1149.8 s` and `production selected 4 annual basins`. This establishes progress into the search stage for at least one pilot; the excerpt does not identify the task index or establish completion of either pilot.

For runtime planning, the recorded Mac pilots used 373.94 s for injection plus 2413.73 s for search/validation in the circular case (about 46 minutes combined), and 380.24 s plus 8958.05 s in the e=0.1 case (about 2 hours 36 minutes combined). These combined times omit small wrapper/cleanup overheads. The Euler injection excerpt is approximately three times slower than either reference injection. If the search stages scale similarly, combined Euler runtimes would be approximately 2.3-2.4 hours and 7.8-8.0 hours. Working planning ranges are therefore 2-3 hours for the circular pilot and 6-9 hours for the eccentric pilot, measured from job start. These are heuristic ranges, not statistical intervals or calibrated Euler search forecasts; fresh optimizer progress or completed pilots are needed to refine them. The two jobs run concurrently, and their 24-hour time limits are resource requests rather than completion estimates.

Performance diagnosis remains open. Inspection confirmed that both the Mac production launcher and the Euler batch wrapper set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to one. These settings do not by themselves prove identical affinity or constrain every JAX runtime thread. The current Mac CPU was verified as Apple M4 (10 physical/logical cores), and its NumPy 2.4.6 build reports Apple Accelerate. The native Peters-Mathews injection evaluates evolved links through the NumPy source-model path, although JAX is initialized by the adapter. Its timing also includes NPZ compression and writing, so the measured threefold difference is not a pure CPU benchmark. Per-core performance, library builds, and file I/O are hypotheses until worker utilization, node hardware and phase timings are inspected. Do not change resource settings or numerical code in the already submitted, hashed pilot plan.

The same excerpt included a lisaconstants/Astropy warning for `VACUUM_PERMEABILITY` and a JAX CPU fallback notice. The Astropy warning was reproduced locally using the frozen reference versions, lisaconstants 2.0.2 and Astropy 8.0.0: the LISA value is 1.25663706143592e-6 and Astropy's value is 1.25663706127e-6, a relative difference of approximately 1.32e-10. Source inspection found no references to this constant or `mu0` in `src/egb_multi`, the production worker, annual search/refinement, or the installed lisaorbits/pyTDI packages. This is a scoped audit of the reference paths, not independent verification of the Euler installation. The warning alone does not warrant changing the pinned environment during a run. JAX CPU execution matches the CPU job configuration. Both pilot point controls and platform agreement still need to pass before production.

## 6. Monitor and check pilot results

```sh
squeue -u stiwar
sacct --starttime=2026-10-13 -j "$(cat "$EGB_PILOT_PLAN/submitted_jobs.txt")" --format=JobID%22,State%20,ExitCode,Elapsed,TotalCPU,AllocCPUS,MaxRSS
tail -n 30 "$EGB_PILOT_PLAN"/logs/*.out
```

On 14 October the user reported `Invalid job id specified` from `squeue` for array `16991673`. Its final outcome is not yet known. A job leaving the active queue is not scientific success or failure evidence; use the accounting command above, with the submission date explicitly included, and inspect both platform-agreement records and worker logs before proceeding.

A subsequent user-provided tail from `16991673_1.out` confirms `Production point passed p_06d9d5194f47b99a FF 0.8762605296975234`, after the four-basin eccentric search and native candidate verification. The saved Mac result is 0.8762605297090141, giving an absolute difference of 1.1490697282567908e-11, below the 1e-4 platform allocation. This is a comparison against the reported log value. The circular pilot, both official platform-agreement records, final scheduler states and resource measurements remain to be inspected; the excerpt alone does not establish that the full platform gate is complete.

Once both finish:

```sh
cat "$EGB_CAMPAIGN"/parallel/platform_validation/points/*/platform_agreement.json
```

Both must report `passed_platform_agreement`, with absolute FF differences at most `0.0001`; their recorded point controls must pass. Review Slurm runtime/MaxRSS for the worker allocation before production. If a pilot fails or times out, stop and inspect its log; do not relax the scientific tolerance. Resource-request changes require a fresh plan because the batch script's hash is sealed.

## 7. Generate and submit the production queue

Only after the pilot checks pass:

```sh
cd "$EGB_ROOT"
"$EGB_PYTHON" benchmarks/referee_task_2_slurm.py plan --mode production --concurrency 32 > production_plan.json
export EGB_PROD_PLAN=$("$EGB_PYTHON" -c 'import json; print(json.load(open("production_plan.json"))["plan_directory"])')
bash "$EGB_PROD_PLAN/submit.sh"
cat "$EGB_PROD_PLAN/submitted_jobs.txt"
squeue -u stiwar
```

There are 2,152 pending base-grid points in the prepared campaign; the eight accepted local results are retained. The planner rechecks the platform gate. Chunks of at most 500 jobs are chained with `afterok`, so the cap is 32 concurrent points across the campaign. Temporary waveform files are removed by the wrapper after normal/caught-interrupt exits; hashes/results/diagnostics remain. A partial submission failure must be inspected using the recorded job IDs before retrying, so already queued work is not duplicated.

## 8. Back up results during the run

Personal Scratch deletes old files and is not backed up; [ETH's rules specify 15 days](https://docs.hpc.ethz.ch/hardware/storage/). Keep the source archive in Home and copy new campaign outputs regularly to durable storage. In a Mac terminal:

```sh
mkdir -p /Users/shubhanshutiwari/IdeaProjects/eGB-multi/outputs/referee/euler_backup_task_2_9
rsync -av --exclude='production_signal*.npz' --exclude='__pycache__/' --exclude='*.tmp' --exclude='.cache/' --exclude='.mplconfig/' stiwar@euler.ethz.ch:/cluster/scratch/stiwar/egb-referee-20261013/ /Users/shubhanshutiwari/IdeaProjects/eGB-multi/outputs/referee/euler_backup_task_2_9/
```

This preserves code, references and scientific diagnostics without copying transient waveform arrays. It uses no deletion option and does not change Scratch timestamps to bypass cleanup. Monitor storage with `lquota`.

## 9. Continue to contour refinement

After the submitted queue has finished, on Euler:

```sh
cd "$EGB_ROOT"
"$EGB_PYTHON" benchmarks/referee_task_2_slurm.py collect --advance-contours
```

If the returned `pending_points` is positive because crossing-refinement points were added, repeat Step 7 with a new plan. Do not rerun a previously submitted script. If `pending_points` is zero and `production_grid_complete` is true, the calculation has reached its grid/bracket endpoint; scientific precision review and figure/table regeneration remain before Task 2.9 is closed. Any failed point control requires investigation.
