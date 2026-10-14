# Task 2.9 production campaign

**Current status, 13 October:** the local campaign is stopped; eight completed results and the partial ninth-point evidence are preserved. The [parallel transfer and submission workflow](referee_task_2_9_parallel_execution.md) is prepared; no Euler jobs have been submitted. The original launch observation below is retained as history.

Launched on **12 October 2026 at 20:42 Europe/Zurich (18:42 UTC)** following the user's instruction to launch the compute-intensive task. This supersedes the earlier proposed start time. The production calculation is running; Task 2.9 is not complete.

The [driver](../benchmarks/referee_task_2_production.py) runs from this workspace and verifies that every eGB module is loaded from `src/egb_multi`. The [manifest](../outputs/referee/task_2_9_20261012_production/manifest.json) preserves the source snapshots, input hashes, runtime settings and the reviewed Task 2.8 decision. This directory has no Git metadata, so no commit identifier is assigned.

## Scientific calculation

- Original grid preserved: **2,160 points**, with six carrier frequencies, ten eccentricities, three inclinations, four sky positions and three mass pairs. The axes were checked against the original production script.
- One year, **2.5 s cadence**, **10 µHz–80 mHz** band, native retarded response, second-generation TDI with measurement order 101, 32,768 circular-envelope intervals and tightened Peters–Mathews evolution controls.
- Full Hermitian A/E covariance from the frozen annual moving-secondary and primary-laser reference; ideal clocks/ranges and no Galactic foreground. The assembled covariance passed finite, Hermitian and positive-definite checks over the analysis band.
- Circular frequency, chirp, full sky and both ordered component masses remain free. A common physical source supplies the coherent A/E orientation fit. The nine historical derivative candidates are retained under the current metric.
- Native winner agreement, final frequency/chirp boundaries, completed basins, independent restarts and retained-candidate comparisons are checked at every point. New winning harmonic branches and near-contour cases receive six additional blind starts. A failed control stops the queue for investigation and preserves the evidence.
- After the base grid, the controller adds shared eccentricity samples to refine the **median FF = 0.9** crossings, separately by mass pair and across all mass pairs. It brackets the actual medians rather than averaging individual crossings. The target bracket width is 0.001 in eccentricity; numerical FF uncertainty still needs review before reporting precision.

These are finite empirical searches under the adopted idealized reference. The calculation does not provide a theorem of global maximization or a complete time-dependent mission likelihood.

## Execution and monitoring

One worker runs at a time on the 16 GB machine. The first circular and eccentric points are members of the original grid and serve as resource/workflow pilots; the remaining queue continues automatically when they pass. Generated native injections and search evidence are retained, with a stop below 25 GiB free disk. The detached controller uses `caffeinate` to inhibit idle sleep while it is alive.

At the initial live check, controller **29048**, worker **29051** and `caffeinate` **29049** were alive. The worker was generating the annual circular pilot, used approximately 1.4 GB resident memory and was actively using CPU. **0/2,160 points had completed.** This is a launch observation, not a statement that all pilot/search controls have passed.

- [Current work state](../outputs/referee/task_2_9_20261012_production/work_state.json)
- [Launch record](../outputs/referee/task_2_9_20261012_production/launch.json)
- [Controller log](../outputs/referee/task_2_9_20261012_production/controller.log)
- [First pilot log](../outputs/referee/task_2_9_20261012_production/points/p_9ff7f1012ff0a89a/run.log)
- [Grid definition](../outputs/referee/task_2_9_20261012_production/grid_definition.json)

The work state and logs are the live status; the PID/memory observation above may become stale. A reliable runtime estimate is not yet available. Early pilot estimates exclude injection, additional controls and harder searches and must not be treated as completion-time promises.

Launch or resume from the workspace after confirming that no controller/worker remains active:

```sh
PYTHONPATH=src:benchmarks:outputs/referee/.precision_deps \
MPLCONFIGDIR=outputs/referee/.mplconfig \
XDG_CACHE_HOME=outputs/referee/.cache \
/opt/miniconda3/bin/python benchmarks/referee_task_2_production.py --launch
```

Completed point results and completed-basin checkpoints are reused; interrupted native injection generation must be repeated. Input/source hash changes are refused. A stopped point with failed controls requires review rather than automatic acceptance.

## Remaining closure

After the grid and adaptive brackets finish, review the point controls and contour precision, compare original and revised results, regenerate Figure 7/Table I and the reference-noise Figure 4, and audit Figures 5, 6 and 8. The controller records `task_2_9_complete=false` even after the calculations finish until that review and figure/table work are completed. Manuscript edits and the final referee response remain deferred to the final stage, as requested.
