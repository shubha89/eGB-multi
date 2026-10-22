# Optional model comparisons

These programs call the installed eGB-multi library and the declared external
model. They are excluded from default pytest discovery and library installation.
Their numerical physics follows the recorded comparison drivers; directory and
helper-import changes do not establish a new comparison result.

| Program | External model and output |
| --- | --- |
| `jaxgb_check.py` | JAXGB: circular source, six links and explicitly labelled native-TDI comparisons |
| `annual_circular.py` | JAXGB: annual A/E comparison using the restored paper covariance |
| `pyefpe_check.py` | pyEFPE: shared-leading-physics and different-PN source comparisons |
| `pyefpe_fd_control.py` | pyEFPE: native frequency-domain full-inspiral functionality study |

Use a separate optional environment. Frozen upstream identities are in
`external_models.json`:

```sh
python -m pip install 'jaxgb @ git+https://gitlab.com/lisamission/jaxgb.git@89d9d94f0f30c2c6d68dfdfffa4e40bbd52c6eae'
git clone https://github.com/gmorras/pyEFPE.git vendor-pyEFPE
git -C vendor-pyEFPE checkout 5b6964c0002893083c69f3ccdd687eb8b5d60e76
python -m pip install ./vendor-pyEFPE
```

Install only the model needed for the selected comparison. The pyEFPE programs
do not import JAXGB. Helpers shared by the comparisons live in
`comparison_utils.py` and import neither external model. Model-specific programs
currently import their selected external package at startup, including for
`--help`.

For example, inspect a source-comparison command without starting a run:

```sh
python tests/comparisons/pyefpe_check.py --help
```

JAXGB conversion uses `f0=mean_motion/pi`, `i_J=pi-i_eGB`,
`phi_J=pi-2*phi_eGB`, the stated analytic amplitude and
`y_eGB=-y_JAXGB`. Shared-source/link agreement does not imply identical native
TDI implementations. The pyEFPE release exposes a frequency-domain interface;
`release_time_domain` explicitly reconstructs its own harmonic amplitudes and
phase interpolants. It is not a native time-domain API. SI masses/distance,
`phi_start=phi_eGB+pi/2`, and unfitted polarizations are preserved.

The annual circular program requires the restored paper data. Source comparisons
and the native FD study can be expensive; they are never started by installation,
pytest, data verification or saved-figure replay. Updated annual comparisons have
not been rerun as part of this directory cleanup.
