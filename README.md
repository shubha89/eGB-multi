# eGB-multi

Standalone JAX eccentric compact-binary response model with switchable source
physics and pyTDI helpers.

This repo focuses on:

- switchable eccentric source physics: `newtonian`, `1pn_no_periastron`, `1pn_periastron`,
- separate fixed and Peters-Mathews source evolution modes,
- batched JAX source strain and six-link LISA response generation,
- optional interpolation and harmonic-carrier link paths,
- pyTDI conversion from six GW-only links to Michelson `X/Y/Z`,
- orthogonal `A/E/T` conversion.

The circular trilinear tensor and FastGB benchmarking code are intentionally not
included here.

## Install

```bash
python -m pip install -e ".[test,orbits]"
```

Or create the documented Conda environment:

```bash
conda env create -f environment.yml
conda activate egb-multi
```
OR simply ask your AI agent to install this ;) 

## Minimal Use

```python
import numpy as np

from egb_jax_eccentric import (
    EccentricBinaryParams,
    eccentric_xyz_jax,
    lisa_orbit,
)

times = np.arange(512.0)
state = lisa_orbit(times)
source = EccentricBinaryParams(
    mean_motion=np.pi * 1.0e-4,
    eccentricity=0.05,
    beta=0.2,
    lambda_=0.4,
    psi=0.3,
    inclination=0.8,
)

xyz = eccentric_xyz_jax(
    state,
    source,
    batch_size=1,
    physics_mode="1pn_periastron",
)
print({channel: value.shape for channel, value in xyz.items()})
```

## Physics And Evolution Switches

Use `physics_mode` to control source physics in the NumPy and JAX exact-link
paths:

- `newtonian`: Newtonian orbital dynamics, no periastron advance.
- `1pn_no_periastron`: 1PN orbital corrections without periastron advance.
- `1pn_periastron`: 1PN orbital corrections with periastron advance. The
  historical `1pn` spelling is still accepted as an alias.

Use `evolution_mode` in the NumPy source generator to control secular
evolution:

- `fixed`: hold the source parameters fixed.
- `peters_mathews` or `pm`: evolve radial mean motion, eccentricity, and mean
  anomaly with Peters-Mathews radiation reaction.
- `peters_mathews_orbital_only` and `peters_mathews_eccentricity_only`: partial
  diagnostics that isolate the two secular pieces.
- `auto` or `adaptive`: compare against `1pn` plus full Peters-Mathews
  evolution and switch from `fixed` to `peters_mathews` once the requested
  source-level mismatch tolerance is reached.

For survey runs, calibrate the automatic switch on a representative grid once,
then reuse the rule:

```python
from egb_jax_eccentric import calibrate_peters_mathews_switch_rule, eccentric_complex_strain

rule = calibrate_peters_mathews_switch_rule(
    calibration_sources,
    calibration_times,
    mismatch_tolerance=1.0e-2,
)

h = eccentric_complex_strain(
    source,
    times,
    physics_mode="1pn_periastron",
    evolution_mode="auto",
    pm_mismatch_tolerance=1.0e-2,
    pm_switch_rule=rule,
)
```

If no `pm_switch_rule` is supplied, `evolution_mode="auto"` computes the
fixed-vs-full-PM mismatch directly for that source and time grid.

## Reusing TDI geometry

For repeated GW-only evaluations on the same orbit and time grid, prepare the
factorized pyTDI operators once:

```python
from egb_jax_eccentric import (
    eccentric_links_jax, precompute_jax_link_geometry, prepare_xyz_from_links,
)

geometry = precompute_jax_link_geometry(state)
tdi = prepare_xyz_from_links(state, generation=2)
for source in sources:
    links = eccentric_links_jax(source, geometry, batch_size=1)
    xyz = tdi(links)
```

This caches nested delays and their Doppler factors; source-dependent
measurement interpolation still runs for every call. Rebuild when the time
grid, spacecraft positions, or interpolation settings change. Large grids
require substantial cache memory; process sources on a shared block when
working with long observations. This API is for GW-only links, not full
instrument beatnotes. The ordinary `xyz_from_links` path also avoids
interpolating zero reference/metrology channels.

See [the measured TDI optimization results](benchmarks/results/tdi_optimization/REPORT.md).

Examples: [minimal prepared waveform](notebooks/minimal_prepared_xyz.ipynb)
and [one year with shared block caches](notebooks/one_year_prepared_xyz.ipynb).
The one-year example retains two independent XYZ waveforms and reports shared
preparation separately from waveform evaluation.

## Tests

```bash
python -m pytest
```

## Notebook

Open the switchable eccentric model example with:

```bash
jupyter lab notebooks/eccentric_pytdi_switches.ipynb
```
