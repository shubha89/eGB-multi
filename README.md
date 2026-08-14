# eGB-multi

Standalone JAX eccentric compact-binary response model with switchable source
physics and pyTDI helpers.

This repo focuses on:

- switchable eccentric source physics: `newtonian`, `1pn_no_periastron`, `1pn`,
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

## Tests

```bash
python -m pytest
```

## Notebook

Open the switchable eccentric model example with:

```bash
jupyter lab notebooks/eccentric_pytdi_switches.ipynb
```
