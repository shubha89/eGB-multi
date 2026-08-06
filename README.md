# eGB-multi

Standalone JAX eccentric compact-binary response model with switchable source
physics and pyTDI helpers.

This repo focuses on:

- switchable eccentric source physics: `newtonian`, `1pn_no_periastron`, `1pn`,
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
    physics_mode="1pn",
)
print({channel: value.shape for channel, value in xyz.items()})
```

## Physics Switches

Use `physics_mode` to control source physics in the JAX exact-link path:

- `newtonian`: Newtonian orbital dynamics, no periastron advance.
- `1pn_no_periastron`: 1PN orbital corrections without periastron advance.
- `1pn`: 1PN orbital corrections with periastron advance.

## Tests

```bash
python -m pytest
```

## Notebook

Open the switchable eccentric model example with:

```bash
jupyter lab notebooks/eccentric_pytdi_switches.ipynb
```
