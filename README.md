# eGB-multi

Eccentric Galactic-binary waveforms and time-domain LISA responses. The package
provides Newtonian and conservative 1PN orbital dynamics, optional leading
Peters–Mathews evolution, six retarded Doppler links, and pyTDI Michelson XYZ/AET.
JAX link generation and reusable geometry support repeated source evaluations.

## Installation

Use Python 3.12 or newer:

```sh
python -m pip install ".[orbits,analysis]"
python examples/smoke_xyz.py
```

The library is installed from `src/`. Tests, model comparisons and research
drivers are optional and are grouped under [tests/](tests/README.md).

## Source waveforms

```python
import numpy as np
from egb_multi.eccentric import EccentricBinaryParams, eccentric_polarizations

source = EccentricBinaryParams(
    mean_motion=np.pi * 1e-3, eccentricity=0.1,
    m1_solar=0.6, m2_solar=0.4, inclination=0.8,
)
times = np.arange(4096) * 2.5
hplus, hcross = eccentric_polarizations(
    source, times,
    physics_mode="1pn_periastron",
    evolution_mode="peters_mathews",
)
```

Source physics and secular evolution are separate options. Fixed intrinsic
parameters, Newtonian/1PN orbital choices and partial Peters diagnostics are
available. The [model/API scope](docs/model_scope.md) describes the approximations,
units and backend-specific treatment of a prescribed frequency derivative.

## Detector responses and reusable geometry

[The short XYZ example](examples/smoke_xyz.py) generates a response using the
installed library. The [prepared-response notebook](examples/notebooks/minimal_prepared_xyz.ipynb)
and [one-year example](examples/notebooks/one_year_prepared_xyz.ipynb) show how to
reuse retarded geometry and pyTDI operators across sources and data blocks.
Rebuild these caches when the orbit, time grid or interpolation settings change.

The canonical import namespace is `egb_multi`. The historical
`egb_jax_eccentric` convenience interface remains available for compatibility.

## Repository layout

```text
src/       Library and compatibility interface
examples/  User examples and prepared-response notebooks
docs/      Model assumptions and API conventions
tests/     Unit tests, optional model comparisons, benchmarks and paper reproduction
```

To run the library tests, install `.[orbits,analysis,test,precision]` and run
`python -m pytest`. External comparisons and annual benchmarks run only when
explicitly invoked; see the instructions under `tests/`.

The paper's data are distributed separately. [The paper guide](tests/paper/README.md)
documents restoration and figure reproduction. No PDFs, manuscript drafts or
referee correspondence are included in the source tree or release archives.

Software author: Shubhanshu Tiwari. See [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE).
