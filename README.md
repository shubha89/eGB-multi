# eGB-multi

Eccentric Galactic-binary waveforms and time-domain LISA responses. The package
provides Newtonian and conservative 1PN orbital dynamics, leading
Peters–Mathews evolution, six retarded Doppler links, and pyTDI Michelson XYZ/AET.
JAX link generation and reusable geometry support repeated source evaluations.
STRIDE is included in this repository and installation: its `lisa_stride`
interface evaluates sampled polarizations, six links and ordered TDI with JAX.
An optional Numba CPU backend and a fastlisaresponse example are also provided.

## Installation

Use Python 3.12 or newer:

```sh
python -m pip install ".[orbits,analysis]"
python examples/smoke_xyz.py
python examples/egb_stride.py
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

Time-domain eccentric waveforms include orbit-averaged Peters–Mathews evolution by default.
Use `evolution_mode="fixed"` to hold the intrinsic orbital elements fixed.
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

For STRIDE, [the eGB example](examples/egb_stride.py) uses the existing retarded
orbit geometry and includes the source polarization angle once. Its detector
cache is reusable when intrinsic or sky parameters change; sky-dependent
quantities are recomputed on every call.

```sh
python -m pip install ".[orbits,cpu]"
python examples/egb_stride.py --backend jax --source-backend jax
python examples/egb_stride.py --backend numba
python examples/egb_stride.py --evolution fixed --source-backend jax
```

The [STRIDE guide](docs/stride.md) describes array interfaces, geometry, ordered
delays, precision, batching, streaming and cache ownership. Import directly
from `lisa_stride`, or use the eGB adapters in `egb_multi.stride`.
The historical `xtdi` namespace remains available.

For annual PM-on waveforms, the [cadence and batching guide](docs/annual_cadence.md)
shows how to choose a sampling interval, reuse a fixed sky, and evaluate proposals
with independent directions on a common grid. The [adaptive cadence notebook](examples/notebooks/adaptive_cadence.ipynb)
generates a full year of XYZ, compares a day with a finer reference, and batches
proposals at shared and different skies. To run it from this checkout:

```sh
python -m pip install -e ".[notebook]"
jupyter lab examples/notebooks/adaptive_cadence.ipynb
```

To use the optional upstream fastlisaresponse engine:

```sh
python -m pip install ".[fastlisa]"
python examples/egb_fastlisaresponse.py --compare-stride
```

The [fastlisaresponse guide](docs/fastlisaresponse.md) explains CPU/CUDA
installation, polarization and output conventions, padding and orbit support.
Both examples default to 4096 retained samples and print a scalar summary.
Use `--output outputs/example.npz` to save their waveform.

The executed [STRIDE and fastlisaresponse usage notebook](examples/notebooks/stride_fastlisaresponse_usage.ipynb)
walks through source polarizations, retarded STRIDE XYZ, detector-cache reuse,
and a same-input native comparison, with waveform and residual plots:

```sh
python -m pip install ".[orbits,notebook,fastlisa]"
jupyter lab examples/notebooks/stride_fastlisaresponse_usage.ipynb
```

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
The full STRIDE reference suite is under `tests/stride/`; install
`.[test,cpu,orbits]` and run `python -m pytest tests/stride`.

The paper's data are distributed separately. [The paper guide](tests/paper/README.md)
documents restoration and figure reproduction. No PDFs, manuscript drafts or
referee correspondence are included in the source tree or release archives.

Software author: Shubhanshu Tiwari. See [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE).
