# eGB waveforms with fastlisaresponse

The executable example is [examples/egb_fastlisaresponse.py](../examples/egb_fastlisaresponse.py).
It uses the installed eGB source, upstream `pyResponseTDI`, and the bundled
STRIDE interface when `--compare-stride` is selected. It imports no research
driver and needs no private data.

The executed [usage notebook](../examples/notebooks/stride_fastlisaresponse_usage.ipynb)
shows the same public interfaces step by step, together with source/XYZ
plots, detector-cache reuse and a matched-geometry STRIDE comparison.

## Install and run on a CPU

From the repository root, using Python 3.12 or later:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[fastlisa]"
python examples/egb_fastlisaresponse.py
python examples/egb_fastlisaresponse.py --compare-stride --output outputs/fastlisa_example.npz
```

The native CPU binary and orbit interface were checked with
fastlisaresponse 1.1.17 and LISA Analysis Tools 1.2.0. Compiled-package support
and dependency constraints depend on the chosen platform; follow the
[upstream installation instructions](https://mikekatz04.github.io/lisa-on-gpu/)
when a wheel is unavailable. The optional dependency is not needed to import
eGB or STRIDE. Its upstream dependencies include Numba.

The default is one 4096-sample output at 2.5 s cadence, with conservative 1PN
orbital dynamics, periastron advance and Peters–Mathews evolution. The initial
`mean_motion / pi` is 3 mHz and eccentricity is 0.1. This frequency scale is
an orbital-parameter label; an eccentric source has multiple radiative
harmonics. For a fixed-element comparison:

```sh
python examples/egb_fastlisaresponse.py --evolution fixed --source-backend jax
```

## What the example computes

1. Prepare the upstream detector response, using its supplied equal-arm
   ephemeris by default. An HDF5 ephemeris can be supplied with
   `--orbit-file path/to/orbits.h5`. The table must use the LISA Analysis Tools
   schema, consistent directed links and the BME/ecliptic frame of the source.
2. Generate eGB's real plus/cross strain on a uniform grid, including 5000 s
   on each side of the retained detector record. The initial source parameters
   refer to source time zero; the first retained detector time is 5000 s.
   Preserve this epoch convention when changing initial conditions.
3. Convert the polarization tensors into the native response basis and form
   its required complex input `h_plus + 1j*h_cross`. Source `psi` is included
   exactly once in this conversion. eGB's own complex-strain convention is
   `h_plus - 1j*h_cross` in its source basis. The tensor conversion accounts
   for both that basis and a general polarization angle.
4. Call native `get_projections` and then `get_tdi_delays`, with order 25,
   second-generation TDI and XYZ output. The final global sign is converted
   to the eGB/pyTDI convention. No time, phase or amplitude is fitted.
5. Retain the interior record and print source, preparation, projection and
   TDI clocks. Optional output contains `times`, `xyz` and JSON metadata;
   XYZ has shape `(3,N)` and dimensionless fractional-frequency units.

The tensor conversion is `egb_multi.fastlisa.polarization_basis_matrix`.
It takes half the contraction of the native and rotated eGB polarization
tensors and checks their reconstruction. It handles changed sky position and
nonzero polarization angle without an empirical sign or phase fit.

The example makes the receiver ordering explicit using
`native.link_space_craft_0_in = native.response_orbits.link_space_craft_r`
for the checked upstream path. It modifies no upstream source or ephemeris.
If using a different upstream version, retain the displayed version metadata
and repeat the convention check.

The configured orbit grid defaults to 37.5 s and covers the requested padded
interval. It is distinct from the 2.5 s source/output cadence. Accuracy of the
ephemeris, orbit interpolation and source sampling still needs verification
for the intended source. The margins must cover detector wavefront shifts,
the longest TDI delays and the centered source/TDI interpolation stencils.
The upstream implementation raises if the initial margin is insufficient.

## Same-input comparison with STRIDE

`--compare-stride` passes exactly the same sampled source through STRIDE,
with source/TDI interpolation orders 39/25. Both engines use the same
configured orbit file and instantaneous-emitter geometry for this comparison.
It reports unweighted XYZ relative L2 error, with no fitted alignment.
Interpolation methods and settings differ; the output is a numerical
consistency measurement rather than a source-model overlap.

The separate [eGB + STRIDE example](../examples/egb_stride.py) uses eGB's
retarded-emitter geometry and the ESA-trailing analytic orbit. Comparing its
output directly with the default native example would mix orbit and emitter
prescriptions. Use the matched comparison to isolate the two implementations.

These scripts are usage examples. Their single first-call clocks include
setup and compilation effects and are not the paper's controlled warm timing
measurements. The optional paired comparison alone does not establish
convergence or equal-response-error timing.

## NVIDIA CUDA

Install the supported fastlisaresponse CUDA backend and matching CuPy runtime
from its [official documentation](https://mikekatz04.github.io/lisa-on-gpu/).
Select a compatible JAX GPU installation from the
[official JAX guide](https://docs.jax.dev/en/latest/installation.html).
Then, on an allocated CUDA machine:

```sh
python examples/egb_fastlisaresponse.py --backend cuda12x --source-backend jax --compare-stride
```

The example requests an actual JAX GPU and native CUDA backend; it does not
silently fall back to CPU. Fixed and full-PM eGB polarizations can use the JAX source and
share its device arrays with CuPy using DLPack. Selecting the NumPy source
uses an explicit transfer. Both routes use float64; CUDA timings and accuracy must
be checked on the actual hardware. This repository integration launches no
cloud compute.

## Citations

Please cite the relevant eGB/STRIDE work and
[Katz et al., arXiv:2204.06633](https://arxiv.org/abs/2204.06633) when using
fastlisaresponse. Its [API documentation](https://mikekatz04.github.io/lisa-on-gpu/user/main.html)
defines the native input, orbit and backend contracts.
