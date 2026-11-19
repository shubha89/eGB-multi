# STRIDE reference suite

This directory contains the current STRIDE checks, separate from historical
research campaigns. It compares centered delays and factorized Michelson
stages with pyTDI; checks independent analytic equal-arm transfer and laser
limits; and exercises JAX/Numba, changing sky, late epochs, retarded geometry,
source batching, streaming boundaries, input validity and derivatives.

```sh
python -m pip install ".[test,cpu,orbits]"
python -m pytest tests/stride
```

For the public JAX-only contract:

```sh
python -m pip install ".[test]"
python -m pytest tests/stride/test_jax_public.py tests/stride/test_jax_batching.py
```

Numba-dependent test modules are skipped at collection when Numba is absent.
The JAX-only import test blocks Numba, pyTDI and waveform-generator imports
to check that the independent STRIDE interface retains this boundary.

`imported_source_sha256.json` records the source payload taken from the
qualified local STRIDE development component. Only its optional-install
messages and one terminal blank line are adjusted for integration. The scientific
kernel arithmetic and default interpolation settings are preserved.

These checks are short reference/regression tests. No annual benchmark,
external model comparison or cloud job is launched by this suite.
