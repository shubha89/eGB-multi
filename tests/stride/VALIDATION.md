# Integration checks: 19 November 2026

Base: public eGB-multi main at `fed27c1e6b40648799bd049e5b4adb440c054113`.
STRIDE component: the current qualified local `0.1.1.dev2` sources, bundled
in eGB-multi `0.1.2.dev0`. All 25 imported STRIDE module payloads match their
recorded hashes after reversing the two installation-hint string changes
and restoring one removed final blank line.
No detector-response kernel, order, precision target or geometry convention
was changed during integration.

## Local numerical checks

- 394 STRIDE tests passed: 17 JAX-only public/batching tests and 377 additional
  pyTDI, analytic, Numba/JAX, interpolation, geometry and eGB-adapter checks.
- 258 eGB unit tests passed; three optional checks were skipped. This includes
  36 new source/basis/phase cases. The suite was run in scoped groups; after
  updating the two analytic reference calculations, their complete module
  and the source-adapter module were rerun.
- The two original parametric-dynamics references evaluated sine at a rounded
  full-orbit angle near a radial-velocity zero. Independent 65-digit decimal
  phase reduction restores the intended analytic reference, with unchanged
  tolerances. The independent reduced phases agree with the corrected source
  phases at every sampled point in these checks.
- Executed the retarded-geometry eGB + STRIDE example with JAX, and the Numba
  example. Executed native CPU fastlisaresponse with a fixed JAX eGB source,
  nonzero polarization angle, and its optional same-input STRIDE comparison.
  Also executed the native CPU route with PM evolution and NumPy source.
- The default 4096-sample native-vs-STRIDE XYZ relative L2 discrepancy was
  `7.937535185760792e-6`. This is a measured difference at the displayed
  interpolation settings, not a fitted alignment or an equal-error timing
  result. The example reports it without hiding failures.

## Environment and packaging

Mac arm64, Python 3.13; NumPy 2.4.6, SciPy 1.18.0, JAX/JAXlib 0.10.2,
Numba 0.66.0, pyTDI 2.1.0 and lisaorbits 3.0.3. The optional native example
uses the existing CPU reference binaries for fastlisaresponse 1.1.17 and
LISA Analysis Tools 1.2.0. Tests use a project virtual environment with the
existing local dependencies; this is not a new dependency-resolution audit
for every supported platform.

The wheel and source archive are checked for the bundled `lisa_stride`,
`xtdi` and eGB adapters and the required third-party notice. Usage examples
and tests are in the source archive. No PDF, manuscript, referee document,
private ephemeris, vendor tree, saved annual catalogue or historical research
campaign is introduced.

No GPU or annual waveform run, cloud rental, release publication or Git push
is part of these integration checks. The CUDA usage route is prepared, and
the imported STRIDE kernels retain the earlier configuration-specific GPU
qualification; this new example has only been executed on CPU here.
