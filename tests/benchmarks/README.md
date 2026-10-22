# Explicit timing studies

These programs are optional. They are not imported by normal library use or
collected by default pytest.

- `source_timing.py`: fixed 1PN source polarizations, with separate JAX first-call,
  warmed and NumPy timings. It imports neither JAXGB nor pyEFPE.
- `prepared_xyz_timing.ipynb`: three full-year cases with JAX warm-up, geometry
  preparation and warmed response evaluation reported separately.
- `prepared_xyz_batch_timing.ipynb` and `prepared_xyz_batch_timing.py`: separate
  and summed responses for a 64-source catalogue, with independent output and
  memory checks. Cadence, block length, precision and TDI settings are explicit.

Install `.[orbits,analysis,notebook]` for the notebooks. Run them from anywhere
inside this checkout. Annual timing measurements and comparisons are explicit
computations; this release preparation does not execute them.

```sh
python tests/benchmarks/prepared_xyz_batch_timing.py --help
```

The two output modes in the batch study perform different tasks. Their timings
are not interchangeable. Reusable geometry is valid only for its stated orbit,
timestamps and interpolation configuration.
