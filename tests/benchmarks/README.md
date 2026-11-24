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

## PM-on eGB to XYZ

Run a short example with the installed library and the analytic eGB orbit.

```sh
python tests/benchmarks/annual_pm_batch.py --sky fixed --batch-size 4
python tests/benchmarks/annual_pm_batch.py --sky distinct --batch-size 4
```

Use `--annual --device gpu` for one Julian year, and compare `--batch-size 1`
with larger groups. Use `--source standard` for separate PM solves or
`--source vectorized` for checked batch preparation. `--cadence 2.5` selects
the reference spacing. Every timed call starts from source parameters and ends
with synchronized XYZ on the device. The script checks the three channels
for each source against separate calls at the chosen cadence before timing.
Check the noise-weighted loss independently when changing the sky or population.

The [archived A100 table](annual_pm_a100.csv) used a numerical orbit. The
optional `--orbit FILE` mode takes a LISATools HDF5 orbit with BME positions
in metres, six directed light times in seconds and an absolute `time_origin_s`
attribute. It requires `lisaanalysistools` and `h5py`. The recorded input
SHA-256 was `de9f75324be61ef6d76a6ab51b3fa8da4c5941a05224d973fe802d712cced7f0`.
The detector began at 3,944,700 s. The archived table had 150 s orbit samples
and the provider used a 37.5 s interpolation grid. Sender and receiver positions
were simultaneous in the angular projection. The numerical ephemeris and
calibration covariance are external inputs. The default analytic example
provides a portable check of the interfaces.
