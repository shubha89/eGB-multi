# Paper data and six-figure reproduction

Install the source distribution with `.[orbits,analysis,test,precision,paper]`.
The library wheel alone contains no research drivers or reference data. Restore
the separate reference-data archive from the source checkout:

```sh
python tests/paper/reproduce.py restore /path/to/egb_reference_data_v0.1.1.tar.gz
python tests/paper/reproduce.py verify
python tests/paper/reproduce.py show
python tests/paper/reproduce.py figures --output tests/results/paper_figures
```

The archive contains 359 current-study evidence files with unchanged bytes. The ten direct plotting inputs
restore under `tests/paper/data/`; the remaining records restore under
`tests/paper/reference_data/`, preserving their original relative paths below that
prefix. Their numerical bytes are unchanged. Historical JSON records retain
the original campaign paths and input identities as provenance, not as current
filesystem commands. Obsolete source snapshots and old-to-new audit material
are retained locally and excluded. No manuscript or referee PDFs are required
or distributed.

Restoration checks the complete member set and all hashes before writing,
rejects links and unsafe paths, and refuses to overwrite modified evidence.
`verify` also checks the selected release-source identities. Neither command
generates a waveform, noise realization or search.

| Figure | Inputs and interpretation |
| --- | --- |
| 1 | Fixed Newtonian polarizations and first twelve radial harmonics at e=0, 0.3, 0.6 |
| 2 | Conservative 1PN/Newtonian binned source envelopes; apsidal period 400.3083 days and half-period marker 200.1542 days |
| 3 | Four-year initial-rate Peters diagnostics, without detector weighting or a complete inspiral extrapolation |
| 4 | Fixed annual fractional-frequency A/E covariance; ideal clocks/ranging and no Galactic foreground |
| 5 | 22 individual annual sources and 14 selected broader comparisons; restricted circular-template SNR loss |
| 6 | Four annual source-polarization pyEFPE comparisons and their recorded timings; shared physics and different PN models distinguished |

Figure replay reads saved inputs and recorded timing values. It performs no
waveform generation, evolution integration, noise simulation, timing measurement
or search. It generates local PDF/PNG figures; PDF files remain ignored by Git.
Figure 2 applies the recorded vector half-period marker. Poppler `pdftoppm`
provides the reference PNG rendering, with an equivalent Matplotlib fallback.

To recalculate only the existing source illustrations using the installed library:

```sh
python tests/paper/generate_source_data.py --output tests/results/source_illustrations
```

This is a separate explicit calculation. The directory cleanup checks its output
against the saved arrays; it does not rerun the annual detector/search campaigns.
Historical broad-search and fixed-candidate campaign controllers remain in the
local project archive, outside this clean release. The released evidence reports
a restricted nine-chirp comparison with physical orientation and phase, plus
selected finite broader searches; it is not a global fitting-factor survey or
Bayesian eccentricity-measurement forecast.

To repackage the verified data without recalculation:

```sh
python tests/paper/pack_reference.py --output tests/results/repacked_data
```

Source/data identities are in `reference_data_manifest.json`, plotting identities
in `plot_data_manifest.json`, and external model versions in
`../comparisons/external_models.json`. The saved rendering environment is pinned
in `../constraints-macos-arm64-py313.txt`.
