# eGB-multi

Eccentric Galactic-binary source models, retarded LISA links, TDI responses,
and noise-weighted circular-template comparisons. This checkout contains the
current referee-development implementation and its reproduction scripts.

`egb_multi` is the canonical package. Existing `egb_jax_eccentric` imports use
the same core source and geometry implementation, with the legacy XYZ/AET
convenience API retained. See [the update and validation guide](docs/referee_development_update.md)
for convention corrections, ideal-clock/ranging laser tests, frozen input
data, and the scope of the remaining instrument arithmetic repair.

The package also includes experiments for the triliear LISA response formalism
in Andersson et al., EPL 140, 19001 (2022).

The current implementation is intentionally small:

- second-order eccentric guiding-center LISA orbit approximation,
- adapters for `lisaorbits` and `fastgb`,
- arm lengths and unit arm vectors,
- trace-free polarization matrix parameterization,
- exact one-way Doppler response from Eq. (6),
- triliear response tensor from Eqs. (11)-(14).
- vectorized triliear source batches for multi-source timing experiments.
- exact six-link GW responses that can be passed through pyTDI to compute
  time-domain Michelson X/Y/Z or A/E/T.
- beta-expanded six-link GW responses with `order`, `retarded_geometry`, and
  `barycentric_phase` options, also routed through pyTDI for X/Y/Z comparisons.
  The default `compute_triliear_xyz` path uses `order=1`,
  `barycentric_phase=True`, and pyTDI interpolation order 3.
- eccentric source generation keeps secular evolution and source physics as
  separate API axes:
  `evolution_mode="fixed"` holds source parameters fixed, while
  `evolution_mode="peters_mathews"` (`"pm"` alias) evolves radial mean motion,
  eccentricity, and mean anomaly with Peters-Mathews radiation reaction. The
  partial diagnostics `peters_mathews_orbital_only` and
  `peters_mathews_eccentricity_only` isolate the two secular pieces.
  `evolution_mode="auto"` selects between the fixed and Peters-Mathews paths
  using a user-specified fixed-1PN-vs-full-PM source mismatch tolerance. Pass a
  precomputed `PetersMathewsSwitchRule` for fast calibrated switching, or omit
  the rule to compute the source-level mismatch directly for that waveform.
  `physics_mode="newtonian"` uses Newtonian orbital dynamics,
  `physics_mode="1pn_no_periastron"` adds the explicit 1PN orbital corrections
  without apsidal precession, and `physics_mode="1pn_periastron"` (also accepted
  as the historical `"1pn"`) adds both 1PN orbital terms and periastron advance.

LISA geometry distances are represented as light-travel times. Source
luminosity distances are represented in meters, and frequencies are in Hz.

For example, the current full eccentric source model is selected by combining
the two axes explicitly:

```python
eccentric_complex_strain(
    source,
    times,
    physics_mode="1pn_periastron",
    evolution_mode="peters_mathews",
)
```

The automatic Peters-Mathews switch can be calibrated on a sparse source grid
and then reused:

```python
rule = calibrate_peters_mathews_switch_rule(
    calibration_sources,
    calibration_times,
    mismatch_tolerance=1e-2,
)

eccentric_complex_strain(
    source,
    times,
    physics_mode="1pn_periastron",
    evolution_mode="auto",
    pm_mismatch_tolerance=1e-2,
    pm_switch_rule=rule,
)
```

If no `pm_switch_rule` is supplied, `evolution_mode="auto"` computes the
fixed-vs-full-PM mismatch directly for that source and time grid.

## Reusing TDI geometry

For repeated GW-only evaluations on the same orbit and time grid, prepare the
factorized pyTDI operators once:

```python
from egb_jax_eccentric import (
    eccentric_links_jax, precompute_jax_link_geometry, prepare_xyz_from_links,
)

geometry = precompute_jax_link_geometry(state)
tdi = prepare_xyz_from_links(state, generation=2)
for source in sources:
    links = eccentric_links_jax(source, geometry, batch_size=1)
    xyz = tdi(links)
```

This caches nested delays and their Doppler factors; source-dependent
measurement interpolation still runs for every call. Rebuild when the time
grid, spacecraft positions, or interpolation settings change. Large grids
require substantial cache memory; process sources on a shared block when
working with long observations. This API is for GW-only links, not full
instrument beatnotes. The ordinary `xyz_from_links` path also avoids
interpolating zero reference/metrology channels.

See [the measured TDI optimization results](benchmarks/results/tdi_optimization/REPORT.md).

Examples: [minimal prepared waveform](notebooks/minimal_prepared_xyz.ipynb)
and [one year with shared block caches](notebooks/one_year_prepared_xyz.ipynb).
The one-year example retains two independent XYZ waveforms and reports shared
preparation separately from waveform evaluation.

For native retarded geometry, prepare the operators with those same delays:

```python
from egb_jax_eccentric import LINK_LABELS

geometry = precompute_jax_link_geometry(state, orbits=orbit)
delays = {label: geometry["length"][i] for i, label in enumerate(LINK_LABELS)}
tdi = prepare_xyz_from_links(state, generation=2, link_delays=delays)
links = eccentric_links_jax(source, geometry, batch_size=1)
xyz = tdi(links)
```

The GW-only shortcut applies to source links. Full instrument noise retains
the SCI/TMI/REF construction documented in the referee-development note.

## Tests

```bash
python -m pytest
```

Optional realistic backends:

```bash
python -m pip install -e ".[realistic,test]"
python benchmarks/fastgb_walltime.py
python benchmarks/beta_walltime_options.py
python benchmarks/fastgb_accuracy.py
python benchmarks/time_domain_reference.py
```

The benchmark currently compares runtime and output scales for the same
non-eccentric galactic-binary parameters. It is not yet a residual validation,
because FastGB returns standard TDI `X/Y/Z` frequency-domain observables while
the triliear implementation currently returns the paper's TDI-like pair
channels.

For `default_lisaorbits("esa-trailing")`, this repository uses the bundled
`lisaorbits.KeplerianOrbits` unequal-arm trailing analytic model. The installed
FastGB backend accepts this orbit object but reduces it to a single scalar
`arm_length` internally, so its frequency-domain output should not be treated as
a native unequal-arm response.
