# Referee-development package update

The Git package now contains the complete `egb_multi` implementation used for
the referee calculations, rather than a partial port into the older source
tree. The older `egb_jax_eccentric` namespace aliases its source, geometry,
constants, tensor and external-adapter modules. Its XYZ/AET convenience API
remains available. This avoids maintaining two different waveform models.

## Corrections and current capabilities

- Transform native ICRS orbit positions into barycentric mean ecliptic J2000
  for the source beta/lambda and polarization conventions.
- With a native orbit provider, evaluate emitters at emission time and use
  reception-time light travel times consistently in GW links and TDI.
- Include SCI, TMI and REF carrier fluctuations when mapping instrument noise
  to PyTDI. The science-only adapter remains appropriate for GW-only streams.
- Use tested high-order interpolation in the instrument-noise helpers, with
  explicit component selection for ideal clocks and ranging.
- Retain the current 1PN/PM source implementation, physical orientation fits,
  sky/mass/frequency/chirp and harmonic-basin searches, native winner checks,
  full Hermitian A/E noise covariance, finite-covariance and laser-symbol
  diagnostics, numerical convergence controls, Slurm orchestration and
  transient waveform-cache policy.

### Native retarded GW response

```python
import numpy as np
from egb_jax_eccentric import (
    EccentricBinaryParams, default_lisaorbits, eccentric_aet_jax,
    state_from_lisaorbits,
)

times = np.arange(4096) * 2.5
orbit = default_lisaorbits("esa-trailing")
state = state_from_lisaorbits(orbit, times)
source = EccentricBinaryParams(
    mean_motion=np.pi * 0.001, eccentricity=0.1,
    m1_solar=0.5, m2_solar=0.5, beta=0.2, lambda_=-1.0,
)
aet = eccentric_aet_jax(
    state, source, orbits=orbit, generation=2,
    measurement_order=45, delay_order=5, batch_size=1,
)
```

Providing no orbit provider retains instantaneous geometric arms for the
analytic-state API. `frame="icrs"` explicitly preserves native ICRS positions;
the source vectors must then use that same frame. Link label ij is received
at i and emitted by j. A=(Z-X)/sqrt(2), E=(X-2Y+Z)/sqrt(6).

### Ideal-clock/ranging instrument-noise diagnostic

```python
from egb_multi.noise import simulate_lisainstrument_tdi_noise

noise = simulate_lisainstrument_tdi_noise(
    size=16384, dt=5.0, generation=2, order=45, delay_order=5,
    enabled_noises=("laser", "test-mass", "oms"),
)
fractional_A = noise.aet["A"] / noise.central_frequency_hz
```

Noise outputs are in Hz; divide by the recorded carrier frequency to obtain
fractional frequency, or divide PSDs by its square. An explicit component list
uses LISA Instrument's noise-selection API and disables the omitted clock and
ranging imperfections. Passing `enabled_noises=None` retains instrument
defaults and does not assert that those measurements are calibrated. Exclude
nested-delay/interpolation edge samples before interpreting residual spectra.

Saved short native-orbit laser-only checks covered 0.1–30 mHz and found residual
laser PSD below approximately 6.6e-7 of secondary-noise PSD. Matched three-seed
laser-on/off tests under ideal clocks/ranges found a largest increment ratio
of 2.95e-6. These are scoped finite-run results, not a full-band mission
sensitivity guarantee. The annual production covariance explicitly includes
the computed ideal primary-laser residual over 10 microHz–80 mHz.

The sampled LISA Instrument calculation has an unresolved end-to-end
precision issue: a small metrology fluctuation is rounded when added to a
large locked-laser fluctuation, before TDI. The first operation was diagnosed
under ideal clocks/ranging and independently checked with TwoSum/Decimal;
the entire instrument chain
has not been repaired by the stationary covariance calculation. A package
update does not establish that repair. Physical interpolation, ranging and
locking residuals also remain distinct from numerical roundoff; see
[Staab et al. (2024)](https://doi.org/10.1103/PhysRevD.109.043040).

Decision (14 October 2026): defer the end-to-end instrument arithmetic repair
to a separate follow-up. Retain the present convention fix and the frozen
Task 2.9 reference-covariance calculation.

## Install and validate

Use Python 3.12 or newer for the current optional LISA ecosystem packages:

```sh
python -m pip install -e ".[realistic,test,precision]"
python -m pytest -q -rs
```

The referee development has frozen source and input hashes. The source files
were imported without changing the numerical implementation, and the original
Euler campaign remains separate. Large results, waveform arrays, local caches
and downloaded papers are not Git source artifacts. Their historical evidence
must remain available for provenance audits; see the import and local-evidence
manifests in `reports/`.

The Slurm workers deliberately refuse changed source, input or retention-policy
hashes. Do not replace a running campaign's source with a later package commit.
The existing two Euler pilots are destination checks of the frozen calculation;
they do not validate a future instrument-precision repair.
