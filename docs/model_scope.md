# Source model and comparison scope

eGB-multi evaluates source waveforms through retarded LISA links and TDI.
The present source model uses constant component masses and quadrupole
polarizations on Newtonian or conservative 1PN quasi-Keplerian orbits.
The 1PN option is not a complete 1PN amplitude model. Radiation reaction is
optional leading orbit-averaged Peters evolution for an isolated binary.

## Physics represented and omitted

The implemented GR periastron advance is one contribution to apsidal motion.
Stellar tides and rotation can change the total rate in white-dwarf systems;
detachment alone does not establish that those effects are negligible.
Mass transfer, coupled spin/tidal evolution, changing component masses and
source acceleration from a tertiary are not self-consistently evolved.
Known LISA detector motion does not supply additional source CoM motion.

A free frequency derivative is a prescribed template freedom, not a model of
any one of these mechanisms. An effect absorbed by fitted parameters can
still bias their astrophysical interpretation. Quantitative applicability to
a real source needs its own stellar/environmental inputs and observation-specific
accuracy check; the illustrative point-mass calculations do not supply a
universal safe frequency or a validation of every Galactic-binary population.


## Frequencies, epochs and backend contract

Source `mean_motion` follows the azimuthal carrier convention used by the
response model: the nearly circular GW carrier is `mean_motion / pi`.
The radial orbital frequency is `n / (2*pi)` and differs when precession is
included. `phi0` is an orbital phase; the leading circular GW phase is twice
that phase. Source epochs and detector timestamps must be stated explicitly.
Sky latitude/longitude and polarization vectors use barycentric mean ecliptic
J2000 coordinates; native ICRS orbit positions must be converted consistently.

| API path | Treatment of a supplied `fdot` |
|---|---|
| Fixed JAX source/link path | Prescribed linear carrier chirp with constant masses/eccentricity |
| NumPy source functions | Does not implement that prescribed `fdot`; do not use it for the chirping-template contract |
| Non-fixed exact-link evolution | Rejects nonzero prescribed `fdot`; use explicit isolated Peters evolution |
| Circular envelope and native search verifier | Uses the fixed JAX chirping-template convention |

This documented backend asymmetry has not been silently repaired in the
release preparation. Automatic evolution switching is a source-level
fixed-versus-Peters mismatch rule, not an omitted-physics or inference guarantee.

## Response, noise and numerical scope

Native retarded geometry uses reception-time light-travel times and the
emitter position at emission. Historical calls without an orbit provider
retain instantaneous geometric arms. TDI generation, interpolation orders,
cadence and analysis band are independent configuration choices; package
defaults do not imply that every configuration has converged.

Instrument noise adapters require an explicit noise configuration. The
ideal-clock/ranging paper reference and external-model comparisons are described
under [tests/](../tests/README.md), separately from the library API.
