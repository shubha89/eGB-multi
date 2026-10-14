# Scientific Equation Audit

Date: 2026-08-25

Scope:
- `src/egb_multi/eccentric.py`
- `src/egb_multi/response.py`
- `src/egb_multi/tdi.py`
- `reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex`

Primary sources checked:
- Peters and Mathews, Phys. Rev. 131, 435 (1963), DOI: https://doi.org/10.1103/PhysRev.131.435
- Peters, Phys. Rev. 136, B1224 (1964), DOI: https://doi.org/10.1103/PhysRev.136.B1224
- Damour, Gopakumar, and Iyer, Phys. Rev. D 70, 064028 (2004), arXiv: https://arxiv.org/abs/gr-qc/0404128
- Tessmer and Gopakumar, MNRAS 374, 721 (2007), arXiv: https://arxiv.org/abs/gr-qc/0610139
- Tanay, Haney, and Gopakumar, Phys. Rev. D 93, 064031 (2016), arXiv: https://arxiv.org/abs/1602.03081
- Cornish and Rubbo, Phys. Rev. D 67, 022001 (2003), arXiv: https://arxiv.org/abs/gr-qc/0209011

## Verified Items

1. Eccentricity domain
   - Public eccentricity validation enforces finite `0 <= e < 1`.
   - Fixed bug: `peters_mathews_derivatives` now validates the full eccentricity array, not only `max(e)`.
   - Regression: negative array elements and `e >= 1` now raise `ValueError`.

2. Kepler equation
   - `mikkola_kepler` solves `u - e sin(u) = l`.
   - Existing regression verifies residuals below `1e-13` for `e = 0, 0.1, 0.7, 0.95`.

3. Newtonian eccentric dynamics
   - Added regression reconstructing `r`, `rdot`, `phi`, and `phidot` from the Keplerian parametric solution.
   - This checks the code against the independent analytic Newtonian limit.

4. 1PN quasi-Keplerian dynamics
   - Added regression using the 1PN coefficient functions from Tanay-Haney-Gopakumar Appendix B for `r`, `rdot`, `phi`, and `phidot`.
   - Verified against `physics_mode="1pn_no_periastron"` at relative tolerance `1e-14`.

5. Quadrupole polarizations
   - Source polarizations match the standard quadrupole expressions in Tanay-Haney-Gopakumar.
   - Added circular Newtonian limit regression:
     `h_plus = -2 A v^2 (1 + cos(i)^2) cos(2 phi)`,
     `h_cross = -4 A v^2 cos(i) sin(2 phi)`.

6. Peters-Mathews secular evolution
   - Added regression for the leading quadrupole formulas:
     `dn/dt proportional n^(11/3) (1 + 73 e^2/24 + 37 e^4/96)/(1-e^2)^(7/2)`,
     `de/dt proportional -n^(8/3) e (1 + 121 e^2/304)/(1-e^2)^(5/2)`.
   - The implementation evolves radial mean motion `n`; when periastron advance is enabled, it reconstructs the azimuthal frequency through the 1PN relation.
   - The manuscript was updated to state this `n`-based software convention.
   - Removed private integrator clipping of eccentricity and mean motion. Internal Peters-Mathews states now raise if they leave the physical domain instead of silently clipping to `[0, 1)`.

7. One-way LISA response
   - Added algebraic regression for a linear waveform. The retarded-time difference cancels the denominator and gives the expected path-length factor.
   - This verifies the local one-way response convention in `response.one_way_doppler`.
   - Added same-convention comparison against installed `lisagwresponse` 3.0.2 using `lisaorbits` equal-arm orbits. The adapter maps the local FastGB-like complex convention to `lisagwresponse`'s real convention by using `A -> -A` and `phi0 -> -phi0`; with nonzero `phi0`, `psi`, and inclination, all six links agree with `np.real(link_responses(...))` at relative L2 error below `5e-4`.
   - Added pyTDI Michelson X/Y/Z comparison where both local and `lisagwresponse` links are passed through the same installed `pytdi` 2.1.0 factorized Michelson builder. X/Y/Z relative L2 errors are below `1e-3`.

8. Manuscript-code consistency
   - Updated the Peters-Mathews block to evolve `n`, `e_t`, `l`, and `lambda`, matching the software.
   - Updated phase-drift wording to "dominant gravitational-wave carrier phase drift"; the quoted numbers correspond to the carrier phase, not the orbital mean anomaly alone.
   - Removed a visible draft marker and replaced a manual equation tag with a normal label.

## Remaining Scientific Limits

These items are not fully certified by this pass:

- The trilinear tensor implementation has internal direct-formula tests, but this pass did not independently rederive Andersson et al. Eqs. 11-14.
- LISA response validation is now externally checked for equal-arm `lisaorbits` geometry and first-generation pyTDI X/Y/Z. Remaining response-validation work is a broader mission-orbit sweep, A/E/T normalization checks, higher-generation TDI checks, and explicit quantification of the local same-time link-geometry approximation against retarded emitter geometry.
- Figure provenance was checked previously from generated CSVs, but this pass did not regenerate every publication figure from scratch.

## Commands Run

- `python -m pytest tests/test_eccentric.py tests/test_response.py -q`
- `python -m pytest tests/test_eccentric.py::test_peters_mathews_private_state_derivative_rejects_nonphysical_states tests/test_reference.py::test_lisagwresponse_links_match_local_physical_strain_convention tests/test_reference.py::test_pytdi_xyz_from_lisagwresponse_links_matches_local_xyz_convention -q`
- `python -m pytest -q`
- `tectonic --keep-logs --keep-intermediates --outdir /tmp/egb_paper_build apssamp.tex`
