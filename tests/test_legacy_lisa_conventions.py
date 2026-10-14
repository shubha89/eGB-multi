"""Check the legacy public response API against the independent native response."""

import numpy as np
import pytest

from egb_jax_eccentric import (
    EccentricBinaryParams, default_lisaorbits, eccentric_aet_jax,
    eccentric_links_jax, has_package, precompute_jax_link_geometry,
    state_from_lisaorbits,
)
from egb_multi.noise import aet_from_xyz, xyz_from_pytdi_data
from egb_multi.tdi import LINK_LABELS, pytdi_data_from_links

pytestmark = pytest.mark.skipif(
    not all(has_package(p) for p in ("lisaorbits", "pytdi", "jax")),
    reason="LISA orbit and TDI packages are required",
)


def test_legacy_native_orbit_response_preserves_retarded_tdi_delays():
    times = np.arange(2048) * 2.5
    orbit = default_lisaorbits("esa-trailing")
    state = state_from_lisaorbits(orbit, times)
    source = EccentricBinaryParams(mean_motion=np.pi * 0.01,
        eccentricity=0.3, m1_solar=0.8, m2_solar=0.6,
        beta=0.2, lambda_=-1.0, inclination=1.21, psi=0.71, phi0=0.617)
    geometry = precompute_jax_link_geometry(state, orbits=orbit)
    links = eccentric_links_jax(source, geometry, batch_size=1)
    native_delays = {label: geometry["length"][i] for i, label in enumerate(LINK_LABELS)}
    data = pytdi_data_from_links(state, links, link_delays=native_delays)
    expected = aet_from_xyz(xyz_from_pytdi_data(data, generation=2, order=45, delay_order=5))
    actual = eccentric_aet_jax(state, source, orbits=orbit, batch_size=1,
        generation=2, measurement_order=45, delay_order=5)
    for channel in ("A", "E", "T"):
        np.testing.assert_allclose(actual[channel], expected[channel], rtol=2e-12, atol=1e-30)
    # This constellation has genuinely retarded delays; geometric lengths
    # would not be an equivalent oracle for the response/TDI boundary.
    assert np.max(np.abs(native_delays["21"] - state.arm_length(0, 1))) > 1e-5


def test_legacy_modules_use_the_canonical_source_classes():
    from egb_multi.eccentric import EccentricBinaryParams as CanonicalSource
    assert EccentricBinaryParams is CanonicalSource
