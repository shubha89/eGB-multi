"""Compare optimized GW-only TDI with the full beatnote pyTDI reference."""
import numpy as np
import pytest

pytest.importorskip("pytdi")
from pytdi.michelson import compute_factorized_michelson
from egb_jax_eccentric import LISAState, lisa_orbit, prepare_xyz_from_links, xyz_from_links
from egb_jax_eccentric.pytdi_bridge import LINK_LABELS, pytdi_data_from_links


def full_reference(state, links, generation, order):
    data = pytdi_data_from_links(state, links)
    return {c: compute_factorized_michelson(data, rot=i, order=order,
        delay_order=order, generation=generation, unit="frequency")
        for i, c in enumerate("XYZ")}


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("varying", [False, True])
def test_full_reference_and_reuse(generation, order, varying):
    t = np.arange(2048)*5.
    state = lisa_orbit(t)
    if varying:
        # Deliberately varying unequal arms exercise nested delay derivatives.
        positions = state.positions.copy()
        positions[:, 1, 0] += .3*np.sin(t/2000)
        positions[:, 2, 1] += .5*np.cos(t/1800)
        state = LISAState(t=t, positions=positions)
    prepared = prepare_xyz_from_links(state, generation=generation,
        measurement_order=order, delay_order=order)
    rng = np.random.default_rng(123)
    for _ in range(2):
        links = {k:rng.normal(size=t.size).astype(complex) for k in LINK_LABELS}
        expected = full_reference(state, links, generation, order)
        for actual in [prepared(links), xyz_from_links(state, links,
                generation=generation, measurement_order=order, delay_order=order)]:
            for channel in "XYZ":
                np.testing.assert_allclose(actual[channel], expected[channel], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("generation", [1, 2])
def test_zero_delay_analytic_limit(generation):
    state = LISAState(t=np.arange(128.), positions=np.zeros((128,3,3)))
    rng = np.random.default_rng(456)
    links = {k:rng.normal(size=128) for k in LINK_LABELS}
    actual = prepare_xyz_from_links(state,generation=generation)(links)
    for value in actual.values():
        np.testing.assert_array_equal(value, np.zeros(128))


def test_prepared_input_validation():
    state=lisa_orbit(np.arange(128.))
    with pytest.raises(ValueError,match="generation"):
        prepare_xyz_from_links(state,generation=3)
    with pytest.raises(ValueError,match="positive odd"):
        prepare_xyz_from_links(state,measurement_order=4)
    prepared=prepare_xyz_from_links(state)
    with pytest.raises(ValueError,match="missing"):
        prepared({})
    with pytest.raises(ValueError,match="shape"):
        prepared({k:np.zeros(127) for k in LINK_LABELS})
