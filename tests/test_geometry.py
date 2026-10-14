import numpy as np

from egb_multi import LISAOrbitConfig, lisa_orbit


def test_lisa_orbit_shapes_and_arm_lengths():
    state = lisa_orbit(np.linspace(0.0, 10_000.0, 5))

    assert state.positions.shape == (5, 3, 3)
    assert state.relative_positions.shape == (5, 3, 3)

    lengths = [state.arm_length(0, 1), state.arm_length(0, 2), state.arm_length(1, 2)]
    for length in lengths:
        assert np.all(length > 0.0)
        assert np.allclose(length, LISAOrbitConfig().arm_length, rtol=2e-2)


def test_unit_arm_has_unit_norm():
    state = lisa_orbit([0.0, 1000.0])
    unit = state.unit_arm(0, 1)

    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)
