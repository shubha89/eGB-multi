import numpy as np

from egb_multi.sources import GalacticBinaryParams


def test_galactic_binary_fastgb_order_and_direction():
    source = GalacticBinaryParams(
        f0=1.0e-3,
        fdot=2.0e-18,
        amplitude=3.0e-22,
        beta=0.4,
        lambda_=-0.2,
        psi=0.7,
        inclination=1.1,
        phi0=2.2,
    )

    assert np.allclose(source.as_fastgb_array(), [1.0e-3, 2.0e-18, 3.0e-22, 0.4, -0.2, 0.7, 1.1, 2.2])
    assert np.isclose(np.linalg.norm(source.propagation_direction()), 1.0)


def test_galactic_binary_polarization_is_tracefree():
    source = GalacticBinaryParams(f0=1.0e-3, beta=0.4, lambda_=-0.2, psi=0.7, inclination=1.1)

    matrix = source.polarization_matrix()
    k = source.propagation_direction()

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.trace(matrix), 0.0)
    assert np.allclose(k @ matrix @ k, 0.0)
    assert source.tracefree_coefficients().shape == (5,)
