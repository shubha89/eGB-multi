import numpy as np

from egb_multi import matrix_to_tracefree5, normalize_direction, tracefree_matrix


def test_tracefree_matrix_roundtrip():
    coeffs = np.array([1 + 2j, 3, -4j, 5, 6 - 1j])
    matrix = tracefree_matrix(coeffs)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.trace(matrix), 0.0)
    assert np.allclose(matrix_to_tracefree5(matrix), coeffs)


def test_normalize_direction():
    direction = normalize_direction([0.0, 3.0, 4.0])

    assert np.allclose(direction, [0.0, 0.6, 0.8])
    assert np.isclose(np.linalg.norm(direction), 1.0)
