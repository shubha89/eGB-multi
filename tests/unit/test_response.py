import numpy as np

from egb_multi import (
    GalacticBinaryParams,
    lisa_orbit,
    one_way_doppler,
    tracefree_matrix,
    triliear_source_batch,
    triliear_tensor,
    triliear_waveform,
)


def test_triliear_tensor_matches_direct_channel_formula():
    state = lisa_orbit(np.linspace(0.0, 20_000.0, 4))
    f0 = 1.0e-4
    coeffs = np.array([1.0, 0.2j, -0.1, 0.3, -0.4j])
    k = np.array([0.2, -0.4, 0.7])
    k = k / np.linalg.norm(k)

    a = triliear_tensor(state, f0)
    modeled = triliear_waveform(np.ones(state.t.size), a, coeffs, k)

    matrix = tracefree_matrix(coeffs)
    manual = []
    for first, second, complement in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        n = state.unit_arm(first, second)
        length = state.arm_length(first, second)
        b = -2j * np.pi * f0 * length
        a_pair = 2.0 * b * (1.0 + b + (2.0 / 3.0) * b**2)
        a_comp = -(b**2)[:, None] * state.relative_positions[:, complement, :] / length[:, None] * (1.0 + b)[:, None]
        nen = np.einsum("ti,ij,tj->t", n, matrix, n)
        manual.append(0.5 * nen * (a_pair + np.einsum("ti,i->t", a_comp, k)))
    manual = np.stack(manual, axis=1)

    assert np.allclose(modeled, manual)


def test_one_way_doppler_returns_complex_time_series():
    state = lisa_orbit(np.linspace(0.0, 100.0, 8))
    coeffs = np.array([1.0, 0.0, 0.2, -0.3, 0.1])
    k = np.array([0.2, 0.3, 0.4])

    def h(t):
        return np.exp(2j * np.pi * 1.0e-4 * t)

    y = one_way_doppler(state, 0, 1, k, coeffs, h)

    assert y.shape == (8,)
    assert np.iscomplexobj(y)
    assert np.all(np.isfinite(y))


def test_one_way_doppler_linear_waveform_matches_path_length_limit():
    state = lisa_orbit(np.linspace(0.0, 100.0, 8))
    coeffs = np.array([1.0, 0.0, 0.2, -0.3, 0.1])
    k = np.array([0.2, 0.3, 0.4])
    k = k / np.linalg.norm(k)

    def h(t):
        return np.asarray(t, dtype=np.float64)

    y = one_way_doppler(state, 0, 1, k, coeffs, h)

    n = state.unit_arm(0, 1)
    length = state.arm_length(0, 1)
    matrix = tracefree_matrix(coeffs)
    nen = np.einsum("ti,ij,tj->t", n, matrix, n)

    assert np.allclose(y, -0.5 * nen * length)


def test_triliear_source_batch_matches_loop_sum():
    state = lisa_orbit(np.linspace(0.0, 10_000.0, 32))
    sources = [
        GalacticBinaryParams(f0=8.0e-4, fdot=1.0e-18, amplitude=1.0e-22, beta=0.2, lambda_=0.4, psi=0.1),
        GalacticBinaryParams(f0=1.2e-3, fdot=-2.0e-18, amplitude=2.0e-22, beta=-0.1, lambda_=-0.3, psi=0.7),
    ]

    loop = np.zeros((state.t.size, 3), dtype=np.complex128)
    for source in sources:
        loop += triliear_waveform(
            source.slow_envelope(state.t),
            triliear_tensor(state, source.f0, fdot=source.fdot),
            source.tracefree_coefficients(),
            source.propagation_direction(),
        )

    assert np.allclose(triliear_source_batch(state, sources), loop)
