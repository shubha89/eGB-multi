import numpy as np
import pytest

from egb_multi import GalacticBinaryParams, lisa_orbit
from egb_multi.external import has_package
from egb_multi.tdi import LINK_LABELS, beta_link_responses, compute_aet, compute_beta_xyz, compute_triliear_xyz, compute_xyz, link_responses, pytdi_data_from_links


def test_link_labels_use_receiver_emitter_convention():
    assert LINK_LABELS == ("21", "32", "13", "31", "23", "12")


def test_link_responses_has_all_six_links():
    state = lisa_orbit(np.arange(64.0))
    source = GalacticBinaryParams(f0=1.0e-3, amplitude=1.0e-22, beta=0.2, lambda_=0.4)

    links = link_responses(state, source)

    assert tuple(links) == LINK_LABELS
    for value in links.values():
        assert value.shape == (64,)
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("pytdi"), reason="pytdi not installed")
def test_pytdi_data_from_links_populates_sci_measurements():
    state = lisa_orbit(np.arange(128.0))
    source = GalacticBinaryParams(f0=1.0e-3, amplitude=1.0e-22, beta=0.2, lambda_=0.4)
    links = link_responses(state, source)

    data = pytdi_data_from_links(state, links)

    assert data.fs == 1.0
    for label in LINK_LABELS:
        assert np.allclose(data.measurements[f"sci_{label}"], links[label])
        assert data.delays[f"d_{label}"].shape == (128,)


def test_pytdi_link_delays_can_match_noise_pipeline_exactly():
    state = lisa_orbit(np.arange(128.))
    links = {label: np.zeros(state.t.size) for label in LINK_LABELS}
    delays = {label: np.full(state.t.size, 8.2+i*.01) for i, label in enumerate(LINK_LABELS)}
    data = pytdi_data_from_links(state, links, link_delays=delays)
    for label in LINK_LABELS:
        np.testing.assert_array_equal(data.delays[f'd_{label}'], delays[label])
    delays['12'][0] = -1.
    with pytest.raises(ValueError, match='delay'):
        pytdi_data_from_links(state, links, link_delays=delays)


@pytest.mark.parametrize('generation',[1,2])
def test_prepared_GW_TDI_matches_native_factorized_path_with_flexing_delays(generation):
    from egb_multi.tdi import PreparedGWTDIMichelson, LINKS
    from egb_multi.noise import xyz_from_pytdi_data
    state = lisa_orbit(np.arange(1024)*5.)
    rng = np.random.default_rng(8031)
    links = {label:rng.normal(size=state.t.size) for label in LINK_LABELS}
    delays = {label:state.arm_length(sender,receiver)+1e-3*np.sin(state.t/500.+index)
        for index,(label,(sender,receiver)) in enumerate(zip(LINK_LABELS,LINKS,strict=True))}
    prepared = PreparedGWTDIMichelson(state,delays,generation=generation,measurement_order=45,delay_order=5)
    expected = xyz_from_pytdi_data(pytdi_data_from_links(state,links,link_delays=delays),generation=generation,order=45,delay_order=5)
    actual = prepared(links)
    for channel in ('X','Y','Z'):
        np.testing.assert_allclose(actual[channel][200:-200],expected[channel][200:-200].real,rtol=2e-12,atol=2e-12)


@pytest.mark.skipif(not has_package("pytdi"), reason="pytdi not installed")
def test_compute_aet_returns_three_channels():
    state = lisa_orbit(np.arange(512.0))
    source = GalacticBinaryParams(f0=1.0e-3, amplitude=1.0e-22, beta=0.2, lambda_=0.4)

    aet = compute_aet(state, source, generation=1, measurement_order=3, delay_order=3)

    assert set(aet) == {"A", "E", "T"}
    for value in aet.values():
        assert value.shape == (512,)
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("pytdi"), reason="pytdi not installed")
def test_compute_xyz_returns_three_channels():
    state = lisa_orbit(np.arange(512.0))
    source = GalacticBinaryParams(f0=1.0e-3, amplitude=1.0e-22, beta=0.2, lambda_=0.4)

    xyz = compute_xyz(state, source, generation=1, measurement_order=3, delay_order=3)

    assert set(xyz) == {"X", "Y", "Z"}
    for value in xyz.values():
        assert value.shape == (512,)
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


def test_beta_link_responses_approach_exact_at_low_frequency():
    state = lisa_orbit(np.linspace(0.0, 1000.0, 128))
    source = GalacticBinaryParams(f0=1.0e-5, amplitude=1.0e-22, beta=0.2, lambda_=0.4)

    exact = link_responses(state, source)
    approx = beta_link_responses(state, source, order=3)

    for label in LINK_LABELS:
        err = np.linalg.norm(approx[label] - exact[label]) / np.linalg.norm(exact[label])
        assert err < 5.0e-2


@pytest.mark.skipif(not has_package("pytdi"), reason="pytdi not installed")
def test_compute_beta_xyz_returns_three_channels():
    state = lisa_orbit(np.arange(512.0))
    source = GalacticBinaryParams(f0=1.0e-4, amplitude=1.0e-22, beta=0.2, lambda_=0.4)

    xyz = compute_beta_xyz(
        state,
        source,
        order=2,
        retarded_geometry=True,
        barycentric_phase=True,
        generation=1,
        measurement_order=3,
        delay_order=3,
    )

    assert set(xyz) == {"X", "Y", "Z"}
    for value in xyz.values():
        assert value.shape == (512,)
        assert np.iscomplexobj(value)
        assert np.all(np.isfinite(value))


@pytest.mark.skipif(not has_package("pytdi"), reason="pytdi not installed")
def test_compute_triliear_xyz_uses_recommended_defaults():
    state = lisa_orbit(np.arange(512.0))
    source = GalacticBinaryParams(f0=1.0e-4, amplitude=1.0e-22, beta=0.2, lambda_=0.4)

    default = compute_triliear_xyz(state, source)
    explicit = compute_beta_xyz(
        state,
        source,
        order=1,
        barycentric_phase=True,
        retarded_geometry=False,
        generation=1,
        measurement_order=3,
        delay_order=3,
    )

    for channel in ("X", "Y", "Z"):
        assert np.allclose(default[channel], explicit[channel])
