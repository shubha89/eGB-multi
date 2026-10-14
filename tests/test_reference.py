import numpy as np
import pytest

from egb_multi import GalacticBinaryParams
from egb_multi.external import default_lisaorbits, has_package, state_from_lisaorbits
from egb_multi.reference import brute_force_links, brute_force_pair_channels, brute_force_xyz
from egb_multi.tdi import LINK_LABELS, compute_xyz, link_responses


def _relative_l2(actual, expected) -> float:
    actual = np.asarray(actual).reshape(-1)
    expected = np.asarray(expected).reshape(-1)
    denominator = np.linalg.norm(expected)
    return float(np.linalg.norm(actual - expected) / denominator)


@pytest.mark.skipif(not has_package("lisagwresponse"), reason="lisagwresponse not installed")
def test_brute_force_links_shape():
    t = np.arange(0.0, 512.0, 2.0)
    orbits = default_lisaorbits("equal")
    source = GalacticBinaryParams(f0=1.0e-3, amplitude=1.0e-22, beta=0.2, lambda_=0.4)

    links = brute_force_links(source, orbits, t)

    assert set(links) == {"12", "23", "31", "13", "32", "21"}
    assert all(value.shape == t.shape for value in links.values())


@pytest.mark.skipif(not has_package("lisagwresponse") or not has_package("pytdi"), reason="reference dependencies not installed")
def test_brute_force_pair_and_xyz_shape():
    t = np.arange(0.0, 512.0, 2.0)
    orbits = default_lisaorbits("equal")
    state = state_from_lisaorbits(orbits, t)
    source = GalacticBinaryParams(f0=1.0e-3, amplitude=1.0e-22, beta=0.2, lambda_=0.4)

    pairs = brute_force_pair_channels(state, source, orbits)
    xyz = brute_force_xyz(state, source, orbits, measurement_order=3, delay_order=3)

    assert pairs.shape == (t.size, 3)
    assert set(xyz) == {"X", "Y", "Z"}


@pytest.mark.skipif(not has_package("lisagwresponse"), reason="lisagwresponse not installed")
def test_lisagwresponse_links_match_local_physical_strain_convention():
    t = np.arange(1000.0, 3048.0, 2.0)
    orbits = default_lisaorbits("equal")
    state = state_from_lisaorbits(orbits, t)
    source = GalacticBinaryParams(
        f0=1.0e-3,
        fdot=1.0e-18,
        amplitude=1.0e-21,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.7,
        phi0=0.8,
    )

    local = link_responses(state, source)
    reference = brute_force_links(source, orbits, t)
    errors = {label: _relative_l2(np.real(local[label]), reference[label]) for label in LINK_LABELS}

    assert max(errors.values()) < 5.0e-4, errors


@pytest.mark.skipif(not has_package("lisagwresponse") or not has_package("pytdi"), reason="reference dependencies not installed")
def test_pytdi_xyz_from_lisagwresponse_links_matches_local_xyz_convention():
    t = np.arange(1000.0, 3048.0, 2.0)
    orbits = default_lisaorbits("equal")
    state = state_from_lisaorbits(orbits, t)
    source = GalacticBinaryParams(
        f0=1.0e-3,
        fdot=1.0e-18,
        amplitude=1.0e-21,
        beta=0.2,
        lambda_=0.4,
        psi=0.3,
        inclination=0.7,
        phi0=0.8,
    )

    local = compute_xyz(state, source, generation=1, measurement_order=3, delay_order=3)
    reference = brute_force_xyz(state, source, orbits, generation=1, measurement_order=3, delay_order=3)
    errors = {channel: _relative_l2(np.real(local[channel]), np.real(reference[channel])) for channel in ("X", "Y", "Z")}

    assert max(errors.values()) < 1.0e-3, errors


@pytest.mark.parametrize('inclination,beta',[(0.,.2),(np.pi/2,np.pi/2),(1.21,-.7)])
def test_native_circular_adapter_preserves_full_polarization_tensor(inclination,beta):
    pytest.importorskip('lisagwresponse')
    from egb_multi.external import cartesian_frame_rotation
    from egb_multi.reference import lisagwresponse_galactic_binary
    source=GalacticBinaryParams(f0=.003,fdot=1e-16,amplitude=1e-21,
        beta=beta,lambda_=-1.,psi=.71,inclination=inclination,phi0=.617)
    native=lisagwresponse_galactic_binary(source,default_lisaorbits('equal'))
    t=np.arange(32)*7.1
    phase=2*np.pi*source.f0*t+np.pi*source.fdot*t**2+source.phi0
    local_tensor=np.real(source.polarization_matrix()[None,:,:]*np.exp(1j*phase)[:,None,None])
    rotation=cartesian_frame_rotation('ecliptic','icrs')
    expected=rotation @ local_tensor @ rotation.T
    ep=np.outer(native.u,native.u)-np.outer(native.v,native.v)
    ec=np.outer(native.u,native.v)+np.outer(native.v,native.u)
    actual=native.compute_hplus(t)[:,None,None]*ep+native.compute_hcross(t)[:,None,None]*ec
    assert _relative_l2(actual,expected)<3e-15
