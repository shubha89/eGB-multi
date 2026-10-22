"""Independent LISA GW Response checks of link/TDI convention boundaries."""
import numpy as np
import pytest

from egb_multi.external import default_lisaorbits, state_from_lisaorbits, has_package, cartesian_frame_rotation
from egb_multi.eccentric import EccentricBinaryParams, eccentric_polarizations
from egb_multi.eccentric_jax import precompute_jax_link_geometry, eccentric_links_jax
from egb_multi.tdi import LINK_LABELS, pytdi_data_from_links
from egb_multi.noise import xyz_from_pytdi_data, aet_from_xyz

pytestmark = pytest.mark.skipif(
    not all(has_package(p) for p in ('lisaorbits','lisagwresponse','pytdi','jax')),
    reason='optional LISA ecosystem packages are required')


def compare_response(source, t):
    from lisagwresponse import ResponseFromStrain

    orbits=default_lisaorbits('esa-trailing')
    state=state_from_lisaorbits(orbits,t,frame='ecliptic')
    geometry=precompute_jax_link_geometry(state,orbits=orbits,frame='ecliptic')
    rotation=cartesian_frame_rotation('ecliptic','icrs')
    sky=-rotation @ source.propagation_direction()
    ra,dec=np.arctan2(sky[1],sky[0]),np.arcsin(sky[2])
    u=np.array([np.sin(ra),-np.cos(ra),0.])
    v=np.array([-np.sin(dec)*np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)])
    native_ep=np.outer(u,u)-np.outer(v,v)
    native_ec=np.outer(u,v)+np.outer(v,u)
    ep,ec=source.polarization_basis()
    c,s=np.cos(2*source.psi),np.sin(2*source.psi)
    rotated_ep=rotation @ (c*ep-s*ec) @ rotation.T
    rotated_ec=rotation @ (s*ep+c*ec) @ rotation.T
    # Project the same physical tensor into the native ICRS polarization
    # basis; this includes both the frame rotation and cross-sign convention.
    pp,pc=.5*np.sum(rotated_ep*native_ep),.5*np.sum(rotated_ep*native_ec)
    cp,cc=.5*np.sum(rotated_ec*native_ep),.5*np.sum(rotated_ec*native_ec)
    class NativeStrain(ResponseFromStrain):
        def compute_hplus(self, times):
            hp,hc=eccentric_polarizations(source,times,physics_mode='1pn')
            return hp*pp+hc*cp
        def compute_hcross(self, times):
            hp,hc=eccentric_polarizations(source,times,physics_mode='1pn')
            return hp*pc+hc*cc

    positions={}
    for axis in range(3):
        positions[axis]={sc:(lambda times, sc=sc, axis=axis:
            np.asarray(orbits.compute_position(np.asarray(times),[sc]))[:,0,axis]) for sc in (1,2,3)}
    ltt={int(label):(lambda times,label=label:
        np.asarray(orbits.compute_ltt(np.asarray(times),[int(label)]))[:,0]) for label in LINK_LABELS}
    native=NativeStrain(orbits='',ra=ra,dec=dec,
        x=positions[0],y=positions[1],z=positions[2],ltt=ltt,shift_sun='never')
    np.testing.assert_allclose(native.k,rotation @ source.propagation_direction(),rtol=0,atol=5e-16)
    external=native.compute_gw_response(t,[int(label) for label in LINK_LABELS]).T
    local=eccentric_links_jax(source,geometry,batch_size=1)
    approximate=eccentric_links_jax(source,precompute_jax_link_geometry(state),batch_size=1)
    wrong_frame=eccentric_links_jax(source,precompute_jax_link_geometry(
        state_from_lisaorbits(orbits,t,frame='icrs'),orbits=orbits,frame='icrs'),batch_size=1)
    local_array=np.array([local[label].real for label in LINK_LABELS])
    approximate_array=np.array([approximate[label].real for label in LINK_LABELS])
    delays={label:geometry['length'][i] for i,label in enumerate(LINK_LABELS)}
    native_data=pytdi_data_from_links(state,dict(zip(LINK_LABELS,external)),link_delays=delays)
    local_data=pytdi_data_from_links(state,local,link_delays=delays)
    native_aet=aet_from_xyz(xyz_from_pytdi_data(native_data,generation=2,order=45))
    local_aet=aet_from_xyz(xyz_from_pytdi_data(local_data,generation=2,order=45))
    halo=int(np.ceil(8*np.max(geometry['length'])/np.diff(t)[0]))+4*46
    sl=slice(halo,-halo)
    relative=lambda a,b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
    return dict(link_relative_l2_error=relative(local_array,external),
        instantaneous_geometry_relative_l2_error=relative(approximate_array,external),
        unconverted_frame_relative_l2_error=relative(np.array([wrong_frame[label].real for label in LINK_LABELS]),external),
        aet_relative_l2_errors={ch:relative(local_aet[ch][sl],native_aet[ch][sl]) for ch in ('A','E','T')},
        maximum_geometric_vs_native_ltt_difference_s=float(np.max(np.abs(
            np.array([state.arm_length(*pair) for pair in ((0,1),(1,2),(2,0),(0,2),(2,1),(1,0))])-geometry['length']))))


@pytest.mark.parametrize('frequency,eccentricity,psi',[(.001,0.,0.),(.01,.3,.71),(.03,.1,1.13)])
def test_retarded_egb_links_and_tdi_match_lisa_gw_response(frequency,eccentricity,psi):
    source=EccentricBinaryParams(mean_motion=np.pi*frequency,eccentricity=eccentricity,
        m1_solar=.8,m2_solar=.6,inclination=1.21,beta=.2,lambda_=-1.,psi=psi,phi0=.617)
    metrics=compare_response(source,np.arange(4096)*2.5)
    assert metrics['link_relative_l2_error'] < 5e-10
    assert max(metrics['aet_relative_l2_errors'].values()) < 5e-9


def test_retarded_geometry_rejects_mismatched_state():
    from egb_multi.geometry import LISAState
    orbit=default_lisaorbits('esa-trailing')
    state=state_from_lisaorbits(orbit,np.arange(16)*5.)
    with pytest.raises(ValueError,match='different positions'):
        precompute_jax_link_geometry(LISAState(state.t,state.positions+1.),orbits=orbit)


def test_icrs_ecliptic_rotation_preserves_distances_and_inverts():
    forward=cartesian_frame_rotation('ecliptic','icrs')
    backward=cartesian_frame_rotation('icrs','ecliptic')
    np.testing.assert_allclose(forward.T @ forward,np.eye(3),atol=7e-16)
    np.testing.assert_allclose(backward @ forward,np.eye(3),atol=7e-16)
    assert np.isclose(np.linalg.det(forward),1.)
