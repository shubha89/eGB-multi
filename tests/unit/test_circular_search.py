"""Known analytic chirps verify the staged optimizer independently of eGB/TDI."""

from dataclasses import replace

import numpy as np
import pytest

from egb_multi.circular_search import CircularFrequencyChirpSearch, circular_source_at_epoch
from egb_multi.comparison import AESeries,AEPSD,AEInnerProduct,TDIConvention
from egb_multi.eccentric import EccentricBinaryParams


def problem(cross=False):
    t = 1000.+np.arange(1024)*20.
    duration, midpoint = t.size*20.,.5*(t[0]+t[-1])
    convention = TDIConvention(2,3,3,'analytic_test','fixed_test_delays')
    def response(source):
        tau = t-source.t0
        phase = 2*source.phi0 + 2*source.mean_motion*tau + np.pi*source.fdot*tau**2
        u = np.cos(source.inclination)
        # Independent tensor projection, with a nonzero extra temporal basis.
        # This is a controlled synthetic model, not a physical LISA prediction.
        hp = (1+u*u)*np.cos(phase)+(1-u*u)*.07*np.sin(2*np.pi*.0015*t)
        hc = 2*u*np.sin(phase)
        p,q = np.cos(2*source.psi),np.sin(2*source.psi)
        channels = {ch:hp*(p*fp-q*fq)+hc*(q*fp+p*fq)
            for ch,fp,fq in (('A',1.,.22),('E',.35,1.))}
        return AESeries(t,channels,convention)
    prototype = EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=0.,t0=midpoint)
    truth = replace(prototype,mean_motion=np.pi*(.003+.713/duration),fdot=-1.117/duration**2,
        inclination=.89,phi0=.9,psi=.63)
    data = response(truth)
    f = np.fft.rfftfreq(t.size,20.)
    provenance = {'sidedness':'one_sided','scaling':'density','noise_model':'test_constant_covariance',
        'CSD_convention':'S_AE=E[tilde_A*conj(tilde_E)]'}
    psd = AEPSD(f,{'A':np.full(f.size,2.),'E':np.full(f.size,3.)},convention,provenance,
        np.full(f.size,.2+.3j) if cross else None)
    metric = AEInnerProduct(data,psd,(.001,.008))
    search = CircularFrequencyChirpSearch(metric,prototype,response,
        frequency_scaled_bounds=(-2.,2.),chirp_scaled_bounds=(-3.,3.),
        evaluation_time_bounds_s=(t[0],t[-1]))
    return search,truth,response


@pytest.mark.parametrize('cross',[False,True])
def test_offset_circular_frequency_and_negative_chirp_are_recovered(cross):
    search,truth,response = problem(cross)
    norm, timestamps, mask = search.metric.reference_norm,search.metric.t.copy(),search.metric.mask.copy()
    restricted = max(search.evaluate((0.,g))['profile']['overlap'] for g in (-2.,0.,2.))
    first = search.run(coarse_shape=(7,3),starts=1)
    final = search.refine(first)
    best = final['best']
    truth_coordinates = search.coordinates_of(truth)
    assert abs(best['coordinates'][0]-truth_coordinates[0]) < 2e-3
    assert abs(best['coordinates'][1]-truth_coordinates[1]) < 3e-3
    assert best['profile']['overlap'] > 1.-2e-9
    assert best['profile']['overlap'] >= first['best']['profile']['overlap'] >= restricted
    recovered = EccentricBinaryParams(**best['source'])
    assert search.metric.compare(response(recovered))['overlap'] > 1.-2e-9
    assert search.metric.reference_norm == norm
    np.testing.assert_array_equal(search.metric.t,timestamps)
    np.testing.assert_array_equal(search.metric.mask,mask)
    assert final['fixed_masses_solar'] == [truth.m1_solar,truth.m2_solar]


def test_reference_epoch_transform_preserves_analytic_chirping_response():
    _,truth,response = problem()
    original = response(truth)
    shifted = response(circular_source_at_epoch(truth,truth.t0+197.))
    for ch in ('A','E'):
        np.testing.assert_allclose(original.channels[ch],shifted.channels[ch],rtol=0,atol=2e-13)


def test_invalid_search_domain_fails_before_response_evaluation():
    search,_,response = problem()
    with pytest.raises(ValueError,match='nonpositive'):
        CircularFrequencyChirpSearch(search.metric,search.prototype,response,
            frequency_scaled_bounds=(-100.,2.),chirp_scaled_bounds=(-3.,3.),
            evaluation_time_bounds_s=(search.metric.t[0],search.metric.t[-1]))
    with pytest.raises(ValueError,match='outside'):
        search.evaluate((2.1,0.))
