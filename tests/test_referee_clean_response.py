"""Causal noise-seed isolation for the revision's clean GW response."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from egb_multi.comparison import TDIConvention
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.external import has_package
from egb_multi.noise import simulate_lisainstrument_tdi_noise

PATH = Path(__file__).resolve().parents[1]/'benchmarks/referee_task_2_objective_orientation.py'
spec = importlib.util.spec_from_file_location('clean_response_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.skipif(any(not has_package(name) for name in ('jax','pytdi','lisaorbits','lisainstrument')),
    reason='LISA ecosystem and JAX required')
def test_noise_seed_changes_noise_but_cannot_change_clean_response(tmp_path):
    samples, dt, halo = 2048, 5., 200
    times = np.arange(samples)*dt
    path = tmp_path/'retarded_orbits.h5'
    orbit = audit.make_common_orbit_file(path, samples*dt, model='retarded')
    convention = TDIConvention(2,45,5,'analytic_trailing_ecliptic','deterministic_TCB_retarded')
    source = EccentricBinaryParams(mean_motion=np.pi*.003, eccentricity=.1,
        inclination=.8, phi0=.213, psi=.4, beta=.2, lambda_=-1., m1_solar=.8, m2_solar=.6)
    signals, noises = [], []
    for seed in (20261007,20261008):
        noise = simulate_lisainstrument_tdi_noise(size=samples, dt=dt, seed=seed,
            generation=2, order=45, delay_order=5, enabled_noises=('laser','test-mass','oms'),
            instrument_kwargs={'orbits':str(path), 'orbit_dataset':'tcb/ltt', 't0':0.})
        np.testing.assert_array_equal(noise.t, times)
        response, delays = audit.make_clean_response(orbit,times,convention,halo)
        for label in delays:
            np.testing.assert_allclose(noise.delays[label],delays[label],rtol=0,atol=1e-10)
        # Deliberately corrupt measurements after the above contract check.
        # This must affect neither timestamps, geometry nor clean GW content.
        noises.append(noise.aet['A'].copy())
        noise.t[:] += .123
        for array in noise.delays.values():
            array[:] += 1e-3
        signals.append(response(source))
    assert not np.array_equal(noises[0],noises[1])
    np.testing.assert_array_equal(signals[0].t,signals[1].t)
    for channel in ('A','E'):
        np.testing.assert_array_equal(signals[0].channels[channel],signals[1].channels[channel])
