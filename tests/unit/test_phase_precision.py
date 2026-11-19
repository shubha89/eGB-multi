"""Long-epoch source arithmetic against an independent decimal phase law."""
from decimal import Decimal,localcontext

import numpy as np
import pytest

from egb_multi.phase import periodic_linear_phase,periodic_phase

PERIOD='6.2831853071795864769252867665590057683943387987502116419498891846156'


def independent_phase(rate,times,t0,phase0,rate_dot=0.):
    answer=[]
    with localcontext() as context:
        context.prec=65
        for t in np.asarray(times).ravel():
            delta=Decimal.from_float(float(t))-Decimal.from_float(float(t0))
            phase=(Decimal.from_float(float(rate))*delta+Decimal.from_float(float(phase0))
                   +Decimal.from_float(float(rate_dot))*delta*delta/2)
            answer.append(float(phase%Decimal(PERIOD)))
    return np.array(answer).reshape(np.shape(times))


@pytest.mark.parametrize('rate',[np.pi*1e-4,np.pi*.001,np.pi*.01,.7])
@pytest.mark.parametrize('t0',[0.,1234567.12345,-31557600.])
def test_linear_phase_against_independent_decimal(rate,t0):
    rng=np.random.default_rng(137)
    t=rng.uniform(-1e8,1e8,500)
    actual=periodic_linear_phase(rate,t,t0=t0,phase0=.713)
    expected=independent_phase(rate,t,t0,.713)
    assert np.max(np.abs(np.exp(1j*actual)-np.exp(1j*expected)))<3e-15
    assert np.max(np.abs(actual))<=np.pi


def test_integrated_phase_offset_preserved():
    angle=np.array([-1e8,1e8,123456.789])
    expected=independent_phase(1.,angle,0.,.73)
    actual=periodic_phase(angle,offset=.73)
    assert np.max(np.abs(np.exp(1j*actual)-np.exp(1j*expected)))<3e-15


def test_jax_phase_and_analytic_jvp():
    jax=pytest.importorskip('jax');import jax.numpy as jnp
    from egb_multi.eccentric_jax import _periodic_phase_jax
    jax.config.update('jax_enable_x64',True)
    t=jnp.asarray(np.random.default_rng(138).uniform(1e6,3e7,256))
    rate=np.pi*.01;rate_dot=1e-16;t0=123456.345;phase0=.713
    phase=jax.jit(_periodic_phase_jax)(rate,t,t0,phase0,rate_dot)
    expected=independent_phase(rate,np.asarray(t),t0,phase0,rate_dot)
    assert np.max(np.abs(np.exp(1j*np.asarray(phase))-np.exp(1j*expected)))<4e-15
    def wave(r):return jnp.sin(_periodic_phase_jax(r,t,t0,phase0,rate_dot))
    gradient=jax.jit(jax.jacfwd(wave))(rate)
    np.testing.assert_allclose(gradient,(np.asarray(t)-t0)*np.cos(expected),rtol=5e-12,atol=1e-7)


@pytest.mark.parametrize('eccentricity',[0.,.1,.6])
def test_numpy_jax_long_epoch_source_agreement(eccentricity):
    pytest.importorskip('jax')
    from egb_multi.eccentric import EccentricBinaryParams,eccentric_complex_strain
    from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax
    source=EccentricBinaryParams(mean_motion=np.pi*.003,eccentricity=eccentricity,
        m1_solar=1.1,m2_solar=.4,phi0=.7,inclination=.9)
    times=27612900.+np.arange(500)*1.25+.13
    expected=eccentric_complex_strain(source,times)
    actual=np.asarray(eccentric_complex_strain_batch_jax(source,times))[0]
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<3e-12
