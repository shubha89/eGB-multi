"""Diagnostic stationary covariance from native noise definitions and pyTDI.

Frozen delay operators commute. This model includes native test-mass and
SCI/TMI/REF OMS spectra, but omits moving-arm laser residuals and filter
transients. It is not a certified replacement for the native simulation.
Covariance uses E[A(f) conj(E(f))], in fractional-frequency units.
"""

import numpy as np
from math import factorial

from .constants import C_M_PER_S
from .tdi import LINK_LABELS


def frozen_raw_tdi_transfer(frequency_hz, delays, *, generation=2):
    """Return analytic XYZ response to each raw SCI/TMI/REF measurement."""
    from pytdi.michelson import X1, Y1, Z1, X2, Y2, Z2
    f = np.asarray(frequency_hz,dtype=float)
    if f.ndim != 1 or np.any(f<=0) or not np.all(np.isfinite(f)):
        raise ValueError('Finite positive one-dimensional frequencies required')
    if set(delays) != set(LINK_LABELS) or any(not np.isfinite(v) or v<=0 for v in delays.values()):
        raise ValueError('Six finite positive constant link delays required')
    if generation not in (1,2):
        raise ValueError('TDI generation must be 1 or 2')
    combinations = (X1,Y1,Z1) if generation==1 else (X2,Y2,Z2)
    measurements = [f'{kind}_{label}' for kind in ('sci','tmi','ref') for label in LINK_LABELS]
    transfer = np.zeros((len(f),3,len(measurements)),dtype=complex)
    for ch,combination in enumerate(combinations):
        for measurement,terms in combination.components.items():
            column = measurements.index(measurement)
            # Collapse equal frozen delays before evaluating; avoids summing
            # nearly cancelling large exponentials at very low frequencies.
            polynomial = {}
            for coefficient,chain in terms:
                counts = tuple(sum(op==f'D_{label}' for op in chain)-sum(op==f'A_{label}' for op in chain) for label in LINK_LABELS)
                if sum(abs(v) for v in counts)>len(chain):
                    raise ValueError('Invalid delay polynomial')
                if any(op[0] not in ('D','A') or op[2:] not in delays for op in chain):
                    raise ValueError('Unknown native delay operator')
                polynomial[counts] = polynomial.get(counts,0.)+coefficient
            coefficients=np.array(list(polynomial.values()))
            counts=np.array(list(polynomial))
            delay=np.array([sum(n*delays[label] for n,label in zip(path,LINK_LABELS)) for path in counts])
            small=2*np.pi*f*np.max(abs(delay))<=.2
            for tau,coefficient in zip(delay,coefficients):
                transfer[~small,ch,column]+=coefficient*np.exp(-2j*np.pi*f[~small]*tau)
            # Resolve infrared cancellation analytically instead of subtracting
            # unit exponentials. Constant and linear moments use exact native
            # coefficients/counts; higher moments use the same float64 delays.
            # At |omega*tau|<=0.2, terms through degree 18 leave <1e-29 absolute
            # Taylor remainder per coefficient. This is numerical evaluation
            # of the identical frozen polynomial, not a PSD floor.
            z=-2j*np.pi*f[small]
            transfer[small,ch,column]=coefficients.sum()
            linear_counts=coefficients@counts
            linear=sum(n*delays[label] for n,label in zip(linear_counts,LINK_LABELS))
            transfer[small,ch,column]+=z*linear
            for power in range(2,19):
                transfer[small,ch,column]+=z**power*(coefficients@delay**power)/factorial(power)
    return transfer,measurements


def frozen_native_secondary_covariance(frequency_hz, delays, instrument, *, generation=2):
    """Project native discrete secondary spectra through frozen pyTDI.

    Test-mass velocity appears in local TMI frequency with coefficient -2/c;
    its beatnote sign is immaterial for independent-noise covariance. Native
    optical-offset corrections of relative order 1e-7 are omitted and recorded.
    OMS noises are independent readout additions to each raw measurement.
    Noise-driven laser locking is canceled in the frozen TDI idealization.
    """
    from lisainstrument.orbiting.constellation_enums import MosaID
    f = np.asarray(frequency_hz,dtype=float)
    xyz,measurements = frozen_raw_tdi_transfer(f,delays,generation=generation)
    rotation = np.array([[-1,0,1],[1,-2,1]],dtype=float)/np.sqrt(np.array([[2.],[6.]]))
    h = np.einsum('ax,fxm->fam',rotation,xyz)
    definitions = instrument.get_noise_defs()
    covariance = np.zeros((len(f),2,2),dtype=complex)
    components = {}
    for kind in ('sci','tmi','ref'):
        contribution = np.zeros_like(covariance)
        for label in LINK_LABELS:
            mosa = MosaID(label)
            definition = getattr(definitions,f'noise_def_oms_{kind}_carrier')(mosa)
            spectrum = 2*definition.discrete_psd_twosided(f)
            if kind=='tmi':
                spectrum = spectrum+(2/C_M_PER_S)**2*2*definitions.noise_def_testmass(mosa).discrete_psd_twosided(f)
            column = h[:,:,measurements.index(f'{kind}_{label}')]
            contribution += np.einsum('fa,fb,f->fab',column,column.conj(),spectrum)
        # Enforce the algebraic Hermitian symmetry at roundoff only. No
        # eigenvalue floor, shrinkage or statistical smoothing is applied.
        contribution = (contribution+contribution.swapaxes(1,2).conj())/2
        covariance += contribution
        components[kind] = contribution
    return covariance,components


def moving_native_secondary_symbol(frequency_hz, times, delays, instrument, epoch_indices, *, delay_order=7):
    """Local secondary covariance symbol using native nested moving delays.

    Retains native frequency-delay derivative factors. Averaging this symbol
    is an adiabatic PSD approximation; it omits frequency-bin covariance,
    finite-window convolution and noise-driven laser-locking residuals. The
    caller must not confuse it with an exact finite-observation covariance.
    """
    from pytdi.michelson import X2,Y2,Z2
    from lisainstrument.orbiting.constellation_enums import MosaID
    f=np.asarray(frequency_hz,dtype=float);t=np.asarray(times,dtype=float)
    selected=np.asarray(epoch_indices,dtype=int)
    if f.ndim!=1 or np.any(f<=0) or not np.all(np.isfinite(f)):
        raise ValueError('Finite positive frequencies required')
    if t.ndim!=1 or len(t)<16 or not np.all(np.isfinite(t)) or t[1]<=t[0] or not np.allclose(np.diff(t),t[1]-t[0]):
        raise ValueError('Uniform finite geometry time grid required')
    if selected.ndim!=1 or selected.size==0 or np.any(selected<4*(delay_order+1)) or np.any(selected>=len(t)-4*(delay_order+1)):
        raise ValueError('Selected epochs require complete native interpolation support')
    if set(delays)!=set(LINK_LABELS) or any(np.shape(v)!=t.shape or np.any(np.asarray(v)<=0) or not np.all(np.isfinite(v)) for v in delays.values()):
        raise ValueError('Six finite positive delays aligned with times required')
    raw={f'd_{label}':np.asarray(v) for label,v in delays.items()}
    derivatives={key:np.gradient(value,t[1]-t[0],edge_order=2) for key,value in raw.items()}
    operators=[item.build_shifts(raw,1/(t[1]-t[0]),derivatives,order=delay_order) for item in (X2,Y2,Z2)]
    rotation=np.array([[-1,0,1],[1,-2,1]],dtype=float)/np.sqrt(np.array([[2.],[6.]]))
    covariance=np.zeros((selected.size,f.size,2,2),dtype=complex)
    definitions=instrument.get_noise_defs()
    for kind in ('sci','tmi','ref'):
        for label in LINK_LABELS:
            name=f'{kind}_{label}';xyz=np.zeros((selected.size,f.size,3),dtype=complex)
            for ch,(combination,(shifts,rates)) in enumerate(zip((X2,Y2,Z2),operators)):
                for factor,chain in combination.components.get(name,[]):
                    shift=np.broadcast_to(shifts[tuple(chain)],t.shape)[selected]
                    rate=1+np.broadcast_to(rates[tuple(chain)],t.shape)[selected]
                    xyz[:,:,ch]+=factor*rate[:,None]*np.exp(2j*np.pi*shift[:,None]*f[None,:])
            h=np.einsum('ax,tfx->tfa',rotation,xyz)
            mosa=MosaID(label)
            spectrum=2*getattr(definitions,f'noise_def_oms_{kind}_carrier')(mosa).discrete_psd_twosided(f)
            if kind=='tmi': spectrum=spectrum+(2/C_M_PER_S)**2*2*definitions.noise_def_testmass(mosa).discrete_psd_twosided(f)
            covariance+=np.einsum('tfa,tfb,f->tfab',h,h.conj(),spectrum)
    return (covariance+covariance.swapaxes(2,3).conj())/2
