"""Cheap annual frequency/chirp/sky proposals from circular envelopes.

These are seed scores, never fitting factors. A narrow positive-frequency
analytic carrier and four unconstrained oscillatory amplitudes approximate
the circular waveform for proposing starts. Radial terms, negative-carrier
tails, envelope changes with frequency offset and power outside the seed
window are omitted here. Every retained proposal must be regenerated and
profiled with the full six-response physical objective before optimization.
The approximation imposes no restriction on the final sky or mass search.
"""

from dataclasses import replace

import numpy as np
from scipy.interpolate import CubicSpline


def _batched_projection_scores(prepared, reference, reference_norm):
    """Same real-coefficient seed projection, batched over frequency offsets.

    This numerical calculation is only a proposal score. The eigenvalue
    threshold is the unchanged seed pseudoinverse rule, not a physical-mode
    reduction in the final six-response orientation fit.
    """
    prepared,reference=np.asarray(prepared),np.asarray(reference)
    if prepared.ndim!=4 or prepared.shape[1]!=4 or prepared.shape[2:]!=reference.shape:
        raise ValueError('Offset, four-basis, channel, frequency seed arrays required')
    if not np.all(np.isfinite(prepared)) or not np.all(np.isfinite(reference)) or not np.isfinite(reference_norm) or reference_norm<=0:
        raise ValueError('Finite seed arrays and positive reference norm required')
    gram=np.einsum('obcf,odcf->obd',prepared.conj(),prepared).real
    projection=np.einsum('obcf,cf->ob',prepared.conj(),reference).real
    scale=np.diagonal(gram,axis1=1,axis2=2).max(axis=1)
    if np.any(~np.isfinite(scale)) or np.any(scale<=0):raise ValueError('Invalid seed template norm')
    eigenvalues,vectors=np.linalg.eigh(gram/scale[:,None,None])
    kept=eigenvalues>1e-12*eigenvalues[:,-1,None]
    components=np.einsum('obi,ob->oi',vectors,projection)
    contributions=np.divide(components**2,eigenvalues,out=np.zeros_like(components),where=kept)
    scores=np.sqrt(contributions.sum(axis=1)/scale/reference_norm)
    if np.any(~np.isfinite(scores)) or np.any(scores>1+1e-10):raise ValueError('Invalid approximate seed projection')
    return scores


def rank_envelope_frequency_seeds(metric, response, prototype, basin, *, sky,
    chirp_scaled_values, seed_samples=2048, window_half_width_cycles=256):
    duration=len(metric.t)*metric.dt
    if metric.t[0]!=0 or prototype.eccentricity!=0 or prototype.phi0!=0:
        raise ValueError("Seed control requires [0,T), circular zero-phase prototype")
    if seed_samples%2 or seed_samples<2*(window_half_width_cycles+basin.half_width_cycles)+4:
        raise ValueError("Seed FFT must cover the complete window and frequency offsets")
    center=int(round(basin.center_hz*duration))
    offsets=np.arange(int(np.ceil(basin.center_hz*duration-basin.half_width_cycles))-center,
        int(np.floor(basin.center_hz*duration+basin.half_width_cycles))-center+1)
    if not len(offsets):raise ValueError("No integer annual frequency seeds")
    q=np.arange(-window_half_width_cycles,window_half_width_cycles+1)
    frequency=(center+q)/duration
    selected=(frequency>=metric.band_hz[0])&(frequency<=metric.band_hz[1])
    q=q[selected];frequency=frequency[selected]
    indices=center+q-int(round(metric.frequency_hz[0]*duration))
    if not len(q) or np.any(indices<0) or np.any(indices>=len(metric.frequency_hz)) or not np.allclose(metric.frequency_hz[indices],frequency,atol=1e-14,rtol=0):
        raise ValueError("Seed frequencies must be actual common objective bins")
    reference=metric.reference[:,indices]
    weight=metric.sqrt_weight[:,indices]
    whitener=None if metric.covariance_whitener is None else metric.covariance_whitener[indices]
    times=np.arange(seed_samples)*duration/seed_samples
    records=[]
    for gamma in chirp_scaled_values:
        if not np.isfinite(gamma):raise ValueError("Finite chirp seeds required")
        for sine_latitude,longitude in sky:
            source=replace(prototype,mean_motion=np.pi*center/duration,fdot=float(gamma/duration**2),
                beta=float(np.arcsin(sine_latitude)),lambda_=float(longitude%(2*np.pi)))
            envelope,_=response.coarse_envelope(source)
            baseband=CubicSpline(response.coarse,envelope,axis=0)(times)
            transformed=(duration/seed_samples)*np.fft.fft(baseband,axis=0)
            z=transformed[(q[None,:]-offsets[:,None])%seed_samples]
            plus=z[:,:,:,0].transpose(0,2,1);cross=z[:,:,:,1].transpose(0,2,1)
            raw=np.stack((plus/2,plus/(2j),cross/2,cross/(2j)),axis=1)
            prepared=raw*weight[None,None,:,:]
            if whitener is not None:prepared=np.einsum('fij,objf->obif',whitener,prepared)
            scores=_batched_projection_scores(prepared,reference,metric.reference_norm)
            index=int(np.argmax(scores));frequency_hz=(center+offsets[index])/duration
            best=dict(seed_score=float(scores[index]),frequency_hz=float(frequency_hz),chirp_scaled=float(gamma),
                sine_latitude=float(sine_latitude),longitude_rad=float(longitude),
                source=replace(source,mean_motion=float(np.pi*frequency_hz)))
            records.append(best)
    return sorted(records,key=lambda r:r["seed_score"],reverse=True)
