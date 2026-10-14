"""Optional full-band circular spline-Fourier objective for annual searches.

Uses the same envelope spline as the sampled evaluator, every comparison bin,
the complete A/E covariance, and all six physically constrained responses.
Only the spline's tiny internal-jump sampling correction is approximated;
validate against sampled FFTs and native TDI before using it. No noise model,
window, mass bound, sky domain or physical template coefficient is changed.
"""

from dataclasses import replace

import numpy as np
from scipy.interpolate import CubicSpline

from .circular_fit import PreparedCircularAmplitudeBasis,profile_circular_orientation
from .spline_fourier import CubicEnvelopeFourier


class CircularSpectralTDI:
    def __init__(self,response,metric):
        if metric.t[0]!=0 or not np.array_equal(response.t,metric.t) or response.convention!=metric.convention:
            raise ValueError("Same immutable [0,T) observation and convention required")
        if np.shape(metric.reference)!=(2,len(metric.frequency_hz)):
            raise ValueError("Stationary spectral metric required")
        self.response,self.metric=response,metric
        self.duration=len(metric.t)*metric.dt
        self.bins=np.rint(metric.frequency_hz*self.duration).astype(int)
        self.radial=CubicEnvelopeFourier(response.coarse,self.duration,metric.dt,self.bins)
        self.kernel_key=self.kernels=None

    def __call__(self,source,*args,**kwargs):
        return self.response(source,*args,**kwargs)

    def circular_basis(self,source):
        return self.response.circular_basis(source)

    def prepared_basis(self,source):
        canonical=replace(source,phi0=0.)
        frequency=source.mean_motion/np.pi
        key=(frequency,source.t0)
        if key!=self.kernel_key:
            self.kernels=[CubicEnvelopeFourier(self.response.coarse,self.duration,self.metric.dt,
                sign*self.bins,carrier_hz=frequency,epoch_s=source.t0) for sign in (1,-1)]
            self.kernel_key=key
        envelope,radial=self.response.coarse_envelope(canonical)
        spline=CubicSpline(self.response.coarse,envelope,axis=0)
        positive=self.kernels[0].transform(spline)
        negative=self.kernels[1].transform(spline).conj()
        real=(positive+negative)/2;imag=(positive-negative)/(2j)
        dc=self.radial.transform(CubicSpline(self.response.coarse,radial,axis=0))
        cosine,sine=np.cos(2*source.phi0),np.sin(2*source.phi0)
        values=[cosine*real[:,:,0]-sine*imag[:,:,0],sine*real[:,:,0]+cosine*imag[:,:,0],
            cosine*real[:,:,1]-sine*imag[:,:,1],sine*real[:,:,1]+cosine*imag[:,:,1],dc[:,:,0],dc[:,:,1]]
        prepared=np.array([value.T for value in values])*self.metric.sqrt_weight[None,:,:]
        if self.metric.covariance_whitener is not None:
            prepared=np.einsum("fij,bjf->bif",self.metric.covariance_whitener,prepared)
        return PreparedCircularAmplitudeBasis(source,prepared)

    def profile_source(self,metric,source,*,local_starts=8):
        if metric is not self.metric:raise ValueError("Spectral objective has one immutable metric")
        profile=profile_circular_orientation(metric,self.prepared_basis(source),local_starts=local_starts)
        profile["response_evaluator"]="full_band_cubic_envelope_Fourier_with_sampled_endpoint_control"
        profile["sampling_approximation"]="Internal third-derivative spline jumps omitted; validate sampled/native controls"
        return profile
