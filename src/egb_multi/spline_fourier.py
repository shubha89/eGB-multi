"""Fourier integral and sampled-grid endpoint control for a cubic envelope.

For uniform cubic intervals, integrate each polynomial times the exact carrier
exponential and sum interval coefficients with an FFT. The sampled [0,T) grid
adds its endpoint term and the closed Euler-Maclaurin endpoint correction.
Internal third-derivative jumps of the C2 spline are not corrected; that small
sampling approximation must be validated against an actual sampled FFT. No
signal window, frequency truncation, PSD floor or parameter reduction occurs.
"""

from math import factorial

import numpy as np
from scipy.special import bernoulli


def _moments(theta):
    """Integral from 0 to 1 of u**p exp(z*u), p=0..3, z=-i*theta."""
    z=-1j*np.asarray(theta);result=np.empty((4,len(z)),complex)
    small=abs(z)<.5
    for power in range(4):
        result[power,small]=sum(z[small]**n/factorial(n)/(n+power+1) for n in range(22))
    if np.any(~small):
        x=z[~small];exponential=np.exp(x)
        result[0,~small]=np.expm1(x)/x
        for power in range(1,4):result[power,~small]=exponential/x-power*result[power-1,~small]/x
    return result


def _endpoint_functions(z):
    """Derivatives through order 3 of coth(z/2)/2 - 1/z."""
    z=np.asarray(z,complex);result=np.empty((4,len(z)),complex);small=abs(z)<.25
    coefficients=bernoulli(22)
    for derivative in range(4):
        result[derivative,small]=0.
        for m in range(1,12):
            power=2*m-1
            if power>=derivative:
                factor=coefficients[2*m]/factorial(2*m)*factorial(power)/factorial(power-derivative)
                result[derivative,small]+=factor*z[small]**(power-derivative)
    if np.any(~small):
        x=z[~small];coth=1/np.tanh(x/2);csch2=1/np.sinh(x/2)**2
        result[0,~small]=.5*coth-1/x
        result[1,~small]=-.25*csch2+1/x**2
        result[2,~small]=.25*csch2*coth-2/x**3
        result[3,~small]=-.25*csch2*coth**2-.125*csch2**2+6/x**4
    return result


class CubicEnvelopeFourier:
    """Reusable kernels for the same physical Fourier bins and carrier."""
    def __init__(self, times, duration, dt, indices, *, carrier_hz=0., epoch_s=0.):
        grid=np.asarray(times,float);bins=np.asarray(indices)
        if grid.ndim!=1 or len(grid)<5 or not np.allclose(np.diff(grid),grid[1]-grid[0],rtol=1e-11,atol=1e-8):
            raise ValueError("Uniform cubic envelope grid required")
        if bins.ndim!=1 or not np.issubdtype(bins.dtype,np.integer) or not np.all(np.isfinite(bins)):
            raise ValueError("Integer physical Fourier bins required")
        if dt<=0 or duration<=0 or not np.isclose(duration/dt,round(duration/dt),atol=1e-8,rtol=0):
            raise ValueError("Exact physical sampled observation required")
        step=grid[1]-grid[0];count=int(round(duration/step));first=int(round(-grid[0]/step))
        if count<4 or first<0 or first+count>=len(grid) or abs(grid[first])>1e-8 or abs(grid[first+count]-duration)>1e-8:
            raise ValueError("Cubic knots must include both physical observation boundaries")
        self.grid,self.first,self.count,self.step=grid,first,count,step
        self.bins=bins;self.lookup=bins%count
        self.phase=np.exp(2j*np.pi*carrier_hz*np.arange(count)*step)
        self.anchor=np.exp(-2j*np.pi*carrier_hz*epoch_s)
        kappa=2*np.pi*(bins/duration-carrier_hz)
        if np.max(abs(kappa*dt))>=2*np.pi:
            raise ValueError("Endpoint alias correction requires carrier/bin separation below sampling frequency")
        self.moments=_moments(kappa*step)*step**np.arange(1,5)[:,None]
        self.endpoint=_endpoint_functions(-1j*kappa*dt)*dt**np.arange(1,5)[:,None]/np.array([factorial(p) for p in range(4)])[:,None]
        self.phase_end=np.exp(2j*np.pi*carrier_hz*duration)
        self.dt=dt

    def transform(self, spline):
        if not np.array_equal(spline.x,self.grid) or spline.c.shape[0]!=4:
            raise ValueError("Identical cubic grid required")
        coefficients=spline.c[::-1,self.first:self.first+self.count]
        shape=coefficients.shape[2:];phase=self.phase.reshape((-1,)+(1,)*len(shape))
        transformed=np.fft.fft(coefficients*phase[None,:],axis=1)
        result=np.zeros((len(self.bins),)+shape,complex)
        for power in range(4):
            result+=self.moments[power].reshape((-1,)+(1,)*len(shape))*transformed[power,self.lookup]
        start=spline(0.);end=spline(self.count*self.step)
        result+=self.dt/2*(start-self.phase_end*end)
        for derivative in range(4):
            difference=self.phase_end*spline(self.count*self.step,nu=derivative)-spline(0.,nu=derivative)
            result+=self.endpoint[derivative].reshape((-1,)+(1,)*len(shape))*difference
        return self.anchor*result
