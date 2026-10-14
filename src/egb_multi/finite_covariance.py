"""Finite stationary noise covariance control on the same real Fourier band.

The usual PSD metric replaces a finite Toeplitz covariance by a circulant
covariance. This diagnostic retains the Toeplitz covariance R[i-j] supplied
by a stationary multichannel spectrum, restricts it to the declared real
Fourier subspace, and applies its inverse using preconditioned CG. Working
with real samples includes both ordinary and pseudo-covariance of complex
positive Fourier bins automatically. No dense annual covariance is stored.

The preparation result stores whitened data and its covariance solve; the
symmetric bilinear form averages the two solve directions. Its shape is
compatible with the physical-orientation profiler, but it is not a spectrum
and must not be used for frequency-basin selection. This is a validation
metric, not a change to production weighting or a moving-noise model.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .comparison import AEInnerProduct, AESeries, AEPSD


class FiniteWindowAEInnerProduct(AEInnerProduct):
    def __init__(self, reference: AESeries, psd: AEPSD, band_hz, integral_spectrum_onesided, *, rtol=1e-9, maxiter=200, progress=None):
        if reference.convention != psd.convention:
            raise ValueError("Signal and PSD conventions differ")
        self.t, self.dt, self.convention = reference.t.copy(), reference.dt, reference.convention
        self.length, self.fs = len(self.t), 1/self.dt
        self.band_hz = tuple(band_hz)
        low, high = self.band_hz
        if not 0 < low < high < self.fs/2 or low < psd.frequency_hz[0] or high > psd.frequency_hz[-1]:
            raise ValueError("One common band inside Nyquist and PSD coverage required")
        all_frequency = np.fft.rfftfreq(self.length, self.dt)
        self.mask = (all_frequency >= low) & (all_frequency <= high)
        if not np.any(self.mask):
            raise ValueError("No band frequencies")
        self.frequency_hz, self.df = all_frequency[self.mask], 1/(self.length*self.dt)
        sa = np.interp(self.frequency_hz, psd.frequency_hz, psd.channels["A"])
        se = np.interp(self.frequency_hz, psd.frequency_hz, psd.channels["E"])
        if np.any(sa<=0) or np.any(se<=0):
            raise ValueError("Positive PSD required throughout band")
        cross = np.zeros(len(sa),complex) if psd.cross_ae is None else np.interp(self.frequency_hz,psd.frequency_hz,psd.cross_ae)
        correlation = np.zeros((len(sa),2,2),complex)
        correlation[:,0,0]=correlation[:,1,1]=1
        correlation[:,0,1]=cross/np.sqrt(sa*se)
        correlation[:,1,0]=correlation[:,0,1].conj()
        lower = np.linalg.cholesky(correlation)
        diagonal = np.zeros_like(lower)
        diagonal[:,0,0]=np.sqrt(2*self.dt/sa)
        diagonal[:,1,1]=np.sqrt(2*self.dt/se)
        self.whitener=np.linalg.solve(lower,diagonal)
        spectral=np.array(integral_spectrum_onesided,dtype=complex,copy=True)
        if spectral.ndim!=3 or spectral.shape[1:]!=(2,2) or not np.all(np.isfinite(spectral)):
            raise ValueError("Finite real-process integral covariance grid required")
        self.integral_length=2*(len(spectral)-1)
        if self.integral_length < 2*self.length:
            raise ValueError("Integral grid must cover at least twice the physical observation")
        scale=float(np.max(abs(spectral)))
        if scale<=0 or np.max(abs(spectral-spectral.swapaxes(1,2).conj()))>1e-12*scale:
            raise ValueError("Nonzero Hermitian covariance required")
        if np.min(np.linalg.eigvalsh(spectral)) < -1e-12*scale or np.max(abs(spectral[[0,-1]].imag))>1e-12*scale:
            raise ValueError("Positive real-process covariance required")
        spectral[1:-1]*=.5
        self.covariance_eigenvalues=self.fs*spectral
        self.rtol,self.maxiter=rtol,maxiter
        self.progress=progress
        self.solve_records=[]
        self.noise_weighting="finite_stationary_real_band_covariance_diagnostic"
        self.operator=LinearOperator((2*self.length,2*self.length),matvec=self._matvec,dtype=float)
        self.reference=self.prepare(reference)
        self.reference_norm=self.norm(self.reference)

    def _whiten(self, value, *, adjoint=False):
        transform=np.fft.rfft(value,axis=-1)
        selected=transform[:,self.mask]
        matrix=self.whitener.swapaxes(1,2).conj() if adjoint else self.whitener
        transformed=np.zeros_like(transform)
        transformed[:,self.mask]=np.einsum("fij,jf->if",matrix,selected)
        return np.fft.irfft(transformed,n=self.length,axis=-1)

    def _project(self,value):
        transformed=np.fft.rfft(value,axis=-1)
        transformed[:,~self.mask]=0
        return np.fft.irfft(transformed,n=self.length,axis=-1)

    def _covariance(self,value):
        transformed=np.fft.rfft(value,n=self.integral_length,axis=-1)
        transformed=np.einsum("fij,jf->if",self.covariance_eigenvalues,transformed)
        return np.fft.irfft(transformed,n=self.integral_length,axis=-1)[:,:self.length]

    def _matvec(self,flat):
        value=np.asarray(flat).reshape(2,self.length)
        # Extend the band-restricted positive operator by the identity on its
        # orthogonal complement, keeping CG on a nonsingular real space.
        return (self._whiten(self._covariance(self._whiten(value,adjoint=True)))+value-self._project(value)).ravel()

    def prepare(self,series):
        if series.convention!=self.convention or not np.array_equal(series.t,self.t):
            raise ValueError("Identical conventions and observation timestamps required")
        value=np.array([series.channels[ch] for ch in ("A","E")])
        white=self._whiten(value).ravel()
        iterations=0
        def callback(_):
            nonlocal iterations
            iterations+=1
            if self.progress:
                self.progress(len(self.solve_records),iterations)
        solved,status=cg(self.operator,white,rtol=self.rtol,atol=0,maxiter=self.maxiter,callback=callback)
        denominator=np.linalg.norm(white)
        residual=float(np.linalg.norm(self.operator@solved-white)/denominator) if denominator>0 else 0.
        self.solve_records.append(dict(iterations=iterations,status=int(status),relative_residual=residual,rtol=self.rtol))
        if status!=0 or not np.isfinite(residual) or residual>2*self.rtol:
            raise RuntimeError(f"Finite covariance solve did not converge: {self.solve_records[-1]}")
        return np.array([white,solved])

    @staticmethod
    def inner(a,b):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("Invalid finite-covariance prepared vectors")
        return float(.5*(np.dot(a[0],b[1])+np.dot(a[1],b[0])))
