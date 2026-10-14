"""Carrier-aware interpolation of the exact circular source and TDI response.

An optional numerical approximation for annual fitting-factor validation.
Only slowly varying complex envelopes and the phase-independent response are
interpolated. Native retarded geometry, pyTDI polynomials and the implemented
1PN circular dynamics are retained. Validate against native sampled TDI before
using this evaluator; it does not certify a search or its approximation error.
"""

from dataclasses import replace

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import C_M_PER_S,G_M3_KG_S2
from .comparison import AESeries
from .circular_fit import CircularAmplitudeBasis
from .external import cartesian_frame_rotation,link_delays_from_lisaorbits
from .tdi import LINK_LABELS


class InterpolatedCircularTDI:
    """Approximate one fixed native grid via a carrier-separated envelope grid.

    The time grid and observation interval never change between candidates.
    Both masses remain physical source inputs; no parameter reduction is used.
    A single intrinsic basis is cached, independent of orientation and phase.
    """

    def __init__(self,orbit,times,convention,*,envelope_intervals=8192):
        if convention.unit!='fractional_frequency':
            raise ValueError('Circular GW response requires fractional_frequency units')
        t=np.asarray(times,dtype=float)
        probe=AESeries(t,{'A':np.zeros(len(t)),'E':np.zeros(len(t))},convention)
        if convention.generation!=2 or envelope_intervals<64:
            raise ValueError('TDI2 and at least 64 envelope intervals required')
        self.orbit,self.t,self.convention=orbit,probe.t,convention
        duration=probe.dt*len(t);step=duration/envelope_intervals
        self.coarse=t[0]+np.arange(-32,envelope_intervals+33)*step
        self.rotation=cartesian_frame_rotation('icrs','ecliptic')
        self.cache_key=self.cache_basis=None
        from pytdi.michelson import X2,Y2,Z2
        delays=link_delays_from_lisaorbits(orbit,self.coarse)
        raw={f'd_{key}':value for key,value in delays.items()}
        derivatives={key:np.gradient(value,step,edge_order=2) for key,value in raw.items()}
        self.terms=[];geometries={}
        for ch,combination in enumerate((X2,Y2,Z2)):
            shifts,rates=combination.build_shifts(raw,1/step,derivatives,order=7)
            for measurement,polynomial in combination.components.items():
                if not measurement.startswith('sci_'):continue
                label=measurement[4:];receiver,sender=int(label[0]),int(label[1])
                for factor,chain in polynomial:
                    key=(label,tuple(chain))
                    if key not in geometries:
                        query=self.coarse+np.broadcast_to(shifts[tuple(chain)],self.coarse.shape)
                        length=np.asarray(orbit.compute_ltt(query,[int(label)]))[:,0]
                        pr=np.asarray(orbit.compute_position(query,[receiver]))[:,0,:]@self.rotation.T/C_M_PER_S
                        ps=np.asarray(orbit.compute_position(query-length,[sender]))[:,0,:]@self.rotation.T/C_M_PER_S
                        arm=pr-ps;n=arm/np.linalg.norm(arm,axis=1)[:,None]
                        rate=1+np.broadcast_to(rates[tuple(chain)],self.coarse.shape)
                        geometries[key]=(query,length,n,ps,pr,rate)
                    self.terms.append((ch,factor,geometries[key]))

    def _basis(self,source):
        if source.eccentricity!=0:
            raise ValueError('Envelope evaluator supports only the circular template family')
        key=(source.mean_motion,source.fdot,source.beta,source.lambda_,source.m1_solar,source.m2_solar,source.t0,source.distance_m)
        if key==self.cache_key:return self.cache_basis
        source=replace(source,eccentricity=0,phi0=0.,psi=0.,inclination=np.pi/2)
        envelope,dc=self.coarse_envelope(source)
        z=CubicSpline(self.coarse,envelope,axis=0)(self.t)
        z*=np.exp(2j*source.mean_motion*(self.t-source.t0))[:,None,None]
        dc=CubicSpline(self.coarse,dc,axis=0)(self.t)
        values=[(z[:,:,0].real),(z[:,:,0].imag),(z[:,:,1].real),(z[:,:,1].imag),dc[:,:,0],dc[:,:,1]]
        basis=CircularAmplitudeBasis(source,tuple(AESeries(self.t,{'A':item[:,0],'E':item[:,1]},self.convention) for item in values))
        self.cache_key,self.cache_basis=key,basis
        return basis

    def coarse_envelope(self,source):
        """Complex P/Q envelopes and radial terms on the native geometry grid.

        Orbital phase is zero in this intrinsic basis. This exposes the same
        pre-interpolation calculation for approximate seed selection; its use
        there does not replace full-grid physical objective evaluation.
        """
        if source.eccentricity!=0:
            raise ValueError('Circular source required')
        source=replace(source,phi0=0.,psi=0.,inclination=np.pi/2)
        direction=source.propagation_direction();p,q=source.polarization_basis()
        scale=G_M3_KG_S2*source.total_mass_kg*source.symmetric_mass_ratio/(C_M_PER_S**4*source.distance_m)
        xyz=np.zeros((self.coarse.size,3,2),dtype=complex)
        radial=np.zeros((self.coarse.size,3,2),dtype=float)
        memo={}
        for ch,factor,geometry in self.terms:
            identity=id(geometry)
            if identity not in memo:
                query,length,n,ps,pr,rate=geometry
                sender=query-length-ps@direction;receiver=query-pr@direction
                # Match the prescribed-chirp JAX circular family exactly.
                # NumPy eccentric_dynamics fixed mode does not implement fdot.
                query_source=np.r_[sender,receiver]-source.t0
                omega=source.mean_motion+np.pi*source.fdot*query_source
                if np.any(omega<=0):raise ValueError('Nonpositive instantaneous carrier frequency')
                gm=G_M3_KG_S2*source.total_mass_kg
                x=(gm*omega/C_M_PER_S**3)**(2/3)
                r=gm*(1+x*(-1+source.symmetric_mass_ratio/3))/(C_M_PER_S**2*x)
                rdot=np.zeros_like(omega)
                phase=source.mean_motion*query_source+.5*np.pi*source.fdot*query_source**2
                gravity=G_M3_KG_S2*source.total_mass_kg/r
                tangent=r*r*omega*omega;velocity=rdot*rdot
                a=gravity+tangent-velocity;b=2*r*rdot*omega
                z=-scale*(a-1j*b)*np.exp(2j*phase)
                dc=-scale*(gravity-tangent-velocity)
                size=len(query);denominator=1-n@direction
                projections=np.column_stack((np.einsum('ti,ij,tj->t',n,p,n),np.einsum('ti,ij,tj->t',n,q,n)))
                oscillatory=.5*(z[:size]-z[size:])[:,None]*projections/denominator[:,None]
                constant=.5*(dc[:size]-dc[size:])[:,None]*projections/denominator[:,None]
                # Every term is demodulated at the output detector time,
                # retaining the full phase of its exact retarded source time.
                carrier=np.exp(-2j*source.mean_motion*(self.coarse-source.t0))
                memo[identity]=(oscillatory*carrier[:,None]*rate[:,None],constant*rate[:,None])
            z,dc=memo[identity];xyz[:,ch,:]+=factor*z;radial[:,ch,:]+=factor*dc
        rotate=np.array([[-1,0,1],[1,-2,1]],dtype=float)/np.sqrt(np.array([[2.],[6.]]))
        envelope=np.einsum('ax,txp->tap',rotate,xyz)
        dc=np.einsum('ax,txp->tap',rotate,radial)
        return envelope,dc

    def __call__(self,source):
        return self._basis(source).reconstruct(inclination=source.inclination,psi=source.psi,phi0=source.phi0)

    def circular_basis(self,source):
        """Return the six responses with the caller's orbital-phase anchor.

        This avoids six full-grid physical reconstructions during each trial.
        It changes neither the physical coefficient constraints nor the metric.
        """
        original=self._basis(source)
        cosine,sine=np.cos(2*source.phi0),np.sin(2*source.phi0)
        if source.phi0==0:
            return CircularAmplitudeBasis(source,original.responses)
        transformed=[]
        for first,second in ((0,1),(2,3)):
            a,b=original.responses[first],original.responses[second]
            transformed.extend((
                AESeries(self.t,{ch:cosine*a.channels[ch]-sine*b.channels[ch] for ch in ('A','E')},self.convention),
                AESeries(self.t,{ch:sine*a.channels[ch]+cosine*b.channels[ch] for ch in ('A','E')},self.convention)))
        return CircularAmplitudeBasis(source,tuple(transformed)+original.responses[4:])
