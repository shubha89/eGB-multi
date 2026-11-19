"""Model-independent six-link projection with exact receiver-query reuse.

Positions are Cartesian light-seconds, light times and timestamps are seconds,
and the caller supplies the propagation direction and transverse tensors in
the same frame. Geometry may use retarded or instantaneous emitter positions;
that physical choice belongs to the caller and is not inferred here.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..delay import real_array,readonly
from ..michelson import LINK_LABELS

RECEIVERS=np.array([int(label[0])-1 for label in LINK_LABELS])
FIRST=np.array([np.flatnonzero(RECEIVERS==i)[0] for i in range(3)])


@njit(cache=True,fastmath=False)
def _antenna(n,plus,cross):
    links,size=n.shape[:2];a=np.empty((links,size));b=np.empty((links,size))
    for link in range(links):
        for sample in range(size):
            p=0.;q=0.
            for i in range(3):
                for j in range(3):
                    p+=(n[link,sample,i]*plus[i,j])*n[link,sample,j]
                    q+=(n[link,sample,i]*cross[i,j])*n[link,sample,j]
            a[link,sample]=p;b[link,sample]=q
    return a,b


@dataclass(frozen=True,eq=False)
class LinkProjection:
    """Prepared GW fractional-frequency response for one sky/basis and grid.

    Polarizations obey h_ab=h_plus*e_plus_ab+h_cross*e_cross_ab. The optional
    polarization angle rotates those tensors by twice that angle. The callback
    accepts the flattened physical query times and returns (...,2,Q); each
    leading batch member retains its own six output links (...,6,N).
    """
    sender_times: np.ndarray
    receiver_times: np.ndarray
    plus_projection: np.ndarray
    cross_projection: np.ndarray
    denominator: np.ndarray
    query_times: np.ndarray

    @classmethod
    def prepare(cls,propagation_direction,plus_tensor,cross_tensor,geometry,*,polarization_angle=0.):
        direction=real_array(propagation_direction,name='propagation direction')
        if direction.shape!=(3,) or not np.isclose(direction@direction,1.,rtol=0,atol=1e-12):
            raise ValueError('propagation direction must be a Cartesian unit vector')
        plus=real_array(plus_tensor,name='plus tensor');cross=real_array(cross_tensor,name='cross tensor')
        if plus.shape!=(3,3) or cross.shape!=(3,3):raise ValueError('polarization tensors must have shape (3,3)')
        if not np.isfinite(polarization_angle):raise ValueError('polarization angle must be finite')
        times=real_array(geometry['times'],name='reception times')
        if times.ndim!=1 or times.size==0:raise ValueError('nonempty one-dimensional reception times required')
        n=real_array(geometry['n'],name='arm directions')
        lengths=real_array(geometry['length'],name='light times')
        senders=real_array(geometry['p_sender'],name='emitter positions')
        receivers=real_array(geometry['p_receiver'],name='receiver positions')
        if n.shape!=(6,times.size,3) or senders.shape!=n.shape or receivers.shape!=n.shape:
            raise ValueError('geometry vectors must have shape (6,N,3) in LINK_LABELS order')
        if lengths.shape!=(6,times.size) or np.any(lengths<0):raise ValueError('nonnegative light times with shape (6,N) required')
        if not np.allclose(np.sum(n*n,axis=-1),1.,rtol=0,atol=1e-12):raise ValueError('arm directions must be unit vectors')
        c,s=np.cos(2*polarization_angle),np.sin(2*polarization_angle)
        plus,cross=c*plus-s*cross,s*plus+c*cross
        a,b=_antenna(n,plus,cross)
        # Keep this NumPy expression order: rounding an absolute late-epoch
        # wavefront time by one ulp can matter after low-frequency cancellation.
        sender=times[None,:]-lengths-np.einsum('lti,i->lt',senders,direction)
        receiver=times[None,:]-np.einsum('lti,i->lt',receivers,direction)
        denominator=1-np.einsum('lti,i->lt',n,direction)
        if np.any(denominator==0):raise ValueError('exact arm/propagation alignment requires a separately evaluated limiting response')
        for index,receiver_id in enumerate(RECEIVERS):
            if not np.array_equal(receiver[index],receiver[FIRST[receiver_id]]):
                raise ValueError('receiver reuse requires identical physical timestamps')
        queries=np.concatenate((sender.ravel(),receiver[FIRST].ravel()))
        return cls(*(readonly(v) for v in (sender,receiver,a,b,denominator,queries)))

    @property
    def nbytes(self):
        return sum(v.nbytes for v in (self.sender_times,self.receiver_times,self.plus_projection,
                                     self.cross_projection,self.denominator,self.query_times))

    def apply(self,polarizations):
        values=real_array(polarizations(self.query_times),name='queried polarizations')
        if values.ndim<2 or values.shape[-2:]!=(2,self.query_times.size):
            raise ValueError('polarization callback must return (...,2,Q) for its physical queries')
        hp,hc=values[...,0,:],values[...,1,:]
        split=self.sender_times.size;size=self.sender_times.shape[-1]
        shape=values.shape[:-2]
        sender=(self.plus_projection*hp[...,:split].reshape(shape+(6,size))
                +self.cross_projection*hc[...,:split].reshape(shape+(6,size)))
        receiver=(self.plus_projection*hp[...,split:].reshape(shape+(3,size))[...,RECEIVERS,:]
                  +self.cross_projection*hc[...,split:].reshape(shape+(3,size))[...,RECEIVERS,:])
        links=.5*(sender-receiver)/self.denominator
        if np.any(~np.isfinite(links)):raise ValueError('nonfinite GW link response')
        return links
