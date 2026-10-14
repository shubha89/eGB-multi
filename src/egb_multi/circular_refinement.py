"""Refine a circular fit with derivatives of its normalized physical waveform.

For the immutable real A/E metric, ||s_hat-h_hat||^2 = 2(1-overlap).
A least-squares waveform Jacobian gives a Gauss-Newton proposal that handles
frequency/sky/chirp correlations. SVD truncation regularizes only the proposal;
the exact physical objective, all mass coordinates and bounds remain active.
"""

from dataclasses import asdict, replace

import numpy as np

from .circular_fit import CircularAmplitudeBasis
from .eccentric import EccentricBinaryParams


def doppler_transport_sky_seed(source, beta, longitude, barycenter, velocity, acceleration):
    """Move a sky seed while retaining approximate local observed f and fdot.

    Geometry is in light seconds, velocity in units of c, acceleration in 1/s.
    This uses a local quadratic barycentric phase only to propose a start.
    The caller must evaluate the exact native response and enforce its domain.
    No orientation transformation or physical sky degeneracy is assumed.
    """
    geometry = [np.asarray(x, dtype=float) for x in (barycenter, velocity, acceleration)]
    if any(x.shape != (3,) or not np.all(np.isfinite(x)) for x in geometry):
        raise ValueError('Finite three-vector barycenter, velocity and acceleration required')
    if not np.isfinite(beta) or abs(beta) > np.pi/2 or not np.isfinite(longitude):
        raise ValueError('Valid finite sky direction required')
    r, v, a = geometry
    partner = replace(source, beta=float(beta), lambda_=float(longitude % (2*np.pi)))
    k, kp = source.propagation_direction(), partner.propagation_direction()
    rate, ratep = 1-np.dot(k,v), 1-np.dot(kp,v)
    if min(rate, ratep) <= 0:
        raise ValueError('Positive barycentric phase rates required')
    frequency = source.mean_motion/np.pi-source.fdot*np.dot(k,r)
    observed = frequency*rate
    observed_chirp = source.fdot*rate**2-frequency*np.dot(k,a)
    fp = observed/ratep
    gp = (observed_chirp+fp*np.dot(kp,a))/ratep**2
    return replace(partner, mean_motion=float(np.pi*(fp+gp*np.dot(kp,r))), fdot=float(gp))


def refine_sky_partners(search, barycenter, velocity, acceleration, plane_normal, *,
                        source=None, maxiter=120):
    """Refine distinct sky starts derived from a recovered candidate.

    Retains all candidates, uses neither injection sky nor masses, and records
    out-of-domain proposals instead of clipping them into the search bounds.
    These finite starts supplement an independent global search.
    """
    source = EccentricBinaryParams(**search.result()['best']['source']) if source is None else source
    normal = np.asarray(plane_normal, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0:
        raise ValueError('Finite nonzero plane normal required')
    normal = normal/np.linalg.norm(normal)
    k = source.propagation_direction()
    reflected = k-2*np.dot(k,normal)*normal
    directions = [(source.beta,source.lambda_+np.pi),(-source.beta,source.lambda_),
        (-source.beta,source.lambda_+np.pi),
        (np.arcsin(np.clip(-reflected[2],-1,1)),np.arctan2(-reflected[1],-reflected[0]))]
    records = []
    for beta, longitude in directions:
        partner = doppler_transport_sky_seed(source,beta,longitude,barycenter,velocity,acceleration)
        index = min(range(len(search.basins)),key=lambda i:abs(search.basins[i].center_hz-partner.mean_motion/np.pi))
        point = search.coordinates_of(partner,index)
        bounds = np.array(search._bounds(index,point[3]))
        record = {'seed':asdict(partner),'basin_index':index}
        if np.any(point<bounds[:,0]) or np.any(point>bounds[:,1]):
            records.append(dict(record,status='outside_declared_domain'));continue
        search.evaluate(point,index)
        fit,x,conditioning = search._local_fit(point,index,ftol=1e-12,maxiter=maxiter,finite_difference_step=2e-6)
        entry = search.evaluate(x,index)
        records.append(dict(record,status='evaluated',overlap=entry['profile']['overlap'],
            optimizer_success=bool(fit.success),conditioning=conditioning))
    return records


def refine_with_waveform_jacobian(search, *, source=None, iterations=20,
                                 relative_rank_tolerance=1e-8):
    if iterations < 1 or not 0 < relative_rank_tolerance < 1:
        raise ValueError('Positive iteration count and SVD tolerance in (0,1) required')
    source = EccentricBinaryParams(**search.result()['best']['source']) if source is None else source
    index = min(range(len(search.basins)),key=lambda i:abs(search.basins[i].center_hz-source.mean_motion/np.pi))
    point = search.coordinates_of(source,index)
    bounds = np.array(search._bounds(index,point[3]))
    reference = search.metric.reference/np.sqrt(search.metric.reference_norm)
    cache = {}
    def state(x):
        key = tuple(x)
        if key not in cache:
            entry = search.evaluate(x,index)
            # The stored physical source contains the recovered orbital phase.
            # Regenerate using the original basis anchor, not that fitted phase.
            basis = CircularAmplitudeBasis.generate(search.source_at(x,index),search.response)
            prepared = np.array([search.metric.prepare(item) for item in basis.responses])
            waveform = np.einsum('i,ijk->jk',entry['profile']['coefficients'],prepared)
            cache[key] = (entry,waveform/np.sqrt(search.metric.norm(waveform)))
        return cache[key]
    def real_vector(x):
        return np.r_[x.real.ravel(),x.imag.ravel()]
    history=[]
    initial = state(point)[0]['profile']['overlap']
    steps = np.array([1e-4,1e-3,1e-3,1e-3,1e-2,1e-2])
    for iteration in range(iterations):
        entry,current = state(point)
        residual = real_vector(reference-current)
        derivatives=[]
        for axis,step in enumerate(steps):
            low,high = bounds[axis]
            minus,plus = max(low,point[axis]-step),min(high,point[axis]+step)
            left,right = point.copy(),point.copy()
            left[axis],right[axis] = minus,plus
            derivative = (state(right)[1]-state(left)[1])/(plus-minus)
            derivatives.append(real_vector(derivative))
        jacobian = np.column_stack(derivatives)
        proposal,_,rank,singular = np.linalg.lstsq(jacobian,residual,rcond=relative_rank_tolerance)
        active = np.zeros(point.size,dtype=bool)
        # An outward proposal at an active physical bound is infeasible. Solve
        # the linear proposal on its tangent face, allowing those coordinates
        # to move inward again on subsequent iterations. The fit domain stays
        # unchanged; a flat mass coordinate cannot block all other coordinates.
        for _ in range(point.size):
            outward=((point<=bounds[:,0]+1e-12)&(proposal<0))|((point>=bounds[:,1]-1e-12)&(proposal>0))
            newly_active=outward&~active
            if not np.any(newly_active): break
            active|=newly_active
            proposal=np.zeros_like(point)
            if np.any(~active):
                proposal[~active],_,rank,singular=np.linalg.lstsq(jacobian[:,~active],residual,rcond=relative_rank_tolerance)
        # Reduce a common step length to respect bounds; never independently
        # clip coordinates, which would alter the proposed correlated direction.
        length=1.
        for axis,delta in enumerate(proposal):
            if delta>0: length=min(length,(bounds[axis,1]-point[axis])/delta)
            elif delta<0: length=min(length,(bounds[axis,0]-point[axis])/delta)
        length=max(0.,min(1.,length))
        accepted=False
        before=entry['profile']['overlap']
        for backtrack in range(20):
            trial=point+(length*2.**(-backtrack))*proposal
            if np.any(trial<bounds[:,0]) or np.any(trial>bounds[:,1]):
                continue
            candidate=state(trial)[0]
            if candidate['profile']['overlap']>before+1e-13:
                point=trial;accepted=True;break
        after=state(point)[0]['profile']['overlap']
        history.append({'iteration':iteration,'before_overlap':before,'after_overlap':after,
            'jacobian_rank':int(rank),'singular_values':singular.tolist(),
            'linear_proposal_active_bounds':np.flatnonzero(active).tolist(),
            'proposed_step':proposal.tolist(),'accepted':accepted,'bound_step_length':length})
        if not accepted: break
    return dict(search.result(),waveform_jacobian_refinement={
        'initial_overlap':initial,'endpoint_overlap':state(point)[0]['profile']['overlap'],
        'coordinates':point.tolist(),'iterations':history,'finite_difference_steps_scaled':steps.tolist(),
        'relative_SVD_rank_tolerance':relative_rank_tolerance,
        'limitation':'Local proposal refinement; independent global starts remain necessary'})
