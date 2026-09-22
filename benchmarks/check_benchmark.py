"""Numerical checks of benchmark blocking and resolution; run outside timings."""
import json
from pathlib import Path
import numpy as np
import jax
from wall_time import (source, gb_params, egb_block, default_lisaorbits, JaxGB, sync,
    state_from_lisaorbits, precompute_jax_link_geometry, eccentric_links_jax)


def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    orbits=default_lisaorbits('equal')
    physics='1pn_periastron'
    src=source(.01,.6)
    halo=128; block=4096; dt=5.
    geometry=precompute_jax_link_geometry(state_from_lisaorbits(orbits,np.arange(1024)*dt))
    links=eccentric_links_jax(src,geometry,batch_size=1,physics_mode=physics)
    max_imaginary=max(float(np.abs(v.imag).max()) for v in links.values())
    if max_imaginary != 0.:
        raise AssertionError('pyTDI cast would discard a nonzero imaginary signal')
    # Identical retained times, with independently calculated halo regions.
    full,_=egb_block(orbits,src,np.arange(-halo,2*block+halo)*dt,physics)
    blocks=[]
    for start in [0,block]:
        values,_=egb_block(orbits,src,(start+np.arange(-halo,block+halo))*dt,physics)
        blocks.append(np.stack(list(values.values()))[:,halo:halo+block])
    full=np.stack(list(full.values()))[:,halo:halo+2*block]
    chunked=np.concatenate(blocks,axis=1)
    blocking=relative(chunked,full)
    if blocking>1e-8:
        raise AssertionError(f'Blocking discrepancy: {blocking}')
    # Check response interpolation/sampling at the hardest source, not equality
    # between the different source/response models in the two packages.
    fine,_=egb_block(orbits,src,np.arange(-2*halo,4*block+2*halo)*dt/2,physics)
    fine=np.stack(list(fine.values()))[:,2*halo:2*halo+4*block:2]
    cadence=relative(full,fine)
    # JAXGB resolution: align physical Fourier bins before comparison.
    convergence=[]
    for f in [.001,.003,.01]:
        responses=[]
        for n in [2048,4096]:
            model=JaxGB(orbits,t_obs=365.25*86400,n=n)
            fn=jax.jit(lambda p:model.get_tdi(p,tdi_generation=2))
            responses.append(np.stack([np.asarray(x) for x in sync(fn(gb_params(f)))]))
        coarse,fine=responses
        central=fine[:,1024:3072]
        outside=np.concatenate([fine[:,:1024],fine[:,3072:]],axis=1)
        convergence.append({'frequency_hz':f,'relative_l2_common_bins':relative(coarse,central),
            'fine_power_outside_coarse_band_fraction':float(np.linalg.norm(outside)**2/np.linalg.norm(fine)**2)})
    result={'maximum_link_imaginary_component':max_imaginary,
            'blocking_relative_l2':blocking,'dt5_vs_dt2p5_relative_l2_xyz':cadence,
            'jaxgb_resolution':convergence,
            'scope':'Blocking and numerical-resolution checks only; no cross-model accuracy equivalence established.'}
    Path('benchmarks/results/checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
