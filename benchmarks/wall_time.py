"""CPU wall times for native XYZ generation, not equal-accuracy comparison.

Run using .venv-benchmark/bin/python benchmarks/wall_time.py --days 365.25.
eGB streams a full year through padded time blocks; JAXGB returns narrow bands.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

# Resolve this checkout explicitly, including environments ignoring editable .pth.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jaxgb.jaxgb import JaxGB
from egb_jax_eccentric import (
    EccentricBinaryParams, default_lisaorbits, state_from_lisaorbits,
    eccentric_links_jax, precompute_jax_link_geometry, xyz_from_links,
)
from egb_jax_eccentric.constants import G_M3_KG_S2, C_M_PER_S, KILOPARSEC_M, SOLAR_MASS_KG


def source(f, e):
    return EccentricBinaryParams(mean_motion=np.pi*f, eccentricity=e,
        m1_solar=0.6, m2_solar=0.4, distance_m=KILOPARSEC_M,
        beta=0.2, lambda_=0.4, psi=0.3, inclination=0.8, phi0=0.2, fdot=0.)


def gb_params(f):
    gm=G_M3_KG_S2*SOLAR_MASS_KG
    amplitude=2*0.24*gm**(5/3)*(np.pi*f)**(2/3)/(C_M_PER_S**4*KILOPARSEC_M)
    return jnp.array([f, 0., amplitude, 0.4, 0.2, 0.3, 0.8, 0.4])


def sync(value):
    return jax.block_until_ready(value)


def egb_block(orbits, src, times, physics):
    t=perf_counter()
    state=state_from_lisaorbits(orbits, times)
    geometry=precompute_jax_link_geometry(state)
    geometry_s=perf_counter()-t
    t=perf_counter()
    links=eccentric_links_jax(src, geometry, batch_size=1, physics_mode=physics)
    # Public wrapper materializes NumPy arrays, synchronizing device execution.
    links_s=perf_counter()-t
    t=perf_counter()
    xyz=xyz_from_links(state, links, generation=2)
    tdi_s=perf_counter()-t
    return xyz, np.array([geometry_s, links_s, tdi_s])


def egb_run(orbits, src, count, dt, block, halo, physics):
    timings=np.zeros(3)
    sums=np.zeros(3)
    for start in range(0, count, block):
        keep=min(block,count-start)
        # Always keep one shape, including the last block, to avoid recompilation.
        times=(start+np.arange(-halo, block+halo))*dt
        xyz, elapsed=egb_block(orbits,src,times,physics)
        timings+=elapsed
        for i,values in enumerate(xyz.values()):
            values=values[halo:halo+keep]
            if not np.isfinite(values).all():
                raise ValueError("Non-finite eGB output")
            sums[i]+=np.vdot(values,values).real
    return timings, np.sqrt(sums/count)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--days',type=float,default=365.25)
    p.add_argument('--dt',type=float,default=5.)
    p.add_argument('--block',type=int,default=65536)
    p.add_argument('--halo',type=int,default=128)
    p.add_argument('--slow-points',type=int,default=2048)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--physics',default='1pn_periastron')
    p.add_argument('--eccentricities',type=float,nargs='+',default=[0,.1,.2,.3,.4,.5,.6])
    p.add_argument('--frequencies',type=float,nargs='+',default=[.001,.003,.01])
    p.add_argument('--output',default='benchmarks/results/wall_time.json')
    a=p.parse_args()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    count=int(round(a.days*86400/a.dt)); duration=count*a.dt
    result={'config':vars(a),'samples':count,'duration_s':duration,
      'platform':platform.platform(),'processor':platform.processor(),'cpu_count':os.cpu_count(),
      'devices':[str(d) for d in jax.devices()],
      'versions':{x:metadata.version(x) for x in ['jax','jaxlib','jaxgb','numpy','scipy','pytdi','lisaorbits']},
      'egb_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'jaxgb_commit':'ae31825','rows':[]}
    def save():
        out.write_text(json.dumps(result,indent=2)+'\n')
    orbits=default_lisaorbits('equal')
    t=perf_counter(); gb=JaxGB(orbits,t_obs=duration,n=a.slow_points)
    sync((gb.position,gb.r,gb.fstar)); result['jaxgb_setup_s']=perf_counter()-t
    fn=jax.jit(lambda pars:gb.get_tdi(pars,tdi_generation=2,tdi_combination='XYZ'))
    pars=gb_params(a.frequencies[0]); sync(pars)
    t=perf_counter(); compiled=fn.lower(pars).compile(); result['jaxgb_compile_s']=perf_counter()-t
    t=perf_counter(); sync(compiled(pars)); result['jaxgb_first_execution_s']=perf_counter()-t
    # Warm the eGB/pyTDI route at the actual block shape. It includes JIT compile.
    times=np.arange(-a.halo,a.block+a.halo)*a.dt
    _,warm=egb_block(orbits,source(a.frequencies[0],.2),times,a.physics)
    result['egb_first_block_s']=dict(zip(['geometry','links_including_compile','tdi'],warm.tolist()))
    print('WARMUP',result['egb_first_block_s'],flush=True); save()
    for f in a.frequencies:
        pars=gb_params(f);sync(pars)
        native=[]
        for _ in range(20):
            t=perf_counter(); vals=sync(compiled(pars)); native.append(perf_counter()-t)
        if not all(np.isfinite(np.asarray(v)).all() for v in vals):
            raise ValueError('Non-finite JAXGB output')
        result['rows'].append({'model':'JAXGB','frequency_hz':f,'eccentricity':0.,'warm_s':native})
        save()
    # Randomize cases within each repeat to reduce systematic order bias.
    rng=np.random.default_rng(20260921)
    cases=[(f,e) for f in a.frequencies for e in a.eccentricities]
    for rep in range(a.repeats):
        for idx in rng.permutation(len(cases)):
            f,e=cases[idx]
            timing,rms=egb_run(orbits,source(f,e),count,a.dt,a.block,a.halo,a.physics)
            row={'model':'eGB-multi','frequency_hz':f,'eccentricity':e,'repeat':rep,
                 'geometry_s':timing[0],'links_s':timing[1],'tdi_s':timing[2],
                 'total_s':float(timing.sum()),'xyz_rms':rms.tolist()}
            result['rows'].append(row);save();print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
