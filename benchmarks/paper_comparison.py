"""Bounded timing diagnostic using the paper's stated wide-grid cadence.

This does not reproduce Fig. 8: its precise timing boundaries/sample count
and compilation policy are not specified in the paper.
"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from wall_time import (source, gb_params, sync, jax, default_lisaorbits,
    state_from_lisaorbits, precompute_jax_link_geometry, eccentric_links_jax,
    xyz_from_links)
from jaxgb.jaxgb import JaxGBaccurate

orbits=default_lisaorbits('equal')
times=np.arange(631152)*50.
t=perf_counter()
state=state_from_lisaorbits(orbits,times)
geometry=precompute_jax_link_geometry(state)
result={'duration_s':31557600,'dt_s':50,'samples':len(times),
    'frequency_hz':.001,'eccentricity':.3,'geometry_setup_s':perf_counter()-t,
    'egb_cached_geometry':[],'jaxgbaccurate':[],
    'limitation':'Diagnostic, not reproduction of Fig.8; its timing-specific sample count and timing boundaries are unknown.'}
src=source(.001,.3)
for i in range(4):
    t=perf_counter()
    links=eccentric_links_jax(src,geometry,batch_size=1,physics_mode='1pn_periastron')
    link_s=perf_counter()-t
    t=perf_counter();xyz=xyz_from_links(state,links,generation=2);tdi_s=perf_counter()-t
    assert all(np.isfinite(x).all() for x in xyz.values())
    result['egb_cached_geometry'].append({'cold':i==0,'links_s':link_s,'tdi_s':tdi_s,'total_s':link_s+tdi_s})
    del links,xyz
del state,geometry
for n in [2048,4096]:
    model=JaxGBaccurate(orbits=orbits,t_obs=31557600,n=n)
    pars=gb_params(.001)
    for compiled in [False,True]:
        fn=lambda p:model.get_tdi(p,tdi_generation=2,tdi_combination='XYZ')
        if compiled:fn=jax.jit(fn)
        sync(fn(pars))
        measured=[]
        for _ in range(10):
            t=perf_counter();sync(fn(pars));measured.append(perf_counter()-t)
        result['jaxgbaccurate'].append({'n':n,'jit':compiled,'warm_s':measured})
Path('benchmarks/results/paper_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
