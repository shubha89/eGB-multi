"""Profile and time unchanged pyTDI mathematics with GW-only/cached paths."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cProfile
import hashlib
import importlib.metadata as metadata
import io
import json
import platform
import pstats
from time import perf_counter
import numpy as np
from pytdi.michelson import compute_factorized_michelson
from egb_jax_eccentric import (lisa_orbit, EccentricBinaryParams,
    precompute_jax_link_geometry, eccentric_links_jax, xyz_from_links,
    prepare_xyz_from_links, pytdi_data_from_links)


def reference(state, links):
    data=pytdi_data_from_links(state, links)
    return {c:compute_factorized_michelson(data,rot=i,order=3,delay_order=3,
        generation=2,unit='frequency') for i,c in enumerate('XYZ')}


def main():
    output=Path('benchmarks/results/tdi_optimization');output.mkdir(parents=True,exist_ok=True)
    result={'platform':platform.platform(),
        'versions':{p:metadata.version(p) for p in ['jax','numpy','pytdi']},
        'generation':2,'measurement_order':3,'delay_order':3,'repeats':7,
        'bridge_sha256':hashlib.sha256(Path('src/egb_jax_eccentric/pytdi_bridge.py').read_bytes()).hexdigest(),
        'scope':'TDI only; real GW-only links; all arrays materialized in NumPy; cached setup separate.',
        'cases':[]}
    for n,dt in [(4096,147.65625),(65536,5.),(631152,50.)]:
        state=lisa_orbit(np.arange(n)*dt)
        src=EccentricBinaryParams(mean_motion=np.pi*.001,eccentricity=.3,
            inclination=.8,beta=.2,lambda_=.4,psi=.3)
        links=eccentric_links_jax(src,precompute_jax_link_geometry(state),batch_size=1)
        ref=reference(state,links)
        t=perf_counter();cached=prepare_xyz_from_links(state,generation=2);setup=perf_counter()-t
        methods={'original':lambda:reference(state,links),
            'gw_only':lambda:xyz_from_links(state,links,generation=2),
            'prepared':lambda:cached(links)}
        timings={k:[] for k in methods};errors={}
        for name,fn in methods.items():
            actual=fn()
            errors[name]=max(float(np.linalg.norm(actual[c]-ref[c])/np.linalg.norm(ref[c])) for c in 'XYZ')
            assert errors[name]<1e-12
        rng=np.random.default_rng(123)
        names=list(methods)
        for _ in range(7):
            for name in rng.permutation(names):
                t=perf_counter();actual=methods[name]();timings[name].append(perf_counter()-t)
        row={'samples':n,'dt_s':dt,'duration_s':n*dt,'prepared_setup_s':setup,
            'warm_s':timings,'median_s':{k:float(np.median(v)) for k,v in timings.items()},
            'maximum_channel_relative_l2':errors}
        result['cases'].append(row)
        (output/'timings.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if n==65536:
            for name,fn in methods.items():
                pr=cProfile.Profile();pr.enable();fn();pr.disable()
                pr.dump_stats(str(output/f'{name}.prof'))
                stream=io.StringIO();pstats.Stats(pr,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(25)
                (output/f'{name}_profile.txt').write_text(stream.getvalue())
        del cached,methods,ref,actual,links,state


if __name__=='__main__':main()
