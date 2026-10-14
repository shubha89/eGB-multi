"""Isolate clock and ranging contributions in paired laser-on/off tests."""
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
import hashlib
import inspect
import json
import shutil
import sys
from referee_laser_on_off import ROOT,dump,pair,assert_local,Instrument
from referee_task_2_objective_orientation import make_common_orbit_file


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Results must remain in this checkout')
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    files=list((ROOT/'src/egb_multi').glob('*.py'))+[
        Path(__file__),ROOT/'benchmarks/referee_laser_on_off.py',ROOT/'benchmarks/referee_task_2_objective_orientation.py']
    for path in files:
        rel=path.relative_to(ROOT);target=out/'snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),python=sys.executable,
        source_hashes=hashes,source_hash_algorithm='SHA256 of file bytes',seed=20261007,samples=262144,dt=5.,nperseg=65536,
        versions={name:version(name) for name in ('numpy','scipy','jax','pytdi','lisainstrument','lisaorbits')},
        instrument_interpolation=['lagrange',45],TDI_generation=2,TDI_order=45,delay_derivative_order=5,
        physics_upsampling=1,antialias_filter=None,low_frequency_cutoff_hz=None,
        purpose='Controlled clock/ranging isolation; every other default retained within each matched pair'))
    dump(out/'instrument_defaults.json',{name:parameter.default for name,parameter in inspect.signature(Instrument).parameters.items()
        if parameter.default is not inspect.Parameter.empty})
    orbit=out/'native_orbits.h5'
    make_common_orbit_file(orbit,1310720.,model='retarded')
    results=[]
    for disabled,label in ((('clock',),'clocks_disabled'),(('ranging',),'ranging_disabled'),
                           (('clock','ranging'),'clocks_and_ranging_disabled')):
        result,_=pair(out,orbit,model='all_defaults',label=label,disabled_noises=disabled,
            seed=20261007,samples=262144,dt=5.,nperseg=65536)
        results.append(result);dump(out/'noise_comparison.json',results)
    assert_local()


if __name__=='__main__':main()
