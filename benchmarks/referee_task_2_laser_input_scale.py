"""Record native laser-input scales without changing its noise spectrum."""

from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil

from referee_laser_on_off import ROOT,generate
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    out=args.output.resolve();parent=args.parent.resolve()
    if not out.is_relative_to(ROOT) or not parent.is_relative_to(ROOT):raise ValueError('Use this checkout')
    out.mkdir(parents=True,exist_ok=False);assert_local();hashes={}
    for path in [*list((ROOT/'src/egb_multi').glob('*.py')),Path(__file__),
        ROOT/'benchmarks/referee_laser_on_off.py',ROOT/'benchmarks/referee_task_2_objective_orientation.py',
        parent/'native_annual_orbits.h5']:
        rel=path.relative_to(ROOT);dest=out/'snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes,
        'scope':'Laser-input scale diagnostic; no precision repair or production certification'})
    for days in (365.25,30):
        print('Input-scale control, cutoff days',days,flush=True)
        result=generate(parent/'native_annual_orbits.h5',model='secondary',laser_on=True,
            seed=20261008,samples=262144,dt=5.,noise_cutoff=1/(days*86400))
        write_json(out/f'cutoff_{days:g}_days_metadata.json',result['metadata'])
        print(result['metadata']['laser_noise_statistics'],flush=True)
    write_json(out/'diagnostic_summary.json',{'status':'diagnostic_completed','production_certified':False,
        'cutoff_days':[365.25,30],'scope':'Native input-scale observations only'})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent',type=Path,default=ROOT/'outputs/referee/task_2_8_20261008_annual_laser_residual')
    run(p.parse_args())
