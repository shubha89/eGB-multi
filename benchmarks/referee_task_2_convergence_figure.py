"""Plot actual annual cadence controls, retaining signed overlap changes."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/referee'
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(args):
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Use this checkout')
    sources=[('near_contour','20261008_annual_refinement_near_contour_numerical',r'$e=0.1$, 1 mHz'),
        ('high_e','20261008_annual_refinement_high_e',r'$e=0.6$, 1 mHz'),
        ('high_pm','20261009_annual_refinement_high_pm_branch_repaired',r'$e=0.5$, 3 mHz')]
    rows=[];hashes={}
    for case,suffix,label in sources:
        folder=BASE/('task_2_8_'+suffix)
        summary=json.loads((folder/'validation_summary.json').read_text())
        if summary['status']!='selected_annual_reoptimization_completed':raise ValueError('Completed controls required: '+case)
        path=folder/'annual_refinement.json';records=json.loads(path.read_text())
        values={r['level']['name']:r['native_winner_metrics']['overlap'] for r in records}
        overlap=[values[name] for name in ('baseline_reopt','cadence_5','cadence_2_5')]
        rows.append(dict(case=case,label=label,dt_s=[10,5,2.5],native_overlap=overlap,
            delta_from_2_5_s=[value-overlap[-1] for value in overlap]))
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'figure_driver.py')
    fig,axis=plt.subplots(figsize=(6.2,4.1),layout='constrained')
    for row,color in zip(rows,('#0072B2','#D55E00','#009E73')):
        axis.plot(row['dt_s'],row['delta_from_2_5_s'],marker='o',label=row['label'],color=color,lw=1.6)
    axis.axhline(0,color='.3',lw=.7)
    axis.axhline(.0002,color='.55',ls='--',lw=.8,label=r'$\pm2\times10^{-4}$ waveform allocation')
    axis.axhline(-.0002,color='.55',ls='--',lw=.8)
    axis.set(xlabel='Waveform cadence (s)',ylabel='Native overlap minus 2.5 s result',
        title='One-year selected circular-family reoptimization',xlim=(10.6,1.9))
    axis.set_yscale('symlog',linthresh=1e-8);axis.set_ylim(-3e-4,3e-4)
    axis.set_xticks([10,5,2.5],['10','5','2.5']);axis.legend(frameon=False,loc='upper left',ncols=2,fontsize=9)
    axis.grid(alpha=.2,which='both')
    fig.savefig(out/'annual_cadence_convergence.pdf');fig.savefig(out/'annual_cadence_convergence.png',dpi=220)
    plt.close(fig)
    (out/'plot_data.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=hashes,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Signed selected optimized cadence changes; interpolation/envelope settings are recorded in source runs',
        dashed_lines='Waveform error allocation 0.0002; empirical shifts do not establish a uniform continuum/global FF bound'),indent=2)+'\n')
    print(out.relative_to(ROOT))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
