"""Render measured timings, preserving individual repetitions in JSON/CSV."""
import csv
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/egb-benchmark-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('benchmarks/results')
data=json.loads((root/'wall_time.json').read_text())
rows=data['rows']
summary=[]
fig,axes=plt.subplots(1,3,figsize=(12,4.4),sharey=True,layout='constrained')
for ax,f in zip(axes,data['config']['frequencies']):
    gb=next(r for r in rows if r['model']=='JAXGB' and r['frequency_hz']==f)
    baseline=float(np.median(gb['warm_s']))
    es=[]; med=[];lo=[];hi=[];link=[]
    for e in data['config']['eccentricities']:
        selected=[r for r in rows if r['model']=='eGB-multi' and r['frequency_hz']==f and r['eccentricity']==e]
        vals=np.array([r['total_s'] for r in selected])
        m=float(np.median(vals));es.append(e);med.append(m);lo.append(m-vals.min());hi.append(vals.max()-m)
        link.append(float(np.median([r['links_s'] for r in selected])))
        summary.append({'frequency_mhz':f*1000,'eccentricity':e,'egb_median_s':m,
            'egb_min_s':float(vals.min()),'egb_max_s':float(vals.max()),
            'egb_geometry_median_s':float(np.median([r['geometry_s'] for r in selected])),
            'egb_links_median_s':link[-1],
            'egb_tdi_median_s':float(np.median([r['tdi_s'] for r in selected])),
            'jaxgb_median_s':baseline,'native_workload_ratio':m/baseline,'repeats':len(vals)})
    ax.errorbar(es,med,yerr=[lo,hi],fmt='o-',color='#245ca6',label='eGB: geometry + links + XYZ')
    ax.plot(es,link,'s--',color='#e08b25',label='eGB: links only')
    ax.axhline(baseline,color='#3b8c60',label='JAXGB: circular XYZ baseline')
    ax.set_title(f'{f*1000:g} mHz');ax.set_xlabel('Eccentricity (eGB only)');ax.set_yscale('log')
    ax.grid(alpha=.2,which='both')
axes[0].set_ylabel('Warm wall time for one year [s]')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=1,frameon=False)
fig.suptitle('Native output workloads: dense time series versus narrow frequency bands\nCPU float64; eGB points: median and range of 3 repeats',fontsize=11)
fig.savefig(root/'wall_time.png',dpi=180)
fig.savefig(root/'wall_time.pdf')
with (root/'summary.csv').open('w') as fp:
    writer=csv.DictWriter(fp,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
print(json.dumps(summary,indent=2))
text=['# Measured CPU wall times','',
    'One source, one year (365.25 days), CPU float64. eGB: 1PN with periastron advance, fixed eccentricity, 5 s cadence. JAXGB: circular baseline, 2,048 Fourier bins. These are native output workloads, not equal-accuracy implementations.','',
    'eGB values below are the median summed wall time of geometry + links + pyTDI over three full-year repeats. JAXGB values are medians of 20 synchronized, JIT-compiled XYZ calls with geometry cached.','',
    '| Eccentricity | eGB 1 mHz [s] | eGB 3 mHz [s] | eGB 10 mHz [s] |',
    '|---|---:|---:|---:|']
for e in data['config']['eccentricities']:
    vals=[next(s for s in summary if s['frequency_mhz']==f*1000 and s['eccentricity']==e)['egb_median_s'] for f in data['config']['frequencies']]
    text.append(f'| {e:g} | '+ ' | '.join(f'{v:.3f}' for v in vals)+' |')
text+=['','| Frequency | JAXGB warm XYZ [ms] |','|---|---:|']
for f in data['config']['frequencies']:
    s=next(s for s in summary if s['frequency_mhz']==f*1000)
    text.append(f"| {f*1000:g} mHz | {1000*s['jaxgb_median_s']:.3f} |")
text+=['','JAXGB setup: %.4f s; compilation: %.4f s; first execution: %.4f s.'%(data['jaxgb_setup_s'],data['jaxgb_compile_s'],data['jaxgb_first_execution_s']),
    'eGB first block (links including compilation/first execution): %.4f s.'%data['egb_first_block_s']['links_including_compile'],
    '', 'See [methodology](../README.md), [all timings and metadata](wall_time.json), and [stage breakdown with repeat ranges](summary.csv).','',
    '![Timing figure](wall_time.png)','']
if (root/'checks.json').exists():
    checks=json.loads((root/'checks.json').read_text())
    text+=['**Resolution limitation:** at 10 mHz and e=0.6, 5 s versus 2.5 s cadence changes XYZ by %.2f%% in relative L2 norm on the check segment. The 5 s timing is not a converged-accuracy result for this case. Both packages retain their own model and numerical approximations.'%(100*checks['dt5_vs_dt2p5_relative_l2_xyz']), '',
        'Padded blocks agreed exactly with a contiguous calculation on retained test samples. All 11 existing repository tests passed.', '',
        'Numerical checks (not cross-model accuracy validation):','', '```json', (root/'checks.json').read_text().strip(), '```','']
(root/'REPORT.md').write_text('\n'.join(text))
if (root/'PAPER_COMPARISON.md').exists():
    report=(root/'REPORT.md').read_text().replace('# Measured CPU wall times\n',
        '# Measured CPU wall times\n\n**Paper comparison:** these timings do not reproduce arXiv:2608.24546 Figure 8. See [the comparison note and additional timing diagnostic](PAPER_COMPARISON.md).\n',1)
    (root/'REPORT.md').write_text(report)
