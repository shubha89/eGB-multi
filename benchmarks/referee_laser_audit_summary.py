"""Summarize the archived cancellation tests and independent response checks."""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/referee/.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/referee/.cache'))
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import numpy as np
import matplotlib.pyplot as plt
from test_lisa_conventions import compare_response
from egb_multi.eccentric import EccentricBinaryParams


def load(stage,name):
    return np.loadtxt(ROOT/f'outputs/referee/laser_audit_20261007_{stage}'/name/'psd.csv',delimiter=',',skiprows=1)


def main():
    out=ROOT/'outputs/referee/laser_audit_20261007_summary'
    out.mkdir(exist_ok=True)
    secondary=load('confirm','native_dt5_order45_secondary')
    ratios={}
    for name in ('native_dt5_order45_laser','native_dt2p5_order45_laser','native_dt5_order63_laser'):
        data=load('confirm',name)
        data=data[:min(len(data),len(secondary))]
        np.testing.assert_allclose(data[:,0],secondary[:len(data),0])
        band=(data[:,0]>=.0001)&(data[:,0]<=.03)
        reference=secondary[:len(data)]
        ratios[name]={channel:dict(zip(('median','p95','maximum'),np.quantile(data[band,k]/reference[band,k],[.5,.95,1.])))
                      for channel,k in (('A',1),('E',2))}
    conventions=[]
    for f,e,psi in ((.001,0.,0.),(.01,.3,.71),(.03,.1,1.13)):
        source=EccentricBinaryParams(mean_motion=np.pi*f,eccentricity=e,m1_solar=.8,m2_solar=.6,
            inclination=1.21,beta=.2,lambda_=-1.,psi=psi,phi0=.617)
        conventions.append(dict(frequency_hz=f,eccentricity=e,**compare_response(source,np.arange(4096)*2.5)))
    all_noise=load('confirm','native_dt5_order45_all')
    band=(all_noise[:,0]>=.0001)&(all_noise[:,0]<=.03)
    initial=load('initial','moving_fast')
    coh=dict(all_noise_median=float(np.median(all_noise[band,3])),secondary_median=float(np.median(secondary[band,3])),
        overlapping_welch_segments=30,coherence_estimator_floor_approximate='order 1/K; overlapping segments are correlated')
    # Use actual segment count from each saved configuration rather than a
    # rounded estimate; the endpoint crop removes a possible final segment.
    metrics=json.loads((ROOT/'outputs/referee/laser_audit_20261007_confirm/native_dt5_order45_all/metrics.json').read_text())
    n=metrics['kwargs']['size']-2*metrics['halo_samples']; m=metrics['nperseg']
    coh['overlapping_welch_segments']=int(1+(n-m)//(m//2))
    summary=dict(laser_to_secondary_psd_ratios=ratios,independent_response_comparison=conventions,
        corrected_noise_coherence=coh,
        interpretation='Residual-only coherence is not a cancellation criterion. PSD ratios are finite-realization estimates; no mission filtering/clock calibration or full fitting factor convergence is certified.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for stage,name,label in (('initial','moving_fast','10 s, instrument 3 / TDI 7'),
        ('initial','moving_order45','10 s, both 45'),('refine','dt5_order45','5 s, both 45'),
        ('refine','dt2p5_order45','2.5 s, both 45')):
        d=load(stage,name); mask=(d[:,0]>=.0001)&(d[:,0]<=.03)
        axes[0,0].loglog(d[mask,0],d[mask,1],label=label,linewidth=.8)
    axes[0,0].set(title='Controlled laser-only runs: A PSD',ylabel=r'$S_A$ [fractional frequency$^2$/Hz]')
    for name,label in (('native_dt5_order45_laser','5 s, order 45'),('native_dt2p5_order45_laser','2.5 s, order 45'),
                      ('native_dt5_order63_laser','5 s, order 63')):
        d=load('confirm',name)[:len(secondary)]; mask=(d[:,0]>=.0001)&(d[:,0]<=.03)
        axes[0,1].loglog(d[mask,0],d[mask,1]/secondary[:len(d)][mask,1],label=label,linewidth=.8)
    axes[0,1].axhline(1.,color='black',linestyle=':',linewidth=1)
    axes[0,1].set(title='Native retarded orbits: laser / secondary A PSD',ylabel='PSD ratio')
    for d,label,color in ((all_noise,'All default noise terms','tab:blue'),(secondary,'Test mass + optical metrology','tab:orange')):
        mask=(d[:,0]>=.0001)&(d[:,0]<=.03)
        axes[1,0].loglog(d[mask,0],d[mask,1],label=label+' A',color=color,linewidth=.8)
        axes[1,0].loglog(d[mask,0],d[mask,2],label=label+' E',color=color,linestyle='--',linewidth=.8)
        axes[1,1].semilogx(d[mask,0],d[mask,3],'.',label=label,color=color,markersize=1.5,alpha=.5)
    axes[1,0].set(title='Corrected 5 s, order 45 noise runs',ylabel=r'PSD [fractional frequency$^2$/Hz]')
    axes[1,1].axhline(1/coh['overlapping_welch_segments'],color='black',linestyle=':',label='Approximate 1/K estimator scale')
    axes[1,1].set(title='Corrected noise-only A/E coherence',ylabel=r'$|S_{AE}|^2/(S_A S_E)$',ylim=(0,1.02))
    for ax in axes.flat:
        ax.set_xlabel('Frequency [Hz]'); ax.set_xlim(.0001,.03); ax.grid(True,which='both',alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out/'laser_cancellation_and_coherence.png',dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
