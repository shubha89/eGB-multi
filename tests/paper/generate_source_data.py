"""Reproduce the existing source illustrations with the installed eGB package.

This evaluates fixed Newtonian/1PN source polarizations and initial Peters
rates only. It generates no annual detector response, noise realization or
search. Only plotted samples and aggregated envelopes are saved.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from scipy.ndimage import uniform_filter1d
from egb_multi.eccentric import (EccentricBinaryParams, eccentric_complex_strain,
    eccentric_dynamics, peters_mathews_derivatives)
from egb_multi.constants import JULIAN_YEAR_S

def source(e=.6,m=1.4):
    return EccentricBinaryParams(mean_motion=np.pi*.001,eccentricity=e,
        m1_solar=m,m2_solar=m,inclination=.8,beta=.2,lambda_=-1.,psi=.4,phi0=.2)

def generate(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new output directory')
    output.mkdir(parents=True)
    arrays={};tails={}
    t=np.arange(0.,512*2000.,2.5);k=np.arange(1,13);zoom=t<=16000
    arrays['time_hours']=t[zoom]/3600;arrays['harmonic_f_over_f0']=k/2
    for row,e in enumerate((0.,.3,.6)):
        h=eccentric_complex_strain(source(e,.5),t,physics_mode='newtonian',evolution_mode='fixed').real
        spec=np.fft.rfft(h-h.mean())*2.5;p=abs(spec[k*512])**2;p/=p.max()
        arrays[f'hplus_normalized_{row}']=h[zoom]/max(abs(h[zoom]))
        arrays[f'harmonic_power_{row}']=p
        f=np.fft.rfftfreq(t.size,2.5)
        tails[str(e)]=float(np.sum(abs(spec[f>=.025])**2)/np.sum(abs(spec)**2))
    np.savez_compressed(output/'source_harmonics.npz',**arrays)
    del t,h,spec,arrays
    print('Reproduced fixed Newtonian harmonic illustration.',flush=True)

    s=source();dt=10.;t=np.arange(0.,356*86400.,dt);stride=int(.2*86400/dt)
    count=t.size//stride*stride;days=t[:count].reshape(-1,stride).mean(axis=1)/86400
    curves=[]
    for mode in ('newtonian','1pn'):
        h=eccentric_complex_strain(s,t,physics_mode=mode,evolution_mode='fixed').real
        blocks=h[:count].reshape(-1,stride)
        rms=np.sqrt(np.maximum(uniform_filter1d(h*h,size=4320,mode='constant'),0))
        curves.append((blocks.min(axis=1),blocks.max(axis=1),rms[::432].copy()))
        del h,blocks,rms
    norm=max(abs(curves[1][0]).max(),abs(curves[1][1]).max())
    sample_days=t[::432]/86400;valid=(sample_days>=.5)&(sample_days<=355.5)
    d=eccentric_dynamics(s,np.array([0.]),physics_mode='1pn',evolution_mode='fixed')
    period=float(2*np.pi/(d['azimuthal_frequency'][0]-d['radial_mean_motion'][0])/86400)
    np.savez_compressed(output/'periastron_envelope.npz',days=days,
        newtonian_min=curves[0][0]/norm,newtonian_max=curves[0][1]/norm,
        onepn_min=curves[1][0]/norm,onepn_max=curves[1][1]/norm,
        rms_days=sample_days[valid],rms_ratio=(curves[1][2]/curves[0][2])[valid],
        normalization=norm,full_apsidal_period_days=period)
    checks=[]
    for center in np.linspace(2,354,9)*86400:
        values=[]
        for step in (10.,5.):
            window=np.arange(center-.25*86400,center+.25*86400,step);r=[]
            for mode in ('newtonian','1pn'):
                h=eccentric_complex_strain(s,window,physics_mode=mode,evolution_mode='fixed').real
                r.append(float(np.sqrt(np.mean(h*h))))
            values.append(r[1]/r[0])
        checks.append(abs(values[1]-values[0]))
    del t,curves,h
    print('Reproduced conservative source envelope and its local cadence checks.',flush=True)

    frequencies=np.geomspace(1e-5,1e-2,400);rows=[]
    for mass in (.5,1.4):
        for e in (0.,.1,.3,.5):
            s=source(e,mass)
            nd,ed=peters_mathews_derivatives(np.pi*frequencies,e,s.total_mass_kg,s.symmetric_mass_ratio)
            phase=nd*(4*JULIAN_YEAR_S)**2
            decay=abs(ed)*4*JULIAN_YEAR_S/e if e else np.zeros_like(frequencies)
            rows += [dict(m1_solar=mass,m2_solar=mass,e=e,f0_hz=float(f),
                phase_estimate_rad=float(p),fractional_e_estimate=float(q))
                for f,p,q in zip(frequencies,phase,decay)]
    with (output/'pm_rate_data.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    record=dict(harmonic_power_fraction_above_25mHz=tails,
        full_apsidal_period_days=period,half_apsidal_period_days=period/2,
        maximum_local_envelope_ratio_change_10s_to_5s=max(checks),
        source_physics='fixed Newtonian harmonics and fixed conservative 1PN orbit; initial Peters rates',
        detector_responses_generated=0,noise_realizations_generated=0,searches_launched=0,
        annual_waveform_catalogue_saved=False,
        source_parameters=dict(f0_hz=.001,m1_solar=1.4,m2_solar=1.4,eccentricity=.6,
            inclination=.8,beta=.2,lambda_=-1.,psi=.4,phi0=.2),
        sampling=dict(harmonics_dt_s=2.5,harmonics_radial_periods=512,
            envelope_dt_s=10.,envelope_duration_days=356,envelope_bin_days=.2,
            rms_window_days=.5,edge_exclusion_days=.5,rate_diagnostic_years=4))
    (output/'source_generation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();generate(args.output)
