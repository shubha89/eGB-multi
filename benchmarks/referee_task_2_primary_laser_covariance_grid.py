"""Resolved annual primary-laser covariance for ideal N1-12 locking.

Physical chain-rule delay rates and a cancellation-safe grouped polynomial
evaluate a dense frequency grid. Compare grouped evaluation against direct
arbitrary-precision summation and refine native geometry sampling separately.
The result is an adiabatic symbol, not sampled-instrument interpolation noise
or an exact moving finite-observation likelihood. Keep the earlier failed
1e-5 ratio controls; this run measures their scale, not a revised pass gate.
"""

import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"outputs/referee/.precision_deps"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/referee/.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/referee/.cache"))

import numpy as np
from lisainstrument.orbiting.constellation_enums import MosaID
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from egb_multi.laser_symbol import moving_primary_laser_transfer
from referee_task_2_frozen_noise_audit import instrument
from referee_task_2_covariance_grid import chunked_model
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    assert_local();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError("Use this checkout")
    out.mkdir(parents=True,exist_ok=False);hashes={}
    for path in [*list((ROOT/"src/egb_multi").glob("*.py")),Path(__file__),ROOT/"benchmarks/referee_task_2_frozen_noise_audit.py",ROOT/"benchmarks/referee_task_2_covariance_grid.py",ROOT/"benchmarks/referee_task_2_objective_orientation.py"]:
        rel=path.relative_to(ROOT);target=out/"snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    year=31557600.;epochs=(np.arange(args.epochs)+.5)*year/args.epochs
    # Resolve the annual TDI-null region at every physical annual FFT bin,
    # retain a uniform broad grid, and cover below-band infrared noise.
    annual_null=np.concatenate([np.arange(int(np.ceil(low*year)),int(np.floor(min(high,args.maximum_frequency)*year))+1)/year
        for low,high in ((.029,.031),(.059,.061),(.089,.091)) if low<args.maximum_frequency])
    f=np.unique(np.r_[np.geomspace(1/(100*year),2e-5,1024),np.arange(1,100001)*5e-7,
        np.arange(1,int(np.ceil(args.maximum_frequency/5e-7))+1)*5e-7,
        annual_null,.001,.003,.006,.01,.02,.03,.04])
    write_json(out/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
        epochs=args.epochs,frequency_bins=len(f),geometry_dt_s=args.geometry_dt,locking="N1-12",
        rate_mode="chain_product",noise_filter_dt_s=.03125,cutoff_hz=1/year,
        maximum_frequency_hz=args.maximum_frequency,analysis_band_hz=args.band,
        scope="Dense adiabatic primary-laser covariance control; no production certificate"))
    orbit=default_lisaorbits("esa-trailing");definitions=instrument(1/year,.03125)
    if definitions.lock_config!="N1-12":raise ValueError("Unimplemented locking")
    laser_psd=2*definitions.get_noise_defs().noise_def_laser(MosaID("12")).discrete_psd_twosided(f)/definitions.central_freq**2
    delays=link_delays_from_lisaorbits(orbit,epochs)
    print("Annual secondary comparison",len(f),"frequencies",flush=True)
    secondary=chunked_model(f,definitions,delays)
    previous=None;rows=[]
    for dt in args.geometry_dt:
        covariance=np.zeros_like(secondary)
        for index,epoch in enumerate(epochs):
            t=epoch+np.arange(-256,257)*dt;local=link_delays_from_lisaorbits(orbit,t)
            derivatives={key:np.polynomial.Chebyshev.fit(t,value,9).deriv()(t) for key,value in local.items()}
            h=moving_primary_laser_transfer(f,t,local,[256],delay_derivatives=derivatives,rate_mode="chain_product",summation="grouped_numpy")[0]
            covariance+=np.einsum("fa,fb,f->fab",h,h.conj(),laser_psd)/args.epochs
            print("laser geometry",dt,"epoch",index+1,flush=True)
        diagonal=np.sqrt(secondary[:,[0,1],[0,1]].real)
        lower=np.linalg.cholesky(secondary/diagonal[:,:,None]/diagonal[:,None,:])
        normalized=covariance/diagonal[:,:,None]/diagonal[:,None,:]
        transformed=np.linalg.solve(lower,normalized)
        transformed=np.linalg.solve(lower.conj(),transformed.swapaxes(1,2)).swapaxes(1,2)
        eigenvalues=np.linalg.eigvalsh((transformed+transformed.swapaxes(1,2).conj())/2)[:,-1]
        band=(f>=args.band[0])&(f<=args.band[1]);null=np.zeros(len(f),bool)
        for low,high in ((.029,.031),(.059,.061),(.089,.091)):null|=(f>=low)&(f<=high)&band
        row=dict(geometry_dt_s=dt,max_in_band_relative_laser=float(eigenvalues[band].max()),
            max_in_band_frequency_hz=float(f[band][np.argmax(eigenvalues[band])]),
            max_near_null_relative_laser=float(eigenvalues[null].max()),
            max_below_band_relative_laser=float(eigenvalues[f<args.band[0]].max()))
        if previous is not None:
            delta=(covariance-previous)/diagonal[:,:,None]/diagonal[:,None,:]
            delta=np.linalg.solve(lower,delta)
            delta=np.linalg.solve(lower.conj(),delta.swapaxes(1,2)).swapaxes(1,2)
            row["maximum_in_band_relative_covariance_change"]=float(abs(np.linalg.eigvalsh((delta+delta.swapaxes(1,2).conj())/2)[band]).max())
        np.savez_compressed(out/f"annual_laser_covariance_dt{dt:g}.npz",frequency_hz=f,covariance=covariance,
            secondary_covariance=secondary,relative_covariance_eigenvalue=eigenvalues)
        rows.append(row);write_json(out/"covariance_controls.json",rows);print(row,flush=True);previous=covariance
    # Exact direct summation validates the fast algebra at selected frequencies
    # spanning the red cutoff, ordinary harmonics and the strongest null bin.
    t=epochs[0]+np.arange(-256,257)*args.geometry_dt[-1];local=link_delays_from_lisaorbits(orbit,t)
    derivatives={key:np.polynomial.Chebyshev.fit(t,value,9).deriv()(t) for key,value in local.items()}
    selected=np.unique(np.r_[np.geomspace(f[0],args.maximum_frequency,64),f[null][np.argmax(eigenvalues[null])]])
    exact=moving_primary_laser_transfer(selected,t,local,[256],delay_derivatives=derivatives,rate_mode="chain_product")[0]
    fast=moving_primary_laser_transfer(selected,t,local,[256],delay_derivatives=derivatives,rate_mode="chain_product",summation="grouped_numpy")[0]
    error=float(abs(exact-fast).max());scale=float(abs(exact).max())
    write_json(out/"validation_summary.json",dict(status="dense_laser_covariance_diagnostic_completed" if error<1e-8*scale else "failed_grouped_sum_control",
        production_certified=False,direct_vs_grouped_absolute_transfer_difference=error,direct_transfer_scale=scale,
        limitations=["Dense frequency control is not a uniform mathematical bound", "Native float64 orbit/delay inputs remain",
            "Physical adiabatic chain-rule rates, not instrument interpolation/locking arithmetic", "Finite covariance with below-band laser noise remains to be tested"]))
    if error>=1e-8*scale:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=16)
    p.add_argument("--geometry-dt",type=float,nargs="+",default=[600.,300.,150.])
    p.add_argument("--maximum-frequency",type=float,default=.05)
    p.add_argument("--band",type=float,nargs=2,default=[2e-5,.04])
    run(p.parse_args())
