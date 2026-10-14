"""Moving/locking primary-laser symbol and geometry/precision refinements.

Tests the ideal N1-12 physical residual independently of combined-stream
roundoff. This does not include sampled laser interpolation errors or complete
finite-observation covariance; these limitations remain explicit.
"""

import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"outputs/referee/.precision_deps"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/referee/.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/referee/.cache"))

import numpy as np
import mpmath
from lisainstrument.orbiting.constellation_enums import MosaID
from egb_multi.external import default_lisaorbits,link_delays_from_lisaorbits
from egb_multi.laser_symbol import moving_primary_laser_transfer
from egb_multi.noise_covariance import frozen_native_secondary_covariance
from referee_task_2_frozen_noise_audit import instrument
from referee_task_2_objective_orientation import assert_local,write_json


def run(args):
    assert_local();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError("Use this checkout")
    out.mkdir(parents=True,exist_ok=False);hashes={}
    for path in [*list((ROOT/"src/egb_multi").glob("*.py")),Path(__file__),ROOT/"benchmarks/referee_task_2_frozen_noise_audit.py",ROOT/"benchmarks/referee_task_2_objective_orientation.py"]:
        rel=path.relative_to(ROOT);target=out/"snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
        locking="N1-12",cutoff_hz=1/31557600.,noise_physics_dt_s=.03125,mpmath_version=mpmath.__version__,
        tested_geometry_dt_s=args.geometry_dt,relative_noise_allowance=1e-5,
        derivative_control=args.derivatives,rate_mode=args.rate_mode,
        scope="Selected-epoch/frequency adiabatic physical laser-symbol diagnostic"))
    f=np.unique(np.r_[np.geomspace(2e-5,.04,128),np.linspace(.029,.031,101),.001,.003,.006,.01,.02,.03,.04])
    orbit=default_lisaorbits("esa-trailing");definitions=instrument(1/31557600.,.03125)
    if definitions.lock_config!="N1-12":raise ValueError("Native locking differs from derived diagnostic")
    laser_psd=2*definitions.get_noise_defs().noise_def_laser(MosaID("12")).discrete_psd_twosided(f)/definitions.central_freq**2
    rows=[];previous=None
    epochs=(np.arange(args.epochs)+.5)*31557600./args.epochs
    for dt in args.geometry_dt:
        ratios=[];responses=[]
        for epoch in epochs:
            t=epoch+np.arange(-256,257)*dt
            delays=link_delays_from_lisaorbits(orbit,t)
            derivatives=None
            if args.derivatives=="chebyshev":
                derivatives={key:np.polynomial.Chebyshev.fit(t,value,9).deriv()(t) for key,value in delays.items()}
            transfer=moving_primary_laser_transfer(f,t,delays,[256],precision_digits=60,
                delay_derivatives=derivatives,rate_mode=args.rate_mode)[0]
            secondary=frozen_native_secondary_covariance(f,{key:float(value[256]) for key,value in delays.items()},definitions)[0]
            # Rank-one laser covariance: its largest generalized eigenvalue
            # equals S_p * H^H C_secondary^{-1} H.
            diagonal=np.sqrt(secondary[:,[0,1],[0,1]].real)
            correlation=secondary/diagonal[:,:,None]/diagonal[:,None,:]
            whitened=np.linalg.solve(np.linalg.cholesky(correlation), (transfer/diagonal)[:,:,None])[:,:,0]
            ratios.append(laser_psd*np.sum(abs(whitened)**2,axis=1))
            responses.append(transfer)
        values=np.array(ratios);response=np.array(responses)
        rows.append(dict(geometry_dt_s=dt,maximum_generalized_laser_over_secondary=float(values.max()),
            maximum_frequency_hz=float(f[np.unravel_index(np.argmax(values),values.shape)[1]]),
            largest_absolute_transfer=float(abs(response).max()),
            maximum_transfer_change_from_previous=float(abs(response-previous).max()) if previous is not None else None))
        np.savez_compressed(out/f"laser_symbol_dt{dt:g}.npz",frequency_hz=f,epochs=epochs,transfer=response,relative_covariance_eigenvalue=values)
        previous=response;write_json(out/"symbol_controls.json",rows);print(rows[-1],flush=True)
    # More precision controls the sum only; native geometry is held fixed.
    t=epochs[0]+np.arange(-256,257)*args.geometry_dt[-1];delays=link_delays_from_lisaorbits(orbit,t)
    derivatives=None if args.derivatives=="gradient" else {key:np.polynomial.Chebyshev.fit(t,value,9).deriv()(t) for key,value in delays.items()}
    fine=moving_primary_laser_transfer(f,t,delays,[256],precision_digits=90,
        delay_derivatives=derivatives,rate_mode=args.rate_mode)[0]
    precision_error=float(abs(fine-previous[0]).max())
    failures=[] if max(r["maximum_generalized_laser_over_secondary"] for r in rows)<=1e-5 else ["Physical symbol exceeds selected residual allowance"]
    write_json(out/"validation_summary.json",dict(status="passed_selected_laser_symbol_controls" if not failures else "discrepancy_requires_investigation",
        failures=failures,production_certified=False,epochs=args.epochs,frequencies=len(f),
        precision_60_to_90_digits_absolute_transfer_change=precision_error,
        limitations=["Local adiabatic symbol, not exact frequency-bin covariance", "Float64 orbit/delay/rate inputs remain",
            "Instrument interpolation, numerical arithmetic artifacts and filter transients excluded", "N1-12 locking only"]))
    if failures:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=16)
    p.add_argument("--geometry-dt",type=float,nargs="+",default=[600.,300.,150.])
    p.add_argument("--derivatives",choices=["gradient","chebyshev"],default="gradient")
    p.add_argument("--rate-mode",choices=["native_first_order","chain_product"],default="native_first_order")
    run(p.parse_args())
