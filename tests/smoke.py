"""Installed-package smoke: native circular/PM responses and spectral agreement.

Short software verification only. No annual convergence, population coverage,
instrument calibration or inference claim. Generated waveforms stay in RAM.
"""
import argparse
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
import egb_multi
import egb_jax_eccentric
from egb_multi.eccentric import EccentricBinaryParams, peters_mathews_derivatives
from egb_multi.external import default_lisaorbits
from egb_multi.comparison import TDIConvention, AEPSD, AEInnerProduct
from egb_multi.native_validation import ChunkedNativeResponse
from egb_multi.circular_envelope import InterpolatedCircularTDI
from egb_multi.circular_spectral import CircularSpectralTDI
from egb_multi.circular_fit import circular_coefficients


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-wheel-install", action="store_true")
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    started=time.perf_counter()
    imported=Path(egb_multi.__file__).resolve()
    if args.require_wheel_install and not imported.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Expected an installed wheel inside this environment: " + str(imported))
    if egb_jax_eccentric.EccentricBinaryParams is not EccentricBinaryParams:
        raise RuntimeError("Legacy namespace uses a different source implementation")
    convention=TDIConvention(2,101,7,"analytic_esa_trailing_BME_J2000","native_analytic_retarded_delays")
    times=np.arange(16384)*2.5
    orbit=default_lisaorbits("esa-trailing")
    source=EccentricBinaryParams(mean_motion=np.pi*.001,eccentricity=0.,m1_solar=.5,m2_solar=.5,
        inclination=.8,beta=.2,lambda_=-1.,psi=.4,phi0=.2)
    fdot=float(peters_mathews_derivatives(source.mean_motion,0.,source.total_mass_kg,source.symmetric_mass_ratio)[0]/np.pi)
    source=replace(source,fdot=fdot)
    response=ChunkedNativeResponse(orbit,times,convention,block_samples=16384,cached_blocks=1)
    circular=response(source,"fixed")
    pm=response(replace(source,eccentricity=.1,fdot=0.),"peters_mathews",pm_ode_rtol=1e-12,pm_ode_phase_atol=1e-10)
    if any(not np.all(np.isfinite(pm.channels[ch])) or not np.any(pm.channels[ch]) for ch in ("A","E")):
        raise RuntimeError("PM response is not finite/nonzero")
    unit=AEPSD([0.,.2],{ch:[1.,1.] for ch in ("A","E")},convention,
        dict(sidedness="one_sided",scaling="density",noise_model="unit_PSD_software_smoke_only"))
    metric=AEInnerProduct(circular,unit,(1e-5,.08))
    interpolated=InterpolatedCircularTDI(orbit,times,convention,envelope_intervals=1024)
    basis=CircularSpectralTDI(interpolated,metric).prepared_basis(source)
    coeff=circular_coefficients(np.cos(source.inclination),source.psi,0.)
    prepared=np.einsum("i,ijk->jk",coeff,basis.prepared)
    difference=metric.reference-prepared
    relative=float(np.sqrt(metric.inner(difference,difference)/metric.reference_norm))
    if relative>5e-7:
        raise RuntimeError(f"Native/spectral short-template error too large: {relative}")
    result=dict(status="passed_installed_package_smoke",version=version("egb-multi"),python=sys.version,
        executable=sys.executable,package_file=str(imported),legacy_package_file=egb_jax_eccentric.__file__,
        legacy_source_class_shared=True,samples=len(times),dt_s=2.5,tdi_generation=2,interpolation_orders=[101,7],
        circular_native_spectral_relative_error=relative,pm_response_finite_nonzero=True,
        versions={n:version(n) for n in ("numpy","scipy","jax","jaxlib","pytdi","lisaorbits")},
        waveforms_saved=False,wall_s=time.perf_counter()-started,
        scope="Short installed-wheel software check using unit PSD; not annual scientific validation")
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
