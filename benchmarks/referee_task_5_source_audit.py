"""Record source-model capabilities and fdot dispatch for referee Task 5.1.

These are short source-level sensitivity checks, not fitting-factor searches or
new astrophysical models. All waveform calls use this checkout's eGB package.
"""

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from egb_multi.constants import C_M_PER_S, G_M3_KG_S2
from egb_multi.eccentric import EccentricBinaryParams, eccentric_complex_strain
from egb_multi.eccentric_jax import eccentric_complex_strain_batch_jax, eccentric_links_jax, precompute_jax_link_geometry
from egb_multi.geometry import lisa_orbit


def difference(a, b):
    return dict(relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                bit_identical=bool(np.array_equal(a,b)))


def run(output):
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError('Keep this audit in the eGB checkout')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = EccentricBinaryParams(mean_motion=np.pi*1e-3, eccentricity=.1,
        m1_solar=.5, m2_solar=.5, inclination=.8, phi0=.2, t0=0.)
    prescribed = replace(source, fdot=1e-11)
    times = np.linspace(0., 100000., 2048)
    records = {}
    for mode in ('newtonian','1pn'):
        baseline = eccentric_complex_strain(source, times, physics_mode=mode)
        shifted = eccentric_complex_strain(prescribed, times, physics_mode=mode)
        jax_values = eccentric_complex_strain_batch_jax([source,prescribed], times, physics_mode=mode)
        records[mode] = dict(numpy_fdot_sensitivity=difference(baseline,shifted),
            jax_fdot_sensitivity=difference(jax_values[0],jax_values[1]),
            zero_fdot_numpy_jax_agreement=difference(baseline,jax_values[0]))
    pm_base = eccentric_complex_strain(source, times, evolution_mode='peters_mathews')
    pm_shift = eccentric_complex_strain(prescribed, times, evolution_mode='peters_mathews')
    records['numpy_PM_fdot_sensitivity'] = difference(pm_base,pm_shift)
    geometry = precompute_jax_link_geometry(lisa_orbit(times[:16]))
    try:
        eccentric_links_jax(prescribed,geometry,batch_size=1,evolution_mode='peters_mathews')
    except ValueError as error:
        rejection = str(error)
    else:
        raise AssertionError('Expected explicit non-fixed/fdot incompatibility')
    circular = replace(prescribed,eccentricity=0.)
    actual = eccentric_complex_strain_batch_jax(circular,times,physics_mode='newtonian')[0]
    dt = times-circular.t0
    omega = circular.mean_motion+np.pi*circular.fdot*dt
    phase = circular.mean_motion*dt+.5*np.pi*circular.fdot*dt**2+circular.phi0
    gm = G_M3_KG_S2*circular.total_mass_kg
    scale = G_M3_KG_S2*circular.total_mass_kg*circular.symmetric_mass_ratio/(C_M_PER_S**4*circular.distance_m)
    velocity_squared = (gm*omega)**(2./3.)
    cos_i = np.cos(circular.inclination)
    hplus = -2*scale*(1+cos_i**2)*velocity_squared*np.cos(2*phase)
    hcross = -4*scale*cos_i*velocity_squared*np.sin(2*phase)
    analytic = hplus-1j*hcross
    analytic_check = difference(analytic,actual)
    assert analytic_check['relative_l2'] < 1e-10
    assert all(records[m]['numpy_fdot_sensitivity']['bit_identical'] for m in ('newtonian','1pn'))
    assert records['numpy_PM_fdot_sensitivity']['bit_identical']
    assert all(records[m]['jax_fdot_sensitivity']['relative_l2'] > .01 for m in ('newtonian','1pn'))
    source_files = ['src/egb_multi/eccentric.py','src/egb_multi/eccentric_jax.py',
        'src/egb_multi/native_validation.py','src/egb_multi/circular_envelope.py',
        'src/egb_multi/circular_search.py','src/egb_multi/sources.py','src/egb_multi/wfgb.py',
        'reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex',
        'reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.bib','README.md',
        'benchmarks/referee_task_5_source_audit.py']
    result = dict(status='source_contract_audit_completed',created_utc=datetime.now(timezone.utc).isoformat(),
        package_checkout=str(ROOT),package_file=__import__('egb_multi').__file__,
        source_parameters=[f.name for f in fields(source)],source=asdict(source),
        prescribed_source=asdict(prescribed),times=dict(start_s=0.,end_s=100000.,samples=2048),
        prescribed_fdot_scope='Deliberately large sensitivity input, not a realistic stellar/environmental injection',
        fdot_dispatch_controls=records,nonfixed_link_fdot_rejection=rejection,
        newtonian_circular_prescribed_chirp_analytic_agreement=analytic_check,
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
        wall_s=time.perf_counter()-started,
        limitations=['Source-level diagnostic, not a noise-weighted overlap or parameter-recovery test',
            'NumPy/JAX fdot asymmetry is documented, not repaired',
            'No tidal, rotational, mass-transfer or tertiary terms are added'])
    (output/'source_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],fdot_dispatch_controls=records,
        analytic_agreement=analytic_check,nonfixed_link_fdot_rejection=rejection,
        wall_s=result['wall_s']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/referee/task_5_1_5_2_20261012')
    run(parser.parse_args().output)
