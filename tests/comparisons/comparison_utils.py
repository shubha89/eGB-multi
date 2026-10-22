"""Serialization and unweighted two-polarization comparison helpers."""
from pathlib import Path
import hashlib
import json
import numpy as np

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def pair_stats(a,b):
    aa=float(np.sum(a*a));bb=float(np.sum(b*b));ab=float(np.sum(a*b))
    return dict(raw_relative_l2=float(np.linalg.norm(a-b)/np.sqrt(aa)),
        real_two_polarization_overlap=ab/np.sqrt(aa*bb),norm_ratio=np.sqrt(bb/aa),
        sign_combinations={f'{p},{q}':float(np.linalg.norm(a-np.array([p,q])[:,None]*b)/np.sqrt(aa)) for p,q in ((1,1),(1,-1),(-1,1),(-1,-1))})


def make_psd(frozen):
    """Load the recorded annual A/E covariance with its original convention."""
    from egb_multi.comparison import AEPSD, TDIConvention
    root = Path(__file__).resolve().parents[2]
    campaign = root / 'tests/paper/reference_data/outputs/referee/task_2_9_20261012_production'
    path = campaign / 'shared/annual_covariance.npz'
    manifest = json.loads((campaign / 'manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['combined_covariance_sha256']:
        raise ValueError('Frozen combined covariance changed')
    with np.load(path) as data:
        frequency, covariance = data['frequency_hz'], data['covariance']
    return AEPSD(frequency, {ch: covariance[:, i, i].real for i, ch in enumerate(('A', 'E'))},
        TDIConvention(**frozen['tdi_convention']), dict(sidedness='one_sided', scaling='density',
            noise_model='frozen_annual_moving_secondary_plus_primary_ideal_reference',
            CSD_convention='S_AE=E[tilde_A*conj(tilde_E)]'), covariance[:, 0, 1])
