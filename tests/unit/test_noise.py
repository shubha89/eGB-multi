import numpy as np
import pytest

from egb_multi.noise import (
    aet_from_xyz,
    estimate_welch_psd,
    fitting_factor_fixed_grid,
    karnesis_2021_cgb_confusion_psd,
    normalized_overlap_fd,
    rfft_with_dt,
)


def test_aet_from_xyz_shapes():
    xyz = {
        "X": np.ones(8),
        "Y": np.arange(8, dtype=float),
        "Z": -np.ones(8),
    }

    aet = aet_from_xyz(xyz)

    assert set(aet) == {"A", "E", "T"}
    assert aet["A"].shape == (8,)
    assert np.allclose(aet["T"], (xyz["X"] + xyz["Y"] + xyz["Z"]) / np.sqrt(3.0))


def test_estimate_welch_psd_returns_one_sided_density():
    dt = 1.0
    t = np.arange(1024) * dt
    series = np.sin(2.0 * np.pi * 0.05 * t)

    freq, psd = estimate_welch_psd(series, dt, nperseg=256)

    assert freq.shape == psd.shape
    assert freq[0] == 0.0
    assert np.all(psd >= 0.0)


def test_normalized_overlap_and_fitting_factor_identical_signal():
    dt = 1.0
    t = np.arange(1024) * dt
    series = np.sin(2.0 * np.pi * 0.05 * t)
    fd = rfft_with_dt(series, dt)
    psd = np.ones_like(fd, dtype=float)
    df = 1.0 / (series.size * dt)

    assert np.isclose(normalized_overlap_fd(fd, fd, psd, df), 1.0)
    assert np.isclose(fitting_factor_fixed_grid(fd, fd, psd, df), 1.0)


def test_karnesis_2021_cgb_confusion_psd_positive_in_band():
    f = np.array([0.0, 1.0e-4, 1.0e-3, 1.0e-2])

    psd = karnesis_2021_cgb_confusion_psd(f, t_obs_years=4.0)

    assert psd[0] == 0.0
    assert np.all(psd[1:] >= 0.0)
    assert psd[2] > psd[3]


def test_instrument_noise_adapter_preserves_local_metrology():
    from types import SimpleNamespace
    from egb_multi.noise import pytdi_data_from_lisainstrument_result
    from egb_multi.tdi import LINK_LABELS

    t = np.arange(128.)
    sci = {label: np.full(t.size, i+1.) for i, label in enumerate(LINK_LABELS)}
    tmi = {label: 10.*value for label, value in sci.items()}
    ref = {label: 100.*value for label, value in sci.items()}
    mprs = {label: np.full(t.size, 8.3) for label in LINK_LABELS}
    result = SimpleNamespace(t=t, mosa_data=SimpleNamespace(sci_carrier_fluctuations=sci,
        tmi_carrier_fluctuations=tmi, ref_carrier_fluctuations=ref, mprs=mprs))
    full = pytdi_data_from_lisainstrument_result(result)
    legacy = pytdi_data_from_lisainstrument_result(result, include_local_metrology=False)
    for label in LINK_LABELS:
        np.testing.assert_array_equal(full.measurements[f'sci_{label}'], sci[label])
        np.testing.assert_array_equal(full.measurements[f'tmi_{label}'], tmi[label])
        np.testing.assert_array_equal(full.measurements[f'ref_{label}'], ref[label])
        assert np.all(legacy.measurements[f'tmi_{label}'] == 0.)
        assert np.all(legacy.measurements[f'ref_{label}'] == 0.)


def test_default_noise_projection_suppresses_laser_only_interior_band():
    pytest.importorskip('lisainstrument')
    pytest.importorskip('pytdi')
    from egb_multi.noise import simulate_lisainstrument_tdi_noise
    noise=simulate_lisainstrument_tdi_noise(size=16384,dt=5.,seed=20261007,
        generation=2,enabled_noises=('laser',))
    # The static unequal-arm model has no flexing or clock errors. Exclude
    # samples without all nested delays and interpolation support.
    halo=int(np.ceil(8*max(np.max(d) for d in noise.delays.values())/noise.dt))+4*46
    f,reference=estimate_welch_psd(noise.links['12'][halo:-halo],noise.dt,nperseg=4096)
    band=(f>=.003)&(f<=.025)
    for channel in ('A','E'):
        _,residual=estimate_welch_psd(noise.aet[channel][halo:-halo],noise.dt,nperseg=4096)
        assert np.quantile(residual[band]/reference[band],.95) < 1e-15
