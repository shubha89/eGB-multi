import math
import pytest
from egb_multi.eccentric import EccentricBinaryParams
from egb_multi.stride import annual_pm_grid, recommend_cadence
from egb_multi import stride


@pytest.mark.parametrize("frequency,expected", [
    (1, (None, None, None, 20)), (3, (55, 15, 10, 5)),
    (6, (25, 10, 5, 5)), (10, (15, 5, 2.5, 2.5)),
])
def test_waveform_calibration(frequency, expected):
    for ecc, dt in zip((0., .1, .19, .3), expected, strict=True):
        choice = recommend_cadence(frequency, ecc)
        assert choice.measured_point and choice.cadence_s == dt


def test_prior_and_noise_criterion():
    region = recommend_cadence((3, 6), (.1, .19))
    assert region.cadence_s == 5 and len(region.checked_points) == 5
    assert not region.measured_point
    assert recommend_cadence(3, .19, criterion="likelihood").cadence_s == 45
    assert recommend_cadence(3, .19, criterion="likelihood", snr=1000).cadence_s < 45
    with pytest.raises(ValueError):
        recommend_cadence((1, 11), .19)


def test_grid_and_unresolved_source(monkeypatch):
    src = EccentricBinaryParams(math.pi*.003, .19)
    assert annual_pm_grid(src).samples == 12_623_040
    grid = annual_pm_grid(src, cadence_s="waveform")
    assert grid.cadence_s == 10 and grid.samples == 3_155_760
    monkeypatch.setattr(stride, "sampled_pm_uniform", lambda *args, **kwargs: args)
    assert grid.sampled_source(src, start_s=5000, padding_s=501) == (src, 4490, 10, grid.samples+102)
    unresolved = EccentricBinaryParams(math.pi*.001, .1)
    with pytest.raises(ValueError, match="No calibrated cadence"):
        annual_pm_grid(unresolved, cadence_s="waveform")
    for invalid in (True, -1., float("nan"), "unknown"):
        with pytest.raises(ValueError):
            annual_pm_grid(src, cadence_s=invalid)


def test_extended_likelihood_map():
    for frequency, ecc, expected in [(1, 0., 240), (1, .1, 155), (1, .19, 110),
                                     (1, .3, 80), (2, .05, 100), (3, 0., 105)]:
        source = EccentricBinaryParams(math.pi * frequency * .001, ecc)
        assert annual_pm_grid(source, cadence_s="likelihood").cadence_s == expected
    assert recommend_cadence(1, (0., .1), criterion="likelihood").cadence_s == 155
    assert recommend_cadence(1, .1, criterion="likelihood", snr=200.).cadence_s == 140
