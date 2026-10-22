import numpy as np
import pytest

from egb_multi.laser_symbol import primary_laser_polynomials, moving_primary_laser_transfer
from egb_multi.tdi import LINK_LABELS


def test_primary_path_substitution_cancels_for_unequal_static_arms():
    pytest.importorskip("pytdi")
    pytest.importorskip("mpmath")
    # Collapse commuting constant delays analytically. Equal paths have exact
    # coefficient cancellation, independently of exponentials and numerics.
    for combination in primary_laser_polynomials():
        groups={}
        for coefficient,path in combination.components["primary"]:
            counts=tuple(path.count(f"D_{label}")-path.count(f"A_{label}") for label in LINK_LABELS)
            groups[counts]=groups.get(counts,0)+coefficient
        assert all(value==0 for value in groups.values())
    t=np.arange(512)*600.
    delays={label:np.full(len(t),8.1+.07*i) for i,label in enumerate(LINK_LABELS)}
    response=moving_primary_laser_transfer([2e-5,.001,.03,.04],t,delays,[256])
    assert np.max(abs(response))<2e-13


def test_primary_symbol_rejects_unimplemented_locking():
    with pytest.raises(ValueError,match="N1-12"):
        primary_laser_polynomials(locking="N2-12")


def test_chain_product_matches_independent_affine_delay_composition():
    mp=pytest.importorskip("mpmath")
    pytest.importorskip("pytdi")
    t=np.arange(512)*5.
    intercept={label:8.1+.07*i for i,label in enumerate(LINK_LABELS)}
    slope={label:(i-2)*2e-5 for i,label in enumerate(LINK_LABELS)}
    delays={label:intercept[label]+slope[label]*t for label in LINK_LABELS}
    derivatives={label:np.full(len(t),slope[label]) for label in LINK_LABELS}
    f=[.001,.03]
    actual=moving_primary_laser_transfer(f,t,delays,[256],delay_derivatives=derivatives,rate_mode="chain_product")[0]
    grouped=moving_primary_laser_transfer(f,t,delays,[256],delay_derivatives=derivatives,rate_mode="chain_product",summation="grouped_numpy")[0]
    np.testing.assert_allclose(grouped,actual,atol=2e-20,rtol=2e-9)
    expected=[]
    with mp.workdps(60):
        for frequency in f:
            xyz=[]
            for combination in primary_laser_polynomials():
                terms=[]
                for coefficient,path in combination.components["primary"]:
                    query=mp.mpf(float(t[256]));factor=mp.mpf(1)
                    for atom in path:
                        label=atom[2:];rate=1-mp.mpf(slope[label])
                        query=rate*query-mp.mpf(intercept[label]);factor*=rate
                    terms.append(coefficient*factor*mp.exp(2j*mp.pi*frequency*(query-t[256])))
                xyz.append(mp.fsum(terms))
            expected.append([complex((xyz[2]-xyz[0])/mp.sqrt(2)),complex((xyz[0]-2*xyz[1]+xyz[2])/mp.sqrt(6))])
    np.testing.assert_allclose(actual,expected,atol=3e-14,rtol=1e-7)
