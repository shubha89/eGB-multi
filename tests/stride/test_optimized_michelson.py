import numpy as np
import pytest
from xtdi import prepare_michelson
from xtdi.optimized import apply_michelson as apply_tiled_plan


@pytest.mark.parametrize('generation',[1,2])
@pytest.mark.parametrize('order',[25,101])
@pytest.mark.parametrize('unit',['phase','frequency'])
@pytest.mark.parametrize('batch',[1,4])
def test_tiled_application_preserves_full_stage_graph(generation,order,unit,batch):
    rng=np.random.default_rng(8119);size=1027;times=np.arange(size)*2.5
    delays=np.stack([8.2+i*.01+1e-5*np.sin(times/100.+i) for i in range(6)])
    plan=prepare_michelson(times,delays,generation=generation,measurement_order=order,delay_order=7)
    links=rng.normal(size=(batch,6,size))
    actual,stages=apply_tiled_plan(plan,links,unit=unit,return_stages=True)
    expected,reference=plan.apply(links,unit=unit,return_stages=True)
    np.testing.assert_array_equal(actual,expected)
    for name in stages:
        for a,b in zip(stages[name],reference[name],strict=True):np.testing.assert_array_equal(a,b)
