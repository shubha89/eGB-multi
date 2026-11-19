import numpy as np
import pytest
from xtdi.optimized._interpolation import apply_tiled
from xtdi.delay import FractionalDelay,_apply


@pytest.mark.parametrize('order',[1,7,25,39,101])
@pytest.mark.parametrize('batch',[1,2,8])
@pytest.mark.parametrize('query_size',[1,5,257,516])
def test_tiled_kernel_preserves_all_sample_arithmetic(order,batch,query_size):
    rng=np.random.default_rng(72981);size=257;half=(order+1)//2
    values=rng.normal(size=(batch,size));fraction=rng.random(query_size)
    stencil=FractionalDelay.prepare(fraction,query_size,order)
    # Include complete interior tiles, partial boundary tiles, far-clipped
    # centers and arbitrary Q != N. The last tile exercises the scalar tail.
    centers=rng.integers(-half-1,size+half,size=query_size,dtype=np.int64)
    reference=_apply(values,centers,stencil.fractions,stencil.scales,stencil.side_coefficients)
    actual=apply_tiled(values,centers,stencil.fractions,stencil.scales,stencil.side_coefficients)
    np.testing.assert_array_equal(actual,reference)


@pytest.mark.parametrize('order',[25,39,101])
def test_interior_tiled_path_is_identical_for_shared_source_weights(order):
    rng=np.random.default_rng(988);size=2048;query_size=1027
    fraction=rng.random(query_size);stencil=FractionalDelay.prepare(fraction,query_size,order)
    centers=np.arange(query_size,dtype=np.int64)+200
    data=rng.normal(size=(4,size))
    np.testing.assert_array_equal(
        apply_tiled(data,centers,stencil.fractions,stencil.scales,stencil.side_coefficients),
        _apply(data,centers,stencil.fractions,stencil.scales,stencil.side_coefficients))
