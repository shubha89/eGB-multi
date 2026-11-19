from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from xtdi import LINK_LABELS, StreamedMichelson

from test_michelson import reference_data, reference_xyz


def delays(t):
    return {k:8.2+.03*i+.02*np.sin(t/1800+i) for i,k in enumerate(LINK_LABELS)}


def links(t):
    return np.array([np.sin(.018*t+i)+.2*np.cos(.047*t-i) for i in range(6)])


@pytest.mark.parametrize("backend", ["numba","jax"])
def test_streamed_matches_contiguous_reference_and_has_bounded_cache(backend):
    import contextlib
    if backend == "jax":
        jax = pytest.importorskip("jax")
        context = jax.enable_x64()
    else:
        context = contextlib.nullcontext()
    with context:
        t = 321000.+np.arange(1673)*2.5
        stream = StreamedMichelson(t,delays,block_samples=257,backend=backend,
                                   cache_bytes=2*1024**2)
        actual = stream.apply_provider(links)
        padded = t[0]+np.arange(-stream.halo,t.size+stream.halo)*2.5
        expected = reference_xyz(reference_data(padded,delays(padded)),
            dict(zip(LINK_LABELS,links(padded))),2,101,7)[:,stream.halo:-stream.halo]
        assert np.max(np.abs(actual-expected))/np.max(np.abs(expected)) < 1e-10
        np.testing.assert_allclose(stream.apply_provider(lambda tt: 2*links(tt)),2*actual,rtol=1e-12,atol=1e-13)
        assert stream.cached_bytes <= stream.cache_bytes
        stream.clear_cache()
        assert stream.cached_bytes == 0
        with pytest.raises(FrozenInstanceError):
            stream.measurement_order = 3


def test_bound_and_output_allocation_guard():
    t = np.arange(1000)*2.5
    stream = StreamedMichelson(t,delays,block_samples=127,cache_bytes=0)
    with pytest.raises(ValueError,match="memory"):
        stream.apply_provider(links,max_output_bytes=100)
    assert stream.cached_bytes == 0
    with pytest.raises(ValueError,match="bound"):
        StreamedMichelson(t,delays,delay_bound_s=1.).apply_provider(links)
